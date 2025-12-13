#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import joblib

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.vq_vae import VQVAE
from dataset import CurveDataset, pad_collate
'''
Examples:
python scripts/probe_pdb_unified.py \
 --pdb /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFD100-chainA \
        /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFDbeta-chainA \
        /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFD1000-chainA \
        /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFD3000-chainA
'''

PRP_ENV_PREFIX = Path(
    "/public/home/zhangyangroup/chengshiz/run/20250717_prp-data/prp-data/.pixi/envs/default"
)

CKPT_PATH = REPO_ROOT / "checkpoints" / "vq_s_gradient_ckpt_test11_15" / "epochepoch=549.ckpt"

TSNE_CACHE_PATH = REPO_ROOT / "latent_analysis" / "class1" / "tsne_cache_class1_len_between_1_80.npz"
UMAP_MODEL_PATH = REPO_ROOT / "latent_analysis" / "class1" / "umap_reducer_class1_len_between_1_80.pkl"

DEFAULT_OUTPUT_DIR = REPO_ROOT / "latent_analysis" / "class1" / "probe_cache"

HIDDEN_DIM = 512
NUM_LAYERS = 4
NUM_HEADS = 8
MAX_SEQ_LEN = 350
CODE_DIM = 128
LATENT_N_TOKENS = 48

USE_AMP = True
PRP_WORKERS = 16
KNN_K = 10


def strip_prefixes(state_dict, prefixes=("model.", "module.", "net.")):
    out = {}
    for k, v in state_dict.items():
        name = k
        for p in prefixes:
            if name.startswith(p):
                name = name[len(p):]
                break
        out[name] = v
    return out


def drop_quantizer_keys(state_dict):
    keys = [k for k in state_dict.keys() if k.startswith("quantizer.")]
    for k in keys:
        state_dict.pop(k, None)
    return state_dict


def get_array_or_none(cache, key):
    if key in cache.files:
        return cache[key]
    return None


def knn_interpolate_tsne(base_latents, base_2d, z_query, k=10, eps=1e-6):
    assert base_latents.ndim == 2 and base_2d.ndim == 2
    assert base_latents.shape[0] == base_2d.shape[0]
    assert base_latents.shape[1] == z_query.shape[0]

    diffs = base_latents - z_query[None, :]
    dists = np.linalg.norm(diffs, axis=1)

    k = max(1, min(k, base_latents.shape[0]))
    idx = np.argpartition(dists, k - 1)[:k]
    knn_dists = dists[idx]

    weights = 1.0 / (knn_dists + eps)
    weights = weights / weights.sum()

    coords = base_2d[idx]
    query_2d = (weights[:, None] * coords).sum(axis=0)
    return query_2d.astype(np.float32, copy=False)


def run_prp_process_multi_pdb(pdb_files, tmp_root: Path, workers: int = PRP_WORKERS):
    env_prefix = PRP_ENV_PREFIX.resolve()

    pdb_input_dir = tmp_root / "pdb_input"
    curves_out_dir = tmp_root / "curves_out"
    pdb_input_dir.mkdir(parents=True, exist_ok=True)
    curves_out_dir.mkdir(parents=True, exist_ok=True)

    mapping = {}
    used_names = set()

    for idx, pdb_path in enumerate(pdb_files):
        pdb_path = pdb_path.resolve()
        base = pdb_path.name
        dest_name = base
        if dest_name in used_names:
            dest_name = "{:04d}__{}".format(idx, base)
        used_names.add(dest_name)

        dst_pdb = pdb_input_dir / dest_name
        if pdb_path != dst_pdb:
            shutil.copy2(str(pdb_path), str(dst_pdb))

        dest_stem = dst_pdb.stem
        mapping[pdb_path] = {"stem": dest_stem}

    cmd = [
        "mamba",
        "run",
        "-p",
        str(env_prefix),
        "prp-data",
        "process",
        "--input",
        str(pdb_input_dir),
        "--output",
        str(curves_out_dir),
        "--workers",
        str(int(workers)),
        "--device",
        "cpu",
        "--metadata",
        "probe_metadata.json",
    ]
    print("[PRP] Running:", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print("[PRP] stdout:\n", result.stdout)
        print("[PRP] stderr:\n", result.stderr)
        raise RuntimeError("prp-data process failed with code {}".format(result.returncode))

    print("[PRP] process finished. stdout:\n", result.stdout)

    npy_files = [p for p in curves_out_dir.iterdir() if p.suffix == ".npy"]
    if not npy_files:
        raise RuntimeError("No .npy produced by prp-data under {}".format(curves_out_dir))

    npy_files_sorted = sorted(npy_files, key=lambda x: x.name)

    pdb_to_npy = {}
    for pdb_path, meta in mapping.items():
        stem = meta["stem"]
        candidate = curves_out_dir / (stem + ".npy")

        if candidate.is_file():
            pdb_to_npy[pdb_path] = candidate
            continue

        found = None
        for fn in npy_files_sorted:
            if fn.name.startswith(stem):
                found = fn
                break

        if found is None and npy_files_sorted:
            found = npy_files_sorted[0]

        if found is None:
            raise RuntimeError(
                "Could not find any .npy for PDB {} (stem={})".format(str(pdb_path), stem)
            )

        pdb_to_npy[pdb_path] = found

    print("[PRP] Resolved {} PDBs to curve npy files".format(len(pdb_to_npy)))
    return pdb_to_npy


@torch.no_grad()
def encode_single_batch_to_latent(model, x, mask, device, use_amp: bool):
    x = x.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)

    try:
        autocast_ctx = torch.amp.autocast(
            device_type="cuda",
            enabled=(use_amp and device.type == "cuda"),
        )
    except Exception:
        autocast_ctx = torch.cuda.amp.autocast(
            enabled=(use_amp and device.type == "cuda")
        )

    model.eval()
    with autocast_ctx:
        h_fuse, _, _ = model.encode(x, mask=mask)
        z_tok = model._tokenize_to_codes(h_fuse, mask)
        z_seq = z_tok.mean(dim=1)

    z = z_seq.cpu().numpy()[0]
    return z.astype(np.float32, copy=False)


def collect_pdb_files_and_groups(pdb_args):
    pdb_files = []
    pdb_groups = []

    for arg in pdb_args:
        root = Path(arg).resolve()
        if root.is_dir():
            group_name = root.name
            for fn in sorted(root.iterdir()):
                if fn.is_file() and fn.suffix.lower() == ".pdb":
                    pdb_files.append(fn)
                    pdb_groups.append(group_name)
        elif root.is_file():
            if root.suffix.lower() == ".pdb":
                group_name = root.stem
                pdb_files.append(root)
                pdb_groups.append(group_name)
            else:
                print("[Warn] Skip non-pdb file:", str(root))
        else:
            print("[Warn] Path not found, skip:", str(root))

    if not pdb_files:
        raise RuntimeError("No valid .pdb files found from: {}".format(pdb_args))

    return pdb_files, pdb_groups


def derive_group_name(pdb_args, pdb_files):
    if len(pdb_args) == 1:
        only = Path(pdb_args[0]).resolve()
        if only.is_dir():
            return only.name
        else:
            return only.stem
    else:
        return "multi_{}_pdbs".format(len(pdb_files))


def parse_args():
    p = argparse.ArgumentParser("Probe PDBs into latent tsne/umap background and save unified cache")
    p.add_argument(
        "--pdb",
        type=str,
        nargs="+",
        required=True,
        help="One or more PDB paths; each can be a file or a directory containing .pdb files",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Optional output directory for probe cache npz; default: under latent_analysis/classX/probe_cache",
    )
    p.add_argument(
        "--knn_k",
        type=int,
        default=KNN_K,
        help="K for KNN interpolation in t-SNE space",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not CKPT_PATH.is_file():
        raise FileNotFoundError("CKPT not found: {}".format(str(CKPT_PATH)))
    if not TSNE_CACHE_PATH.is_file():
        raise FileNotFoundError("Background cache .npz not found: {}".format(str(TSNE_CACHE_PATH)))
    if not UMAP_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "UMAP model .pkl not found: {}. Run visualize_latent_and_codebook2.py first.".format(
                str(UMAP_MODEL_PATH)
            )
        )

    pdb_files, pdb_groups = collect_pdb_files_and_groups(args.pdb)
    run_tag = derive_group_name(args.pdb, pdb_files)

    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device] Using:", device)
    print("[Probe] Total PDB files:", len(pdb_files))

    base_cache = np.load(str(TSNE_CACHE_PATH), allow_pickle=True)

    base_latents = base_cache["latents"].astype(np.float32, copy=False)
    base_tsne_2d = base_cache["tsne_2d"].astype(np.float32, copy=False)

    base_umap_2d = get_array_or_none(base_cache, "umap_2d")
    if base_umap_2d is not None:
        base_umap_2d = base_umap_2d.astype(np.float32, copy=False)

    base_lengths = get_array_or_none(base_cache, "lengths")
    base_helix_frac = get_array_or_none(base_cache, "helix_frac")
    base_sheet_frac = get_array_or_none(base_cache, "sheet_frac")
    base_loop_frac = get_array_or_none(base_cache, "loop_frac")
    base_labels = get_array_or_none(base_cache, "labels")
    base_cath_full = get_array_or_none(base_cache, "cath_full")
    base_rel_paths = get_array_or_none(base_cache, "rel_paths")
    base_cath_kept = get_array_or_none(base_cache, "cath_kept_classes")

    base_min_len = int(base_cache["min_len"]) if "min_len" in base_cache.files else -1
    base_max_len = int(base_cache["max_len"]) if "max_len" in base_cache.files else -1

    if base_cath_kept is not None:
        classes_list = [str(int(x)) for x in np.asarray(base_cath_kept).ravel().tolist()]
        classes_tag = "_".join(classes_list) if classes_list else "all"
    else:
        classes_tag = "all"

    print(
        "[Background] latents={}x{}, tsne_2d={}x{}".format(
            base_latents.shape[0],
            base_latents.shape[1],
            base_tsne_2d.shape[0],
            base_tsne_2d.shape[1],
        )
    )

    print("[Load] Loading UMAP model:", str(UMAP_MODEL_PATH))
    reducer = joblib.load(str(UMAP_MODEL_PATH))

    model = VQVAE(
        input_dim=6,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_seq_len=MAX_SEQ_LEN,
        use_vq=False,
        codebook_size=1,
        code_dim=CODE_DIM,
        label_smoothing=0.0,
        ss_tv_lambda=0.0,
        usage_entropy_lambda=0.0,
        xyz_align_alpha=0.7,
        dist_lambda=0.0,
        rigid_aug_prob=0.0,
        pairwise_sample_k=32,
        noise_warmup_steps=0,
        max_noise_std=0.0,
        reinit_dead_codes=False,
        reinit_prob=0.0,
        dead_usage_threshold=0,
        codebook_init_path="",
        latent_tokens=int(LATENT_N_TOKENS),
        tokenizer_heads=NUM_HEADS,
        tokenizer_layers=2,
        tokenizer_dropout=0.1,
        print_init=False,
    ).to(device)

    ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = strip_prefixes(state)
    state = drop_quantizer_keys(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[Load] missing={} unexpected={}".format(len(missing), len(unexpected)))

    probe_latents = []
    probe_tsne_coords = []
    probe_umap_coords = []
    probe_pdb_paths = []
    probe_groups = []

    with tempfile.TemporaryDirectory(prefix="probe_pdbs_unified_") as tmp_root_str:
        tmp_root = Path(tmp_root_str).resolve()
        pdb_to_npy = run_prp_process_multi_pdb(pdb_files, tmp_root, workers=PRP_WORKERS)

        for idx, (pdb_path, group_name_i) in enumerate(zip(pdb_files, pdb_groups)):
            pdb_name = pdb_path.name
            curve_npy_path = pdb_to_npy[pdb_path]
            npy_dir = curve_npy_path.parent
            npy_base = curve_npy_path.name

            print(
                "\n[Probe] {} / {}: {} (group: {}) (curve npy: {})".format(
                    idx + 1,
                    len(pdb_files),
                    pdb_name,
                    group_name_i,
                    str(curve_npy_path),
                )
            )

            list_path = tmp_root / ("probe_list_{}.txt".format(idx))
            with list_path.open("w") as f:
                f.write(npy_base + "\n")

            ds = CurveDataset(
                npy_dir=str(npy_dir),
                list_path=str(list_path),
                train=False,
            )
            loader = DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=pad_collate,
                drop_last=False,
            )

            batch = next(iter(loader))
            if isinstance(batch, (list, tuple)):
                x, mask = batch
            else:
                x, mask = batch, None
                mask = torch.ones((x.size(0), x.size(1)), dtype=torch.bool)

            print(
                "[Probe] Loaded curve: x shape={}, mask shape={}".format(
                    tuple(x.shape), tuple(mask.shape)
                )
            )

            z_query = encode_single_batch_to_latent(
                model=model,
                x=x,
                mask=mask,
                device=device,
                use_amp=bool(USE_AMP),
            )
            print(
                "[Probe] Latent dim={}, first 5 values {}".format(
                    z_query.shape[0], z_query[:5]
                )
            )

            tsne_coord = knn_interpolate_tsne(
                base_latents=base_latents,
                base_2d=base_tsne_2d,
                z_query=z_query,
                k=int(args.knn_k),
            )
            print(
                "[Probe] t-SNE coord: ({:.3f}, {:.3f})".format(
                    float(tsne_coord[0]), float(tsne_coord[1])
                )
            )

            z_query_reshaped = z_query.reshape(1, -1)
            umap_2d = reducer.transform(z_query_reshaped)[0]
            umap_coord = umap_2d.astype(np.float32, copy=False)
            print(
                "[Probe] UMAP coord: ({:.3f}, {:.3f})".format(
                    float(umap_coord[0]), float(umap_coord[1])
                )
            )

            probe_latents.append(z_query.astype(np.float32, copy=False))
            probe_tsne_coords.append(tsne_coord)
            probe_umap_coords.append(umap_coord)
            probe_pdb_paths.append(str(pdb_path))
            probe_groups.append(str(group_name_i))

    probe_latents = np.stack(probe_latents, axis=0).astype(np.float32, copy=False)
    probe_tsne_coords = np.stack(probe_tsne_coords, axis=0).astype(np.float32, copy=False)
    probe_umap_coords = np.stack(probe_umap_coords, axis=0).astype(np.float32, copy=False)
    probe_pdb_paths = np.array(probe_pdb_paths, dtype=object)
    probe_groups = np.array(probe_groups, dtype=object)

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_cache_name = "probe_cache_class{}_len_between_{}_{}_{}.npz".format(
        classes_tag,
        base_min_len,
        base_max_len,
        run_tag,
    )
    probe_cache_path = out_dir / probe_cache_name

    np.savez(
        str(probe_cache_path),
        base_latents=base_latents,
        base_tsne_2d=base_tsne_2d,
        base_umap_2d=base_umap_2d,
        base_lengths=base_lengths,
        base_helix_frac=base_helix_frac,
        base_sheet_frac=base_sheet_frac,
        base_loop_frac=base_loop_frac,
        base_labels=base_labels,
        base_cath_full=base_cath_full,
        base_rel_paths=base_rel_paths,
        base_cath_kept_classes=base_cath_kept,
        base_cache_path=str(TSNE_CACHE_PATH),
        ckpt_path=str(CKPT_PATH),
        umap_model_path=str(UMAP_MODEL_PATH),
        probe_latents=probe_latents,
        probe_tsne_2d=probe_tsne_coords,
        probe_umap_2d=probe_umap_coords,
        probe_pdb_paths=probe_pdb_paths,
        probe_groups=probe_groups,
    )

    print("\n[Probe] Saved unified probe cache to:")
    print("  {}".format(str(probe_cache_path)))
    print("[Probe] Done.")


if __name__ == "__main__":
    main()
