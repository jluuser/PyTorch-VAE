#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Probe generated curves (or PDBs) into an existing latent background (t-SNE/UMAP).

This script supports two input modes:

(A) --curve_dir:
    Directly read curve .npy files from a directory (recommended for your current pipeline).
    The files should be compatible with CurveDataset (e.g., dict with "curve_coords"/"ss_one_hot",
    or ndarray [L,6] = [xyz, ss_one_hot]).

(B) --pdb:
    Convert PDB(s) to curve .npy using prp-data process (legacy / optional).

It encodes each curve using your AE model (stage1_ae) and projects:
- UMAP: reducer.transform(z)  (true out-of-sample)
- t-SNE: KNN interpolation on base_latents -> base_tsne_2d (approx, since TSNE cannot transform)

Outputs a unified probe cache .npz with:
- base_latents, base_tsne_2d, base_umap_2d, and metadata (copied from base cache)
- probe_latents, probe_tsne_2d, probe_umap_2d, probe_paths, probe_groups

python scripts/playground/probe_pdb_unified.py \
  --curve_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/results/decoded_curves_122_sigmoid_filtered \
  --recursive \
  --config configs/stage1_ae.yaml \
  --ckpt /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/checkpoints/aeot_sigmoid/epochepoch=epoch=089.ckpt \
  --base_cache /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/latent_analysis_ae_sigmoid/class1/tsne_cache_ae_tokens_mean_class1_len_between_1_80.npz \
  --umap_model /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/latent_analysis_ae_sigmoid/class1/umap_reducer_ae_tokens_mean_class1_len_between_1_80.pkl \
  --latent_rep tokens_mean \
  --knn_k 10 \
  --batch_size 128 \
  --num_workers 8 \
  --amp \
  --out_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/latent_analysis_ae_sigmoid/class1

python scripts/playground/probe_pdb_unified.py \
  --pdb \
    /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFD100-chainA \
    /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFDbeta-chainA \
    /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFD1000-chainA \
    /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFD3000-chainA \
  --prp_env_prefix /public/home/zhangyangroup/chengshiz/run/20250717_prp-data/prp-data/.pixi/envs/default \
  --prp_workers 16 \
  --config configs/stage1_ae.yaml \
  --ckpt /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/checkpoints/aeot_sigmoid/epochepoch=epoch=089.ckpt \
  --base_cache /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/latent_analysis_ae_sigmoid/class1/tsne_cache_ae_tokens_mean_class1_len_between_1_80.npz \
  --umap_model /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/latent_analysis_ae_sigmoid/class1/umap_reducer_ae_tokens_mean_class1_len_between_1_80.pkl \
  --latent_rep tokens_mean \
  --knn_k 10 \
  --batch_size 128 \
  --num_workers 8 \
  --amp \
  --out_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/latent_analysis_ae_sigmoid/class_pdb

"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import joblib

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import build_experiment_from_yaml
from dataset import CurveDataset, pad_collate


# -----------------------------
# Helpers: state_dict cleanup
# -----------------------------
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


# -----------------------------
# Helpers: cache reading
# -----------------------------
def get_array_or_none(cache, key):
    if key in cache.files:
        return cache[key]
    return None


# -----------------------------
# TSNE: KNN interpolation
# -----------------------------
def knn_interpolate_tsne_from_neighbors(base_2d, nn_idx, nn_dist, eps=1e-6):
    """
    base_2d: [N,2]
    nn_idx:  [B,K]
    nn_dist: [B,K]
    returns: [B,2]
    """
    nn_dist = np.asarray(nn_dist, dtype=np.float32)
    nn_idx = np.asarray(nn_idx, dtype=np.int64)

    # Handle exact match (dist == 0)
    exact = (nn_dist[:, 0] <= 1e-12)
    out = np.zeros((nn_idx.shape[0], 2), dtype=np.float32)

    if np.any(exact):
        out[exact] = base_2d[nn_idx[exact, 0]]

    non_exact = ~exact
    if np.any(non_exact):
        d = nn_dist[non_exact]
        idx = nn_idx[non_exact]
        w = 1.0 / (d + eps)
        w = w / np.sum(w, axis=1, keepdims=True)
        coords = base_2d[idx]  # [B,K,2]
        out[non_exact] = np.sum(w[:, :, None] * coords, axis=1).astype(np.float32, copy=False)

    return out


def build_knn_index(base_latents: np.ndarray, k: int):
    """
    Build a nearest neighbor index on base_latents.
    Uses sklearn if available; otherwise falls back to brute-force per batch.
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
        nn.fit(base_latents)
        return ("sklearn", nn)
    except Exception:
        return ("bruteforce", None)


def query_knn(index_type, index_obj, base_latents: np.ndarray, z: np.ndarray, k: int):
    """
    z: [B,D]
    returns (idx [B,K], dist [B,K])
    """
    if index_type == "sklearn":
        dist, idx = index_obj.kneighbors(z, n_neighbors=k, return_distance=True)
        return idx, dist

    # brute force
    B = z.shape[0]
    idx_out = np.zeros((B, k), dtype=np.int64)
    dist_out = np.zeros((B, k), dtype=np.float32)
    for i in range(B):
        diffs = base_latents - z[i:i+1]
        d = np.linalg.norm(diffs, axis=1)
        kk = max(1, min(k, base_latents.shape[0]))
        nn_idx = np.argpartition(d, kk - 1)[:kk]
        nn_dist = d[nn_idx]
        # sort by distance
        order = np.argsort(nn_dist)
        nn_idx = nn_idx[order]
        nn_dist = nn_dist[order]
        if nn_idx.shape[0] < k:
            pad_n = k - nn_idx.shape[0]
            nn_idx = np.pad(nn_idx, (0, pad_n), mode="edge")
            nn_dist = np.pad(nn_dist, (0, pad_n), mode="edge")
        idx_out[i] = nn_idx[:k]
        dist_out[i] = nn_dist[:k].astype(np.float32)
    return idx_out, dist_out


# -----------------------------
# Optional: PRP conversion (legacy)
# -----------------------------
def run_prp_process_multi_pdb(
    pdb_files: List[Path],
    tmp_root: Path,
    env_prefix: Path,
    workers: int = 16,
) -> Dict[Path, Path]:
    """
    Run prp-data process to convert multiple PDBs into curve .npy.
    Returns {original_pdb_path: produced_curve_npy_path}.
    """
    env_prefix = env_prefix.resolve()

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

        mapping[pdb_path] = {"stem": dst_pdb.stem}

    cmd = [
        "mamba", "run", "-p", str(env_prefix),
        "prp-data", "process",
        "--input", str(pdb_input_dir),
        "--output", str(curves_out_dir),
        "--workers", str(int(workers)),
        "--device", "cpu",
        "--metadata", "probe_metadata.json",
    ]
    print("[PRP] Running:", " ".join(cmd))
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        print("[PRP] stdout:\n", result.stdout)
        print("[PRP] stderr:\n", result.stderr)
        raise RuntimeError("prp-data process failed with code {}".format(result.returncode))

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
            raise RuntimeError("Could not find any .npy for PDB {}".format(str(pdb_path)))

        pdb_to_npy[pdb_path] = found

    print("[PRP] Resolved {} PDBs to curve npy files".format(len(pdb_to_npy)))
    return pdb_to_npy


# -----------------------------
# AE encoding: tokens_mean / tokens_flatten
# -----------------------------
@torch.no_grad()
def encode_batch_to_latents(
    model,
    x: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    latent_rep: str,
) -> np.ndarray:
    """
    Return latents as numpy float32:
      - tokens_mean:    [B, D]
      - tokens_flatten: [B, N*D]
    """
    x = x.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)

    try:
        autocast_ctx = torch.amp.autocast(
            device_type="cuda",
            enabled=(use_amp and device.type == "cuda"),
        )
    except Exception:
        autocast_ctx = torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda"))

    model.eval()
    with autocast_ctx:
        h_fuse, _, _ = model.encode(x, mask=mask)              # [B, L, H] (internals)
        z_tok = model._tokenize_to_codes(h_fuse, mask)         # [B, N, D]

        rep = str(latent_rep).lower()
        if rep in ("tokens_mean", "mean", "tok_mean"):
            z = z_tok.mean(dim=1)                              # [B, D]
        elif rep in ("tokens_flatten", "flatten", "tok_flatten"):
            z = z_tok.reshape(z_tok.size(0), -1)               # [B, N*D]
        else:
            raise ValueError(f"Unknown latent_rep={latent_rep}, use tokens_mean or tokens_flatten")

    return z.detach().float().cpu().numpy().astype(np.float32, copy=False)


def compute_batch_stats_from_x(x: torch.Tensor, mask: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Compute length and SS fractions from x[...,3:6] assuming one-hot SS.
    Returns numpy arrays.
    """
    ss = x[..., 3:]
    valid = mask.unsqueeze(-1).to(ss.dtype)
    ss_valid = ss * valid
    counts = ss_valid.sum(dim=1)  # [B,3]
    length = mask.sum(dim=1).to(torch.float32)  # [B]
    denom = torch.clamp(length.unsqueeze(-1), min=1.0)
    frac = (counts / denom).detach().cpu().numpy().astype(np.float32, copy=False)

    return {
        "lengths": length.detach().cpu().numpy().astype(np.float32, copy=False),
        "helix_frac": frac[:, 0],
        "sheet_frac": frac[:, 1],
        "loop_frac": frac[:, 2],
    }


# -----------------------------
# Input collection: curve_dir
# -----------------------------
def collect_curve_files_from_dir(curve_dir: Path, recursive: bool = True) -> List[Path]:
    if not curve_dir.is_dir():
        raise FileNotFoundError(f"curve_dir not found: {curve_dir}")
    if recursive:
        files = sorted([p for p in curve_dir.rglob("*.npy") if p.is_file()])
    else:
        files = sorted([p for p in curve_dir.glob("*.npy") if p.is_file()])
    if not files:
        raise RuntimeError(f"No .npy files found under: {curve_dir}")
    return files


def write_list_file(curve_dir: Path, curve_files: List[Path], list_path: Path) -> np.ndarray:
    """
    Write a list file containing RELATIVE paths under curve_dir, one per line.
    Returns rel_paths array.
    """
    rels = []
    with list_path.open("w") as f:
        for p in curve_files:
            rel = os.path.relpath(str(p), str(curve_dir))
            f.write(rel + "\n")
            rels.append(rel)
    return np.array(rels, dtype=object)


# -----------------------------
# Input collection: pdb args (legacy)
# -----------------------------
def collect_pdb_files_and_groups(pdb_args: List[str]) -> Tuple[List[Path], List[str]]:
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
                pdb_files.append(root)
                pdb_groups.append(root.stem)
            else:
                print("[Warn] Skip non-pdb file:", str(root))
        else:
            print("[Warn] Path not found, skip:", str(root))
    if not pdb_files:
        raise RuntimeError("No valid .pdb files found from: {}".format(pdb_args))
    return pdb_files, pdb_groups


def derive_run_tag(curve_dir: Optional[Path], pdb_args: Optional[List[str]], n_files: int) -> str:
    if curve_dir is not None:
        return curve_dir.name
    if pdb_args is not None and len(pdb_args) == 1:
        only = Path(pdb_args[0]).resolve()
        return only.name if only.is_dir() else only.stem
    return "multi_{}_items".format(n_files)


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Probe generated curves into latent background (AE) and save unified cache")

    # Input mode A: curves
    p.add_argument("--curve_dir", type=str, default="", help="Directory containing curve .npy files (recommended)")
    p.add_argument("--recursive", action="store_true", help="Recursively scan curve_dir for .npy files")
    p.add_argument("--group", type=str, default="", help="Optional group name for all curves (default: curve_dir name)")

    # Input mode B: PDBs (legacy)
    p.add_argument("--pdb", type=str, nargs="+", default=[], help="One or more PDB paths (file or dir)")
    p.add_argument("--prp_env_prefix", type=str, default="", help="Path to prp-data mamba env prefix (for PDB mode)")
    p.add_argument("--prp_workers", type=int, default=16, help="Workers for prp-data process (PDB mode)")

    # AE model
    p.add_argument("--config", type=str, required=True, help="Path to stage1_ae.yaml")
    p.add_argument("--ckpt", type=str, required=True, help="Path to AE checkpoint")

    # Background / base
    p.add_argument("--base_cache", type=str, required=True, help="Path to base tsne_cache_ae_*.npz")
    p.add_argument("--umap_model", type=str, required=True, help="Path to base umap_reducer_ae_*.pkl")
    p.add_argument("--latent_rep", type=str, default="", help="tokens_mean or tokens_flatten (default: infer from base cache if present)")
    p.add_argument("--knn_k", type=int, default=10, help="K for TSNE KNN interpolation")
    p.add_argument("--only_umap", action="store_true", help="Skip TSNE KNN interpolation (UMAP only)")

    # Runtime
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")

    # Output
    p.add_argument("--out_dir", type=str, default="", help="Output directory for probe cache npz")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
@torch.no_grad()
def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print("[Device] Using:", device)

    base_cache_path = Path(args.base_cache).resolve()
    umap_model_path = Path(args.umap_model).resolve()
    if not base_cache_path.is_file():
        raise FileNotFoundError(f"Base cache not found: {base_cache_path}")
    if not umap_model_path.is_file():
        raise FileNotFoundError(f"UMAP model not found: {umap_model_path}")

    base_cache = np.load(str(base_cache_path), allow_pickle=True)

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

    # Infer latent_rep if stored in cache
    latent_rep = str(args.latent_rep).strip()
    if not latent_rep:
        if "latent_rep" in base_cache.files:
            latent_rep = str(base_cache["latent_rep"].item())
        else:
            # fallback: infer by dimension only (best effort)
            latent_rep = "tokens_flatten" if base_latents.shape[1] > 512 else "tokens_mean"
    latent_rep = str(latent_rep).lower()
    print(f"[Base] latents shape={base_latents.shape}, tsne_2d shape={base_tsne_2d.shape}, latent_rep={latent_rep}")

    print("[Load] Loading UMAP model:", str(umap_model_path))
    reducer = joblib.load(str(umap_model_path))

    # -------------------------
    # Load AE model from YAML + ckpt
    # -------------------------
    print(f"[AE] Loading experiment from config: {args.config}")
    exp, _ = build_experiment_from_yaml(args.config)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state_dict = strip_prefixes(state_dict, prefixes=("model.", "module.", "net."))
    missing, unexpected = exp.model.load_state_dict(state_dict, strict=False)
    print(f"[AE] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    model = exp.model.to(device).eval()

    # -------------------------
    # Collect inputs: curve_dir or pdb
    # -------------------------
    curve_dir = Path(args.curve_dir).resolve() if args.curve_dir else None
    pdb_args = args.pdb if args.pdb else None

    if curve_dir is None and pdb_args is None:
        raise ValueError("You must provide either --curve_dir or --pdb")

    probe_files: List[Path] = []
    probe_groups: List[str] = []

    with tempfile.TemporaryDirectory(prefix="probe_unified_") as tmp_root_str:
        tmp_root = Path(tmp_root_str).resolve()

        if curve_dir is not None:
            # Direct curve mode
            curve_files = collect_curve_files_from_dir(curve_dir, recursive=bool(args.recursive))
            probe_files = curve_files

            group_name = args.group.strip() or curve_dir.name
            probe_groups = [group_name] * len(probe_files)

            list_path = tmp_root / "probe_curve_list.txt"
            rel_paths = write_list_file(curve_dir, probe_files, list_path)

            ds = CurveDataset(npy_dir=str(curve_dir), list_path=str(list_path), train=False)
            loader = DataLoader(
                ds,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.num_workers),
                pin_memory=(device.type == "cuda"),
                collate_fn=pad_collate,
                drop_last=False,
                persistent_workers=(int(args.num_workers) > 0),
            )

            probe_paths = np.array([str(p) for p in probe_files], dtype=object)
            probe_groups_arr = np.array(probe_groups, dtype=object)

        else:
            # PDB mode (legacy)
            if not args.prp_env_prefix:
                raise ValueError("PDB mode requires --prp_env_prefix")
            prp_env_prefix = Path(args.prp_env_prefix).resolve()

            pdb_files, pdb_groups = collect_pdb_files_and_groups(pdb_args)
            probe_files = pdb_files
            probe_groups = pdb_groups

            pdb_to_npy = run_prp_process_multi_pdb(
                pdb_files=pdb_files,
                tmp_root=tmp_root,
                env_prefix=prp_env_prefix,
                workers=int(args.prp_workers),
            )

            # Build a list for CurveDataset under prp output dir
            # We must use the directory where npy lives (curves_out)
            any_npy = next(iter(pdb_to_npy.values()))
            npy_dir = any_npy.parent

            list_path = tmp_root / "probe_pdb_list.txt"
            with list_path.open("w") as f:
                for pdb in pdb_files:
                    f.write(pdb_to_npy[pdb].name + "\n")

            ds = CurveDataset(npy_dir=str(npy_dir), list_path=str(list_path), train=False)
            loader = DataLoader(
                ds,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
                collate_fn=pad_collate,
                drop_last=False,
            )

            probe_paths = np.array([str(p) for p in pdb_files], dtype=object)
            probe_groups_arr = np.array(pdb_groups, dtype=object)

        # -------------------------
        # Encode all probe curves
        # -------------------------
        print(f"[Probe] Encoding {len(probe_files)} items ...")
        probe_latents_list = []
        probe_stats = {"lengths": [], "helix_frac": [], "sheet_frac": [], "loop_frac": []}

        for batch in loader:
            x, mask = batch
            # Ensure mask exists
            if mask is None:
                mask = torch.ones((x.size(0), x.size(1)), dtype=torch.bool)

            z = encode_batch_to_latents(
                model=model,
                x=x,
                mask=mask,
                device=device,
                use_amp=bool(args.amp),
                latent_rep=latent_rep,
            )
            probe_latents_list.append(z)

            st = compute_batch_stats_from_x(x, mask)
            for k in probe_stats.keys():
                probe_stats[k].append(st[k])

        probe_latents = np.concatenate(probe_latents_list, axis=0).astype(np.float32, copy=False)
        probe_lengths = np.concatenate(probe_stats["lengths"], axis=0).astype(np.float32, copy=False)
        probe_helix = np.concatenate(probe_stats["helix_frac"], axis=0).astype(np.float32, copy=False)
        probe_sheet = np.concatenate(probe_stats["sheet_frac"], axis=0).astype(np.float32, copy=False)
        probe_loop = np.concatenate(probe_stats["loop_frac"], axis=0).astype(np.float32, copy=False)

        print(f"[Probe] Latents shape={probe_latents.shape} (base dim={base_latents.shape[1]})")
        if probe_latents.shape[1] != base_latents.shape[1]:
            raise RuntimeError(
                f"Latent dim mismatch: probe_dim={probe_latents.shape[1]} vs base_dim={base_latents.shape[1]}.\n"
                f"Check --latent_rep and ensure base cache is built with the same representation."
            )

        # -------------------------
        # Project to UMAP
        # -------------------------
        print("[UMAP] Transforming probe latents ...")
        probe_umap_2d = reducer.transform(probe_latents).astype(np.float32, copy=False)

        # -------------------------
        # Project to TSNE (KNN interpolation)
        # -------------------------
        if bool(args.only_umap):
            probe_tsne_2d = None
            print("[t-SNE] Skipped (only_umap=True).")
        else:
            k = int(args.knn_k)
            k = max(1, min(k, base_latents.shape[0]))
            print(f"[t-SNE] KNN interpolation with k={k} ...")

            idx_type, idx_obj = build_knn_index(base_latents, k=k)

            # query in batches to reduce peak memory
            B = probe_latents.shape[0]
            bs = 2048
            tsne_out = []
            for i in range(0, B, bs):
                z_b = probe_latents[i:i+bs]
                nn_idx, nn_dist = query_knn(idx_type, idx_obj, base_latents, z_b, k=k)
                ts = knn_interpolate_tsne_from_neighbors(base_tsne_2d, nn_idx, nn_dist)
                tsne_out.append(ts)
            probe_tsne_2d = np.concatenate(tsne_out, axis=0).astype(np.float32, copy=False)
            print("[t-SNE] Done.")

    # -------------------------
    # Output
    # -------------------------
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        # default: alongside base cache
        out_dir = base_cache_path.parent / "probe_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive classes_tag for naming (best effort)
    if base_cath_kept is not None:
        classes_list = [str(int(x)) for x in np.asarray(base_cath_kept).ravel().tolist()]
        classes_tag = "_".join(classes_list) if classes_list else "all"
    else:
        classes_tag = "all"

    run_tag = derive_run_tag(curve_dir, pdb_args, len(probe_files))

    probe_cache_name = f"probe_cache_ae_{latent_rep}_class{classes_tag}_len_between_{base_min_len}_{base_max_len}_{run_tag}.npz"
    probe_cache_path = out_dir / probe_cache_name

    np.savez(
        str(probe_cache_path),
        # base
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
        base_cache_path=str(base_cache_path),
        umap_model_path=str(umap_model_path),
        # probe
        probe_latents=probe_latents,
        probe_tsne_2d=probe_tsne_2d,
        probe_umap_2d=probe_umap_2d,
        probe_lengths=probe_lengths,
        probe_helix_frac=probe_helix,
        probe_sheet_frac=probe_sheet,
        probe_loop_frac=probe_loop,
        probe_paths=probe_paths,
        probe_groups=probe_groups_arr,
        # meta
        config_path=str(Path(args.config).resolve()),
        ckpt_path=str(Path(args.ckpt).resolve()),
        latent_rep=str(latent_rep),
        knn_k=int(args.knn_k),
        seed=int(args.seed),
    )

    print("\n[Probe] Saved unified probe cache to:")
    print("  {}".format(str(probe_cache_path)))
    print("[Probe] Done.")


if __name__ == "__main__":
    main()
