# scripts/visualize_tsne_cath_seq.py
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE  # need scikit-learn
'''
python scripts/visualize_tsne_cath_seq.py \
  --ckpt /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/vq_s_gradient_ckpt_test11_15/epochepoch=549.ckpt \
  --data_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/prp-dataset/curves_npy_CATH \
  --raw_pdb_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/prp-dataset/raw_pdbs \
  --latent_n_tokens 48 \
  --code_dim 128 \
  --hidden_dim 512 \
  --num_layers 4 \
  --num_heads 8 \
  --max_seq_len 350 \
  --max_points 30000 \
  --perplexity 30 \
  --batch_size 256 \
  --num_workers 8
'''
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.vq_vae import VQVAE
from dataset import CurveDataset, pad_collate


def parse_args():
    p = argparse.ArgumentParser("Sequence-level t-SNE colored by CATH class")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument(
        "--list_txt",
        type=str,
        default="",
        help="optional list file; if empty, auto scan *_curve.npy",
    )
    p.add_argument(
        "--raw_pdb_dir",
        type=str,
        required=True,
        help="root dir of raw PDBs grouped by CATH ID",
    )
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=350)
    p.add_argument("--code_dim", type=int, default=128)
    p.add_argument("--latent_n_tokens", type=int, default=48)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--max_points",
        type=int,
        default=30000,
        help="max number of curves (points) for t-SNE",
    )
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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


def build_core_to_cath_class(raw_pdb_root):
    mapping = {}
    if not os.path.isdir(raw_pdb_root):
        print(f"[CATH] raw_pdb_dir not found: {raw_pdb_root}")
        return mapping

    kept_classes = (1, 2, 3, 4, 6)
    for cath_id in os.listdir(raw_pdb_root):
        cath_dir = os.path.join(raw_pdb_root, cath_id)
        if not os.path.isdir(cath_dir):
            continue
        parts = cath_id.split(".")
        if not parts:
            continue
        try:
            c_class = int(parts[0])
        except ValueError:
            continue
        if c_class not in kept_classes:
            continue
        for fn in os.listdir(cath_dir):
            if not fn.endswith(".pdb"):
                continue
            core = os.path.splitext(fn)[0]
            if core in mapping:
                continue
            mapping[core] = c_class

    print(f"[CATH] core->class mapping size: {len(mapping)}")
    return mapping


def build_auto_list_filtered(data_dir, max_len, suffix="_curve.npy"):
    auto_list = os.path.join(data_dir, f"_auto_list_len_le_{max_len}.txt")
    if os.path.isfile(auto_list):
        print(f"[List] using existing auto list: {auto_list}")
        return auto_list

    fnames = [
        fn
        for fn in os.listdir(data_dir)
        if fn.endswith(suffix)
    ]
    fnames = sorted(fnames)

    kept = 0
    skipped = 0
    with open(auto_list, "w") as f:
        for fn in fnames:
            path = os.path.join(data_dir, fn)
            length = None
            try:
                arr = np.load(path, allow_pickle=True)
            except Exception as e:
                print(f"[List] skip {fn}: load error: {e}")
                skipped += 1
                continue

            if isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object:
                try:
                    arr = arr.item()
                except Exception:
                    pass

            if isinstance(arr, dict):
                if "curve_coords" in arr:
                    v = arr["curve_coords"]
                    if isinstance(v, np.ndarray) and v.ndim >= 1:
                        length = int(v.shape[0])
                else:
                    for v in arr.values():
                        if isinstance(v, np.ndarray) and v.ndim >= 1:
                            length = int(v.shape[0])
                            break
            elif isinstance(arr, np.ndarray) and arr.ndim >= 1:
                length = int(arr.shape[0])

            if length is None:
                print(f"[List] skip {fn}: cannot infer length")
                skipped += 1
                continue

            if length <= max_len:
                f.write(fn + "\n")
                kept += 1
            else:
                skipped += 1

    print(
        f"[List] auto list created: {auto_list} | "
        f"kept={kept}, skipped_len>{max_len}={skipped}"
    )
    return auto_list


def build_labels_for_list(list_path, core_to_class):
    if not core_to_class:
        return None

    names = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(".json"):
                continue
            names.append(os.path.basename(line))

    labels = np.full(len(names), -1, dtype=np.int64)
    missing = 0
    for i, base in enumerate(names):
        core = base
        if core.endswith("_curve.npy"):
            core = core[:-10]
        elif core.endswith(".npy"):
            core = core[:-4]
        else:
            core = os.path.splitext(core)[0]
        c = core_to_class.get(core, -1)
        labels[i] = c
        if c == -1:
            missing += 1

    print(
        f"[CATH] labels built for {len(names)} samples "
        f"(missing={missing}, known={len(names) - missing})"
    )
    return labels


def stratified_curve_indices(labels, max_points):
    if labels is None:
        return None

    num_curves = labels.shape[0]
    if num_curves == 0:
        return None

    classes = [1, 2, 3, 4, 6]
    class_to_indices = {c: [] for c in classes}
    for idx, c in enumerate(labels):
        if c in class_to_indices:
            class_to_indices[c].append(idx)

    present_classes = [c for c in classes if len(class_to_indices[c]) > 0]
    if not present_classes:
        print("[CATH] no curves with valid CATH class")
        return None

    max_points = min(max_points, num_curves)
    per_class_quota = max_points // len(present_classes)
    if per_class_quota <= 0:
        per_class_quota = 1

    selected = []
    for c in present_classes:
        idxs = class_to_indices[c]
        if not idxs:
            continue
        k = min(len(idxs), per_class_quota)
        if len(idxs) <= k:
            chosen = idxs
        else:
            chosen = np.random.choice(idxs, size=k, replace=False)
        selected.extend(list(chosen))

    selected = sorted(set(selected))
    print(
        f"[Sample] total_curves={num_curves}, present_classes={present_classes}, "
        f"selected_curves={len(selected)}, max_points={max_points}"
    )
    return selected


@torch.no_grad()
def collect_seq_latents_with_labels(
    model,
    loader,
    device,
    use_amp,
    labels_for_loader,
    latent_n_tokens,
):
    rows = []
    labs = []
    idx_offset = 0

    try:
        autocast_ctx = torch.amp.autocast(
            device_type="cuda", enabled=(use_amp and device.type == "cuda")
        )
    except Exception:
        autocast_ctx = torch.cuda.amp.autocast(
            enabled=(use_amp and device.type == "cuda")
        )

    model.eval()
    with autocast_ctx:
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, mask = batch
            else:
                x, mask = batch, None

            B = x.size(0)
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True) if mask is not None else None

            h_fuse, _, _ = model.encode(x, mask=mask)      # [B,L,H]
            z_tok = model._tokenize_to_codes(h_fuse, mask) # [B,N,D]
            z_seq = z_tok.mean(dim=1)                      # [B,D]

            rows.append(z_seq.cpu())

            if labels_for_loader is not None:
                batch_labels = labels_for_loader[idx_offset : idx_offset + B]
                if batch_labels.shape[0] < B:
                    batch_labels = labels_for_loader[idx_offset :]
                labs.append(batch_labels.astype(np.int64, copy=False))
            idx_offset += B

    latents = torch.cat(rows, dim=0)
    labels = None
    if labels_for_loader is not None and labs:
        labels = np.concatenate(labs, axis=0)
    print(f"[Latents] collected curves: {latents.shape[0]}")
    return latents, labels


def plot_tsne_cath(lat2d, labels, out_png, title):
    if labels is None:
        print("[CATH] no labels, skip plot")
        return

    labels = labels.astype(np.int64, copy=False)
    mask = labels >= 0
    if not np.any(mask):
        print("[CATH] no valid labels, skip plot")
        return

    lat2d = lat2d[mask]
    labels = labels[mask]

    class_names = {
        1: "1 Mainly Alpha",
        2: "2 Mainly Beta",
        3: "3 Alpha Beta",
        4: "4 Few Secondary",
        6: "6 Special",
    }
    class_colors = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
        4: "tab:red",
        6: "tab:purple",
    }
    point_colors = [class_colors.get(int(c), "gray") for c in labels]

    plt.figure(figsize=(7.5, 7.0), dpi=140)
    plt.scatter(lat2d[:, 0], lat2d[:, 1], s=4, alpha=0.8, c=point_colors)
    plt.xlabel("t-SNE dim-1")
    plt.ylabel("t-SNE dim-2")
    plt.title(title)

    from matplotlib.lines import Line2D
    unique_classes = sorted(set(labels.tolist()))
    handles = []
    for c in unique_classes:
        cname = class_names.get(int(c), str(c))
        color = class_colors.get(int(c), "gray")
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=cname,
                markerfacecolor=color,
                markersize=6,
            )
        )
    plt.legend(handles=handles, title="CATH Class", loc="best")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    # list file with length filter
    if args.list_txt:
        if os.path.isabs(args.list_txt):
            list_path = args.list_txt
        else:
            list_path = os.path.join(args.data_dir, args.list_txt)
    else:
        list_path = build_auto_list_filtered(
            args.data_dir, max_len=int(args.max_seq_len)
        )

    # CATH labels and stratified sampling over curves
    core_to_class = build_core_to_cath_class(args.raw_pdb_dir)
    labels_all = build_labels_for_list(list_path, core_to_class)
    if labels_all is not None:
        selected_indices = stratified_curve_indices(
            labels_all, max_points=int(args.max_points)
        )
    else:
        selected_indices = None

    ds = CurveDataset(npy_dir=args.data_dir, list_path=list_path, train=False)
    if selected_indices is not None and len(selected_indices) > 0:
        ds_for_loader = Subset(ds, selected_indices)
        labels_for_loader = labels_all[selected_indices]
        print(
            f"[Data] using stratified subset: {len(selected_indices)} curves "
            f"out of {len(ds)}"
        )
    else:
        ds_for_loader = ds
        labels_for_loader = labels_all
        print(f"[Data] using full dataset: {len(ds)} curves")

    loader = DataLoader(
        ds_for_loader,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_collate,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    os.makedirs("viz_out_test_9PNG", exist_ok=True)

    model = VQVAE(
        input_dim=6,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        use_vq=False,
        codebook_size=1,
        code_dim=args.code_dim,
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
        latent_tokens=int(args.latent_n_tokens),
        tokenizer_heads=args.num_heads,
        tokenizer_layers=2,
        tokenizer_dropout=0.1,
        print_init=False,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = strip_prefixes(state)
    state = drop_quantizer_keys(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Load] missing={len(missing)} unexpected={len(unexpected)}")

    latents_seq, labels_seq = collect_seq_latents_with_labels(
        model=model,
        loader=loader,
        device=device,
        use_amp=bool(args.amp),
        labels_for_loader=labels_for_loader,
        latent_n_tokens=int(args.latent_n_tokens),
    )
    X = latents_seq.numpy().astype(np.float32, copy=False)
    print(f"[t-SNE] running TSNE on {X.shape[0]} points of dim={X.shape[1]}")

    tsne = TSNE(
        n_components=2,
        perplexity=float(args.perplexity),
        learning_rate="auto",
        init="pca",
        metric="euclidean",
        random_state=args.seed,
    )
    X2d = tsne.fit_transform(X)
    print("[t-SNE] done")

    out_png = "viz_out/tsne_seq_cath_class.png"
    plot_tsne_cath(
        X2d,
        labels_seq,
        out_png,
        title="TSNE colored by CATH Class (sequence-level)",
    )

    print("[Done] Saved:")
    print(f" - {out_png}")


if __name__ == "__main__":
    main()
