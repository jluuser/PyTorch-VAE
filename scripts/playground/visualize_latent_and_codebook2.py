# scripts/visualize_latent_and_codebook2.py
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import yaml  # optional config override

# UMAP + joblib for model saving
import umap
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

"""
Example:

python scripts/playground/visualize_latent_and_codebook2.py \
  --ckpt /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/checkpoints/vq_token64_K1024_D512_ResidualVQ_fromscratch/epochepoch=139.ckpt \
  --data_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/prp-dataset/curves_npy_CATH_by_cath \
  --latent_n_tokens 64 \
  --code_dim 512 \
  --codebook_size 1024 \
  --num_quantizers 4 \
  --latent_source zq \
  --hidden_dim 512 \
  --num_layers 4 \
  --num_heads 8 \
  --max_seq_len 350 \
  --min_len 1 \
  --max_len 80 \
  --max_points 350000 \
  --perplexity 30 \
  --n_neighbors 20 \
  --min_dist 0.1 \
  --batch_size 256 \
  --num_workers 16
"""

# -------------------------------------------------------
# Repo path setup
# -------------------------------------------------------
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
# file is under scripts/playground/, repo root is two levels up
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.vq_vae import VQVAE
from dataset import CurveDataset, pad_collate

# -------------------------------------------------------
# CATH classes to keep (top level)
# -------------------------------------------------------
KEPT_CLASSES = (1,)
CLASSES_TAG = "_".join(str(c) for c in KEPT_CLASSES)


# -------------------------------------------------------
# Argument parsing
# -------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        "Sequence-level t-SNE / UMAP with CATH / SS / length coloring (Residual VQ aware)"
    )

    # optional YAML config: override model hyperparams safely
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="optional YAML config; if set, overrides model hyperparams (hidden_dim, code_dim, latent_tokens, num_quantizers, codebook_size, ...)",
    )

    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument(
        "--list_txt",
        type=str,
        default="",
        help="optional list file; if empty, auto scan data_dir recursively and filter by length",
    )

    # model hyperparams (can be overridden from YAML)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument(
        "--max_seq_len",
        type=int,
        default=350,
        help="max sequence length for the model (padding / positional encoding)",
    )
    p.add_argument(
        "--code_dim",
        type=int,
        default=512,
        help="latent code dimension",
    )
    p.add_argument(
        "--codebook_size",
        type=int,
        default=1024,
        help="number of codes per residual-VQ level",
    )
    p.add_argument(
        "--num_quantizers",
        type=int,
        default=4,
        help="number of residual VQ levels (L); total codes = L * codebook_size",
    )
    p.add_argument(
        "--latent_n_tokens",
        type=int,
        default=64,
        help="number of latent tokens per sequence (N); e.g. 64",
    )
    p.add_argument(
        "--latent_source",
        type=str,
        default="zq",
        choices=["ze", "zq"],
        help="use pre-quantized token latents (ze) or quantized token latents (zq) for visualization",
    )

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--max_points",
        type=int,
        default=30000,
        help="max number of curves (points) for visualization; will be stratified by CATH class",
    )
    p.add_argument(
        "--min_len",
        type=int,
        default=1,
        help="minimum true length (number of residues) when building the list / cache",
    )
    p.add_argument(
        "--max_len",
        type=int,
        default=350,
        help="maximum true length (number of residues) when building the list / cache",
    )
    p.add_argument("--perplexity", type=float, default=30.0)

    # UMAP specific arguments
    p.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors parameter")
    p.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist parameter")

    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------
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


def infer_cath_class_from_relpath(rel_path):
    """
    Infer top-level CATH class (1,2,3,4,6) from a relative path like
    '1.10.8.10/xxxx_curve.npy'. Returns -1 if not available or not kept.
    """
    if not rel_path:
        return -1
    parts = rel_path.split(os.sep)
    if not parts:
        return -1
    cath_id = parts[0]
    cath_parts = cath_id.split(".")
    if not cath_parts:
        return -1
    try:
        c_class = int(cath_parts[0])
    except ValueError:
        return -1
    if c_class in KEPT_CLASSES:
        return c_class
    return -1


def build_labels_for_list_from_cath_dirs(data_dir, list_path):
    """
    Build label array for an existing list file whose lines are relative
    paths under data_dir, with CATH class encoded in the first directory.
    Labels for classes not in KEPT_CLASSES will be -1.
    """
    names = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(".json"):
                continue
            names.append(line)

    labels = np.full(len(names), -1, dtype=np.int64)
    missing = 0
    for i, rel_path in enumerate(names):
        c_class = infer_cath_class_from_relpath(rel_path)
        labels[i] = c_class
        if c_class == -1:
            missing += 1

    print(
        "[CATH] labels built for {} samples (missing={}, known={})".format(
            len(names), missing, len(names) - missing
        )
    )
    return labels


def load_rel_paths_from_list(list_path):
    """
    Load relative paths from a list file, skipping empty lines and *.json.
    This must be consistent with build_labels_for_list_from_cath_dirs.
    """
    names = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(".json"):
                continue
            names.append(line)
    return names


def build_auto_list_and_labels(data_dir, min_len, max_len, suffix="_curve.npy"):
    """
    Recursively scan data_dir, keep files whose true length is between
    [min_len, max_len] and whose parent directory encodes a CATH class
    in KEPT_CLASSES. Returns (list_path, labels_array).
    """
    min_len = int(min_len)
    max_len = int(max_len)
    classes_tag = CLASSES_TAG if CLASSES_TAG else "all"
    auto_list = os.path.join(
        data_dir,
        "_auto_list_class{}_len_between_{}_{}.txt".format(
            classes_tag, min_len, max_len
        ),
    )

    if os.path.isfile(auto_list):
        print("[List] using existing auto list: {}".format(auto_list))
        labels = build_labels_for_list_from_cath_dirs(data_dir, auto_list)
        return auto_list, labels

    kept = 0
    skipped_len = 0
    skipped_label = 0
    skipped_load = 0

    rel_paths = []
    labels_list = []

    with open(auto_list, "w") as f:
        for root, _, files in os.walk(data_dir):
            for fn in files:
                if not fn.endswith(suffix):
                    continue
                full_path = os.path.join(root, fn)
                rel_path = os.path.relpath(full_path, data_dir)

                c_class = infer_cath_class_from_relpath(rel_path)
                if c_class == -1:
                    skipped_label += 1
                    continue

                length = None
                try:
                    arr = np.load(full_path, allow_pickle=True)
                except Exception as e:
                    print("[List] skip {}: load error: {}".format(rel_path, e))
                    skipped_load += 1
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
                    print("[List] skip {}: cannot infer length".format(rel_path))
                    skipped_len += 1
                    continue

                if length < min_len or length > max_len:
                    skipped_len += 1
                    continue

                f.write(rel_path + "\n")
                rel_paths.append(rel_path)
                labels_list.append(c_class)
                kept += 1

    labels = np.array(labels_list, dtype=np.int64)
    print(
        "[List] auto list created: {} | kept={} skipped_len_or_unknown={} skipped_label={} skipped_load={}".format(
            auto_list, kept, skipped_len, skipped_label, skipped_load
        )
    )
    return auto_list, labels


def stratified_curve_indices(labels, max_points):
    """
    Stratified sampling of curve indices by CATH class.
    Only classes in KEPT_CLASSES are considered.
    """
    if labels is None:
        return None

    num_curves = labels.shape[0]
    if num_curves == 0:
        return None

    classes = list(KEPT_CLASSES)
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
        "[Sample] total_curves={}, present_classes={}, selected_curves={}, max_points={}".format(
            num_curves, present_classes, len(selected), max_points
        )
    )
    return selected


# -------------------------------------------------------
# Latent collection (Residual VQ aware)
# -------------------------------------------------------
@torch.no_grad()
def collect_seq_latents_with_labels_and_stats(
    model,
    loader,
    device,
    use_amp,
    labels_for_loader,
    latent_source: str = "zq",
):
    """
    Collect sequence-level latent vectors and basic statistics.
    If latent_source == "ze": use encoder+tokenizer output.
    If latent_source == "zq": pass through residual VQ and use quantized latents,
                              also collect local code indices per level.
    """
    rows = []
    labs = []
    hel_list = []
    sheet_list = []
    loop_list = []
    len_list = []
    idx_offset = 0
    all_indices_local = []

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
            if mask is not None:
                mask = mask.to(device, non_blocking=True)
            else:
                mask = torch.ones(
                    (B, x.size(1)), dtype=torch.bool, device=device
                )

            # encoder -> token-level continuous latents
            h_fuse, _, _ = model.encode(x, mask=mask)
            z_e_tok = model._tokenize_to_codes(h_fuse, mask)  # [B, N, D]

            indices_local_this = None

            if str(latent_source).lower() == "zq":
                if getattr(model, "quantizer", None) is None:
                    raise RuntimeError("latent_source=zq requires a VQ-enabled model (model.quantizer is None)")

                B_, N_, D_ = z_e_tok.shape
                token_mask = torch.ones(
                    (B_, N_), dtype=torch.bool, device=device
                )

                # residual VQ forward
                z_q_st, z_q, all_indices_cat, _ = model.quantizer(
                    z_e_tok,
                    do_ema_update=False,
                    allow_reinit=False,
                    mask=token_mask,
                )
                z_tok = z_q  # [B, N, D]

                # all_indices_cat: [L*N], L=num_quantizers, N=B*N_tokens
                L = int(getattr(model.quantizer, "num_quantizers", getattr(model, "num_quantizers", 1)))
                if L > 1:
                    K_per = int(getattr(model.quantizer, "K_per", 0))
                    if K_per <= 0:
                        raise RuntimeError("quantizer.K_per must be > 0 for residual VQ")

                    # reshape: [L, B*N]
                    all_indices_lv = all_indices_cat.view(L, B_, N_)
                    # [B, N, L] (global indices)
                    indices_global = all_indices_lv.view(L, B_, N_).permute(1, 2, 0)  # [B,N,L]

                    # convert to local indices 0..K_per-1 using level offsets
                    level_ids = torch.arange(L, device=indices_global.device).view(1, 1, L)
                    offsets = level_ids * K_per
                    indices_local = indices_global - offsets  # [B,N,L]

                    indices_local_this = indices_local.cpu().numpy()
                else:
                    # single-level: all_indices_cat is [B*N]
                    indices_global = all_indices_cat.view(B_, N_)
                    indices_local_this = indices_global.unsqueeze(-1).cpu().numpy()  # [B,N,1]
            else:
                # use ze directly
                z_tok = z_e_tok

            # sequence-level latent = mean of token latents
            z_seq = z_tok.mean(dim=1)  # [B, D]
            rows.append(z_seq.cpu())

            if indices_local_this is not None:
                all_indices_local.append(indices_local_this)

            # length stats
            length = mask.sum(dim=1).to(torch.float32)
            len_list.append(length.cpu().numpy())

            # secondary structure fractions
            ss = x[..., 3:]
            valid = mask.unsqueeze(-1).to(ss.dtype)
            ss_valid = ss * valid
            counts = ss_valid.sum(dim=1)
            denom = torch.clamp(length.unsqueeze(-1), min=1.0)
            frac = (counts / denom).cpu().numpy()
            hel_list.append(frac[:, 0])
            sheet_list.append(frac[:, 1])
            loop_list.append(frac[:, 2])

            if labels_for_loader is not None:
                batch_labels = labels_for_loader[idx_offset: idx_offset + B]
                if batch_labels.shape[0] < B:
                    batch_labels = labels_for_loader[idx_offset:]
                labs.append(batch_labels.astype(np.int64, copy=False))
            idx_offset += B

    latents = torch.cat(rows, dim=0)
    labels = None
    if labels_for_loader is not None and labs:
        labels = np.concatenate(labs, axis=0)

    helix_frac = np.concatenate(hel_list, axis=0)
    sheet_frac = np.concatenate(sheet_list, axis=0)
    loop_frac = np.concatenate(loop_list, axis=0)
    lengths = np.concatenate(len_list, axis=0)

    indices_local = None
    if all_indices_local:
        indices_local = np.concatenate(all_indices_local, axis=0)  # [num_samples, N_tokens, L]

    print("[Latents] collected curves: {}".format(latents.shape[0]))
    print(
        "[Stats] helix_frac range=({:.3f},{:.3f}), sheet_frac range=({:.3f},{:.3f}), "
        "loop_frac range=({:.3f},{:.3f}), length range=({:.0f},{:.0f})".format(
            helix_frac.min(),
            helix_frac.max(),
            sheet_frac.min(),
            sheet_frac.max(),
            loop_frac.min(),
            loop_frac.max(),
            lengths.min(),
            lengths.max(),
        )
    )
    if indices_local is not None:
        print("[Indices] collected local indices with shape {}".format(indices_local.shape))
    return latents, labels, helix_frac, sheet_frac, loop_frac, lengths, indices_local


# -------------------------------------------------------
# Color utilities for SS simplex palette
# -------------------------------------------------------
HELIX_COLOR = np.array([239, 68, 68], dtype=np.float32) / 255.0
SHEET_COLOR = np.array([34, 197, 94], dtype=np.float32) / 255.0
LOOP_COLOR  = np.array([59, 130, 246], dtype=np.float32) / 255.0
GRAY_BG     = np.array([229, 231, 235], dtype=np.float32) / 255.0


def mix_three_colors_simplex(
    helix_base,
    sheet_base,
    loop_base,
    helix_frac,
    sheet_frac,
    loop_frac,
    weight_exp=1.0,
):
    h = np.asarray(helix_frac, dtype=np.float32)
    s = np.asarray(sheet_frac, dtype=np.float32)
    l = np.asarray(loop_frac, dtype=np.float32)

    w = np.stack([h, s, l], axis=1)
    w = np.clip(w, 0.0, 1.0)
    w_sum = np.sum(w, axis=1, keepdims=True)
    w = np.divide(w, w_sum, out=np.zeros_like(w), where=w_sum > 0.0)

    max_w = np.max(w, axis=1)
    purity = (max_w - 1.0 / 3.0) / (1.0 - 1.0 / 3.0)
    purity = np.clip(purity, 0.0, 1.0)

    if weight_exp != 1.0:
        purity = np.power(purity, weight_exp)

    winner_idx = np.argmax(w, axis=1)
    base_colors = np.stack([helix_base, sheet_base, loop_base], axis=0)
    winner_colors = base_colors[winner_idx]

    colors = GRAY_BG[None, :] * (1.0 - purity)[:, None] + winner_colors * purity[:, None]
    colors = np.clip(colors, 0.0, 1.0)
    return colors


def generate_simplex_palette(
    out_png,
    helix_color,
    sheet_color,
    loop_color,
    size=400,
    padding=40,
    weight_exp=1.0,
):
    bg_color = np.array([248, 250, 252], dtype=np.float32) / 255.0
    img = np.tile(bg_color[None, None, :], (size, size, 1))

    v1 = np.array([size / 2.0, padding], dtype=np.float32)
    v2 = np.array([size - padding, size - padding], dtype=np.float32)
    v3 = np.array([padding, size - padding], dtype=np.float32)

    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    px = grid_x
    py = grid_y

    detT = (v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * (v1[1] - v3[1])
    w1 = ((v2[1] - v3[1]) * (px - v3[0]) + (v3[0] - v2[0]) * (py - v3[1])) / detT
    w2 = ((v3[1] - v1[1]) * (px - v3[0]) + (v1[0] - v3[0]) * (py - v3[1])) / detT
    w3 = 1.0 - w1 - w2

    mask = (w1 >= -0.005) & (w2 >= -0.005) & (w3 >= -0.005)

    cw1 = np.clip(w1, 0.0, 1.0)
    cw2 = np.clip(w2, 0.0, 1.0)
    cw3 = np.clip(w3, 0.0, 1.0)
    sum_w = cw1 + cw2 + cw3
    sum_w[sum_w == 0.0] = 1.0
    nw1 = cw1 / sum_w
    nw2 = cw2 / sum_w
    nw3 = cw3 / sum_w

    h_flat = nw1[mask].ravel()
    s_flat = nw2[mask].ravel()
    l_flat = nw3[mask].ravel()
    colors = mix_three_colors_simplex(
        helix_color,
        sheet_color,
        loop_color,
        h_flat,
        s_flat,
        l_flat,
        weight_exp=weight_exp,
    )
    img[mask] = colors

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=140)
    ax.imshow(img, origin="upper")
    ax.set_xlim(0, size - 1)
    ax.set_ylim(size - 1, 0)
    ax.axis("off")
    ax.set_title("Color Palette (Simplex)", fontsize=12)

    tri_x = [v1[0], v2[0], v3[0], v1[0]]
    tri_y = [v1[1], v2[1], v3[1], v1[1]]
    ax.plot(tri_x, tri_y, color="#334155", linewidth=1.5)

    ax.text(
        v1[0],
        v1[1] - 10,
        "Helix",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#0f172a",
    )
    ax.text(
        v2[0] + 10,
        v2[1] + 5,
        "Sheet",
        ha="left",
        va="top",
        fontsize=8,
        color="#0f172a",
    )
    ax.text(
        v3[0] - 10,
        v3[1] + 5,
        "Loop",
        ha="right",
        va="top",
        fontsize=8,
        color="#0f172a",
    )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Optional: override model hyperparams from YAML config
    if getattr(args, "config", ""):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        mp = (cfg or {}).get("model_params", {})
        for k_src, k_dst in [
            ("hidden_dim", "hidden_dim"),
            ("num_layers", "num_layers"),
            ("num_heads", "num_heads"),
            ("max_seq_len", "max_seq_len"),
            ("code_dim", "code_dim"),
            ("latent_tokens", "latent_n_tokens"),
            ("num_quantizers", "num_quantizers"),
            ("codebook_size", "codebook_size"),
        ]:
            if k_src in mp:
                setattr(args, k_dst, int(mp[k_src]))
        print(
            "[Config] override model params from {}: hidden_dim={}, num_layers={}, num_heads={}, "
            "max_seq_len={}, code_dim={}, latent_n_tokens={}, num_quantizers={}, codebook_size={}".format(
                args.config,
                args.hidden_dim,
                args.num_layers,
                args.num_heads,
                args.max_seq_len,
                args.code_dim,
                args.latent_n_tokens,
                args.num_quantizers,
                args.codebook_size,
            )
        )

    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    # list file with length filter and CATH labels inferred from directory
    if args.list_txt:
        if os.path.isabs(args.list_txt):
            list_path = args.list_txt
        else:
            list_path = os.path.join(args.data_dir, args.list_txt)
        labels_all = build_labels_for_list_from_cath_dirs(args.data_dir, list_path)
    else:
        list_path, labels_all = build_auto_list_and_labels(
            args.data_dir,
            min_len=int(args.min_len),
            max_len=int(args.max_len),
        )

    # all rel paths (aligned with labels_all)
    rel_paths_all = load_rel_paths_from_list(list_path)
    rel_paths_all = np.array(rel_paths_all, dtype=object)

    # stratified sampling over curves by CATH class
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
        rel_paths_for_loader = rel_paths_all[selected_indices]
        print(
            "[Data] using stratified subset: {} curves out of {}".format(
                len(selected_indices), len(ds)
            )
        )
    else:
        ds_for_loader = ds
        labels_for_loader = labels_all
        rel_paths_for_loader = rel_paths_all
        print("[Data] using full dataset: {} curves".format(len(ds)))

    # full CATH id (e.g. "1.10.420.10") from relative paths
    cath_full_for_loader = np.array(
        [
            rp.split(os.sep)[0] if isinstance(rp, str) and rp else ""
            for rp in rel_paths_for_loader
        ],
        dtype=object,
    )

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

    classes_tag = CLASSES_TAG if CLASSES_TAG else "all"
    out_dir = os.path.join(
        "latent_analysis",
        "class{}".format(classes_tag),
    )
    os.makedirs(out_dir, exist_ok=True)

    # optional palette visualization
    simplex_palette_png = os.path.join(out_dir, "simplex_palette.png")
    generate_simplex_palette(
        simplex_palette_png,
        HELIX_COLOR,
        SHEET_COLOR,
        LOOP_COLOR,
        size=400,
        padding=40,
        weight_exp=1.0,
    )
    print("[Palette] saved simplex palette to {}".format(simplex_palette_png))

    # Build a VQ-enabled model so we can visualize either ze or zq.
    model = VQVAE(
        input_dim=6,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        use_vq=True,
        residual_vq=(int(args.num_quantizers) > 1),
        num_quantizers=int(args.num_quantizers),
        codebook_size=int(args.codebook_size),
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
    # important: do NOT drop quantizer keys now; we want the trained codebook
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[Load] missing={} unexpected={}".format(len(missing), len(unexpected)))

    # collect latents
    latents_seq, labels_seq, helix_frac, sheet_frac, loop_frac, lengths, indices_local = \
        collect_seq_latents_with_labels_and_stats(
            model=model,
            loader=loader,
            device=device,
            use_amp=bool(args.amp),
            labels_for_loader=labels_for_loader,
            latent_source=str(args.latent_source),
        )

    X = latents_seq.numpy().astype(np.float32, copy=False)
    print("[t-SNE] running TSNE on {} points of dim={}".format(X.shape[0], X.shape[1]))

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

    # -----------------------------------------------------------------
    # UMAP: fit + save reducer + coordinates
    # -----------------------------------------------------------------
    print(f"[UMAP] running UMAP on {X.shape[0]} points of dim={X.shape[1]}...")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=2,
        metric="euclidean",
        random_state=args.seed,
    )
    X_umap_2d = reducer.fit_transform(X)
    print("[UMAP] done")

    umap_model_name = "umap_reducer_class{}_len_between_{}_{}.pkl".format(
        classes_tag,
        int(args.min_len),
        int(args.max_len),
    )
    umap_model_path = os.path.join(out_dir, umap_model_name)
    joblib.dump(reducer, umap_model_path)
    print("[UMAP] saved reducer model to {}".format(umap_model_path))

    # -----------------------------------------------------------------
    # Save full cache (latents + tsne + umap + stats + indices)
    # -----------------------------------------------------------------
    cache_name = "tsne_cache_resvq_{}_class{}_len_between_{}_{}.npz".format(
        str(args.latent_source),
        classes_tag,
        int(args.min_len),
        int(args.max_len),
    )
    cache_path = os.path.join(out_dir, cache_name)

    np.savez(
        cache_path,
        latents=latents_seq.numpy().astype(np.float32, copy=False),
        tsne_2d=X2d.astype(np.float32, copy=False),
        umap_2d=X_umap_2d.astype(np.float32, copy=False),
        indices_local=indices_local,
        labels=labels_seq,
        helix_frac=helix_frac.astype(np.float32, copy=False),
        sheet_frac=sheet_frac.astype(np.float32, copy=False),
        loop_frac=loop_frac.astype(np.float32, copy=False),
        lengths=lengths.astype(np.float32, copy=False),
        rel_paths=rel_paths_for_loader,
        cath_full=cath_full_for_loader,
        ckpt=args.ckpt,
        seed=args.seed,
        perplexity=float(args.perplexity),
        latent_n_tokens=int(args.latent_n_tokens),
        code_dim=int(args.code_dim),
        codebook_size=int(args.codebook_size),
        num_quantizers=int(args.num_quantizers),
        latent_source=str(args.latent_source),
        max_seq_len=int(args.max_seq_len),
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        cath_kept_classes=np.array(KEPT_CLASSES, dtype=np.int64),
    )
    print("[Cache] saved t-SNE + UMAP cache to {}".format(cache_path))


if __name__ == "__main__":
    main()
