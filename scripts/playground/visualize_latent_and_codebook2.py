#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
python scripts/playground/visualize_latent_and_codebook2.py \
  --config configs/stage1_ae.yaml \
  --ckpt /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/checkpoints/aeot_sigmoid/epochepoch=epoch=089.ckpt \
  --data_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/prp-dataset/curves_npy_CATH_by_cath \
  --min_len 1 --max_len 80 \
  --max_points 300000 \
  --latent_rep tokens_mean \
  --perplexity 30 \
  --n_neighbors 20 --min_dist 0.1 \
  --batch_size 256 --num_workers 16 \
  --amp

python scripts/playground/visualize_latent_and_codebook2.py \
  --config configs/stage1_ae.yaml \
  --ckpt /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/checkpoints/aeot/epochepoch=epoch=089.ckpt \
  --data_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/prp-dataset/curves_npy_CATH_by_cath \
  --min_len 1 --max_len 80 \
  --max_points 30000 \
  --latent_rep tokens_flatten \
  --perplexity 30 \
  --n_neighbors 20 --min_dist 0.1 \
  --batch_size 256 --num_workers 16 \
  --amp

'''
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import yaml
import umap
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.manifold import TSNE


# -------------------------------------------------------
# Repo path setup
# -------------------------------------------------------
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiment import build_experiment_from_yaml
from dataset import CurveDataset, pad_collate


# -------------------------------------------------------
# CATH classes to keep (top level)
# -------------------------------------------------------
KEPT_CLASSES = (1,)
CLASSES_TAG = "_".join(str(c) for c in KEPT_CLASSES) if KEPT_CLASSES else "all"


# -------------------------------------------------------
# SS simplex palette colors
# -------------------------------------------------------
HELIX_COLOR = np.array([239, 68, 68], dtype=np.float32) / 255.0
SHEET_COLOR = np.array([34, 197, 94], dtype=np.float32) / 255.0
LOOP_COLOR  = np.array([59, 130, 246], dtype=np.float32) / 255.0
GRAY_BG     = np.array([229, 231, 235], dtype=np.float32) / 255.0


def parse_args():
    p = argparse.ArgumentParser(
        "AE latent t-SNE / UMAP base + full PNG plots (compatible with old script structure)"
    )

    p.add_argument("--config", type=str, required=True, help="Path to stage1_ae.yaml")
    p.add_argument("--ckpt", type=str, required=True, help="Path to AE checkpoint")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--list_txt", type=str, default="")

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--max_points", type=int, default=30000)
    p.add_argument("--min_len", type=int, default=1)
    p.add_argument("--max_len", type=int, default=350)

    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.1)

    p.add_argument("--latent_rep", type=str, default="tokens_mean",
                   choices=["tokens_mean", "tokens_flatten"])

    p.add_argument("--seed", type=int, default=42)

    # output root: keep your new folder name style
    p.add_argument("--out_root", type=str, default="latent_analysis_ae_sigmoid")

    # plot styles
    p.add_argument("--plot_size", type=float, default=3.0)
    p.add_argument("--plot_alpha", type=float, default=0.85)

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


def infer_cath_class_from_relpath(rel_path: str) -> int:
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
    if KEPT_CLASSES and (c_class not in KEPT_CLASSES):
        return -1
    return c_class


def build_labels_for_list_from_cath_dirs(data_dir: str, list_path: str) -> np.ndarray:
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

    print(f"[CATH] labels built for {len(names)} samples (missing={missing}, known={len(names)-missing})")
    return labels


def load_rel_paths_from_list(list_path: str) -> List[str]:
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


def build_auto_list_and_labels(data_dir: str, min_len: int, max_len: int, suffix="_curve.npy") -> Tuple[str, np.ndarray]:
    min_len = int(min_len)
    max_len = int(max_len)
    classes_tag = CLASSES_TAG if CLASSES_TAG else "all"

    auto_list = os.path.join(
        data_dir,
        f"_auto_list_class{classes_tag}_len_between_{min_len}_{max_len}.txt",
    )

    if os.path.isfile(auto_list):
        print(f"[List] using existing auto list: {auto_list}")
        labels = build_labels_for_list_from_cath_dirs(data_dir, auto_list)
        return auto_list, labels

    kept = 0
    skipped_len = 0
    skipped_label = 0
    skipped_load = 0

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
                    print(f"[List] skip {rel_path}: load error: {e}")
                    skipped_load += 1
                    continue

                if isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object:
                    try:
                        arr = arr.item()
                    except Exception:
                        pass

                if isinstance(arr, dict):
                    if "curve_coords" in arr and isinstance(arr["curve_coords"], np.ndarray):
                        length = int(arr["curve_coords"].shape[0])
                    elif "ca_coords" in arr and isinstance(arr["ca_coords"], np.ndarray):
                        length = int(arr["ca_coords"].shape[0])
                    else:
                        for v in arr.values():
                            if isinstance(v, np.ndarray) and v.ndim >= 1:
                                length = int(v.shape[0])
                                break
                elif isinstance(arr, np.ndarray) and arr.ndim >= 1:
                    length = int(arr.shape[0])

                if length is None:
                    skipped_len += 1
                    continue
                if length < min_len or length > max_len:
                    skipped_len += 1
                    continue

                f.write(rel_path + "\n")
                labels_list.append(c_class)
                kept += 1

    labels = np.array(labels_list, dtype=np.int64)
    print(f"[List] auto list created: {auto_list} | kept={kept} skipped_len={skipped_len} skipped_label={skipped_label} skipped_load={skipped_load}")
    return auto_list, labels


def stratified_curve_indices(labels: np.ndarray, max_points: int) -> Optional[List[int]]:
    if labels is None or labels.shape[0] == 0:
        return None

    classes = list(KEPT_CLASSES) if KEPT_CLASSES else sorted(list(set(labels.tolist())))
    class_to_indices = {c: [] for c in classes}
    for idx, c in enumerate(labels):
        if c in class_to_indices:
            class_to_indices[c].append(idx)

    present_classes = [c for c in classes if len(class_to_indices[c]) > 0]
    if not present_classes:
        print("[CATH] no curves with valid CATH class")
        return None

    num_curves = int(labels.shape[0])
    max_points = min(int(max_points), num_curves)
    per_class_quota = max_points // len(present_classes)
    if per_class_quota <= 0:
        per_class_quota = 1

    selected = []
    for c in present_classes:
        idxs = class_to_indices[c]
        k = min(len(idxs), per_class_quota)
        if len(idxs) <= k:
            chosen = idxs
        else:
            chosen = np.random.choice(idxs, size=k, replace=False)
        selected.extend(list(chosen))

    selected = sorted(set(selected))
    print(f"[Sample] total_curves={num_curves}, present_classes={present_classes}, selected_curves={len(selected)}, max_points={max_points}")
    return selected


def mix_three_colors_simplex(helix_base, sheet_base, loop_base, helix_frac, sheet_frac, loop_frac, weight_exp=1.0):
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
    return np.clip(colors, 0.0, 1.0)


def generate_simplex_palette(out_png, helix_color, sheet_color, loop_color, size=400, padding=40):
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

    colors = mix_three_colors_simplex(helix_color, sheet_color, loop_color, h_flat, s_flat, l_flat)
    img[mask] = colors

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=140)
    ax.imshow(img, origin="upper")
    ax.axis("off")
    ax.set_title("Color Palette (Simplex)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _cath_color_map():
    return {
        1: np.array([59, 130, 246], dtype=np.float32) / 255.0,
        2: np.array([34, 197, 94], dtype=np.float32) / 255.0,
        3: np.array([239, 68, 68], dtype=np.float32) / 255.0,
        4: np.array([168, 85, 247], dtype=np.float32) / 255.0,
        6: np.array([245, 158, 11], dtype=np.float32) / 255.0,
        -1: GRAY_BG,
    }


def _colors_from_labels(labels: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], List[Line2D]]:
    if labels is None:
        return None, []

    cmap = _cath_color_map()
    colors = np.stack([cmap.get(int(c), GRAY_BG) for c in labels], axis=0).astype(np.float32, copy=False)

    legend = []
    uniq = sorted(list(set(int(x) for x in labels.tolist())))
    for c in uniq:
        col = cmap.get(c, GRAY_BG)
        name = f"CATH {c}" if c != -1 else "Unknown"
        legend.append(Line2D([0], [0], marker="o", color="w", label=name,
                             markerfacecolor=col, markersize=7))
    return colors, legend


def _save_scatter(out_png: str, X2d: np.ndarray, title: str,
                  colors: Optional[np.ndarray] = None,
                  cvals: Optional[np.ndarray] = None,
                  cmap: str = "viridis",
                  point_size: float = 3.0,
                  alpha: float = 0.85,
                  legend: Optional[List[Line2D]] = None,
                  colorbar: bool = False):
    fig = plt.figure(figsize=(7.5, 6.5), dpi=150)
    ax = fig.add_subplot(111)

    if colors is not None:
        ax.scatter(X2d[:, 0], X2d[:, 1], s=point_size, c=colors, alpha=alpha, linewidths=0)
    elif cvals is not None:
        sc = ax.scatter(X2d[:, 0], X2d[:, 1], s=point_size, c=cvals, cmap=cmap, alpha=alpha, linewidths=0)
        if colorbar:
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=8)
    else:
        ax.scatter(X2d[:, 0], X2d[:, 1], s=point_size, alpha=alpha, linewidths=0)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Dim 1", fontsize=9)
    ax.set_ylabel("Dim 2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(False)

    if legend:
        ax.legend(handles=legend, loc="best", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _save_hexbin(out_png: str, X2d: np.ndarray, title: str):
    fig = plt.figure(figsize=(7.5, 6.5), dpi=150)
    ax = fig.add_subplot(111)
    hb = ax.hexbin(X2d[:, 0], X2d[:, 1], gridsize=70, mincnt=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Dim 1", fontsize=9)
    ax.set_ylabel("Dim 2", fontsize=9)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _save_hist(out_png: str, x: np.ndarray, title: str, bins: int = 50):
    fig = plt.figure(figsize=(7.2, 4.6), dpi=150)
    ax = fig.add_subplot(111)
    ax.hist(x, bins=bins)
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


@torch.no_grad()
def collect_ae_latents_with_stats(model, loader, device, use_amp: bool, labels_for_loader: Optional[np.ndarray], latent_rep: str):
    rows = []
    labs = []
    hel_list = []
    sheet_list = []
    loop_list = []
    len_list = []
    idx_offset = 0

    try:
        autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda"))
    except Exception:
        autocast_ctx = torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda"))

    model.eval()
    with autocast_ctx:
        for batch in loader:
            x, mask = batch
            B = x.size(0)
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            h_fuse, _, _ = model.encode(x, mask=mask)
            z_tok = model._tokenize_to_codes(h_fuse, mask)  # [B, N, D]

            rep = str(latent_rep).lower()
            if rep == "tokens_mean":
                z = z_tok.mean(dim=1)  # [B, D]
            elif rep == "tokens_flatten":
                z = z_tok.reshape(z_tok.size(0), -1)  # [B, N*D]
            else:
                raise ValueError(f"Unknown latent_rep={latent_rep}")

            rows.append(z.detach().float().cpu())

            length = mask.sum(dim=1).to(torch.float32)
            len_list.append(length.cpu().numpy())

            ss = x[..., 3:]
            valid = mask.unsqueeze(-1).to(ss.dtype)
            ss_valid = ss * valid
            counts = ss_valid.sum(dim=1)
            denom = torch.clamp(length.unsqueeze(-1), min=1.0)
            frac = (counts / denom).detach().cpu().numpy().astype(np.float32, copy=False)
            hel_list.append(frac[:, 0])
            sheet_list.append(frac[:, 1])
            loop_list.append(frac[:, 2])

            if labels_for_loader is not None:
                batch_labels = labels_for_loader[idx_offset: idx_offset + B]
                if batch_labels.shape[0] < B:
                    batch_labels = labels_for_loader[idx_offset:]
                labs.append(batch_labels.astype(np.int64, copy=False))
            idx_offset += B

    latents = torch.cat(rows, dim=0).numpy().astype(np.float32, copy=False)
    labels = np.concatenate(labs, axis=0) if (labels_for_loader is not None and labs) else None
    helix_frac = np.concatenate(hel_list, axis=0).astype(np.float32, copy=False)
    sheet_frac = np.concatenate(sheet_list, axis=0).astype(np.float32, copy=False)
    loop_frac  = np.concatenate(loop_list, axis=0).astype(np.float32, copy=False)
    lengths    = np.concatenate(len_list, axis=0).astype(np.float32, copy=False)

    print(f"[Latents] collected curves: {latents.shape[0]} | dim={latents.shape[1]} | rep={latent_rep}")
    print(f"[Stats] length range=({lengths.min():.0f},{lengths.max():.0f})")
    return latents, labels, helix_frac, sheet_frac, loop_frac, lengths


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print("[Device] Using:", device)

    # list + labels
    if args.list_txt:
        list_path = args.list_txt if os.path.isabs(args.list_txt) else os.path.join(args.data_dir, args.list_txt)
        labels_all = build_labels_for_list_from_cath_dirs(args.data_dir, list_path)
    else:
        list_path, labels_all = build_auto_list_and_labels(args.data_dir, int(args.min_len), int(args.max_len))

    rel_paths_all = np.array(load_rel_paths_from_list(list_path), dtype=object)

    selected_indices = stratified_curve_indices(labels_all, int(args.max_points)) if labels_all is not None else None

    ds = CurveDataset(npy_dir=args.data_dir, list_path=list_path, train=False)
    if selected_indices is not None and len(selected_indices) > 0:
        ds_for_loader = Subset(ds, selected_indices)
        labels_for_loader = labels_all[selected_indices]
        rel_paths_for_loader = rel_paths_all[selected_indices]
        print(f"[Data] using stratified subset: {len(selected_indices)} curves out of {len(ds)}")
    else:
        ds_for_loader = ds
        labels_for_loader = labels_all
        rel_paths_for_loader = rel_paths_all
        print(f"[Data] using full dataset: {len(ds)} curves")

    cath_full_for_loader = np.array(
        [rp.split(os.sep)[0] if isinstance(rp, str) and rp else "" for rp in rel_paths_for_loader],
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

    out_dir = os.path.join(args.out_root, f"class{CLASSES_TAG}")
    os.makedirs(out_dir, exist_ok=True)

    # palette
    simplex_palette_png = os.path.join(out_dir, "simplex_palette.png")
    generate_simplex_palette(simplex_palette_png, HELIX_COLOR, SHEET_COLOR, LOOP_COLOR)
    print("[Palette] saved:", simplex_palette_png)

    # load AE
    exp, _ = build_experiment_from_yaml(args.config)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    new_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            new_state[k[6:]] = v
        else:
            new_state[k] = v

    missing, unexpected = exp.model.load_state_dict(new_state, strict=False)
    print(f"[AE] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")

    model = exp.model.to(device).eval()

    # collect latents
    latents, labels, helix_frac, sheet_frac, loop_frac, lengths = collect_ae_latents_with_stats(
        model=model,
        loader=loader,
        device=device,
        use_amp=bool(args.amp),
        labels_for_loader=labels_for_loader,
        latent_rep=str(args.latent_rep),
    )

    X = latents.astype(np.float32, copy=False)

    # TSNE
    print(f"[t-SNE] running TSNE on {X.shape[0]} points of dim={X.shape[1]}")
    tsne = TSNE(
        n_components=2,
        perplexity=float(args.perplexity),
        learning_rate="auto",
        init="pca",
        metric="euclidean",
        random_state=args.seed,
    )
    tsne_2d = tsne.fit_transform(X).astype(np.float32, copy=False)
    print("[t-SNE] done")

    # UMAP
    print(f"[UMAP] running UMAP on {X.shape[0]} points of dim={X.shape[1]}")
    reducer = umap.UMAP(
        n_neighbors=int(args.n_neighbors),
        min_dist=float(args.min_dist),
        n_components=2,
        metric="euclidean",
        random_state=args.seed,
    )
    umap_2d = reducer.fit_transform(X).astype(np.float32, copy=False)
    print("[UMAP] done")

    umap_model_name = f"umap_reducer_ae_{args.latent_rep}_class{CLASSES_TAG}_len_between_{int(args.min_len)}_{int(args.max_len)}.pkl"
    umap_model_path = os.path.join(out_dir, umap_model_name)
    joblib.dump(reducer, umap_model_path)
    print("[UMAP] saved reducer model to", umap_model_path)

    cache_name = f"tsne_cache_ae_{args.latent_rep}_class{CLASSES_TAG}_len_between_{int(args.min_len)}_{int(args.max_len)}.npz"
    cache_path = os.path.join(out_dir, cache_name)

    latent_tokens = int(getattr(model, "latent_n_tokens", getattr(model, "latent_tokens", 0)) or 0)
    code_dim = int(getattr(model, "code_dim", 0) or 0)

    np.savez(
        cache_path,
        latents=X,
        tsne_2d=tsne_2d,
        umap_2d=umap_2d,
        labels=labels,
        helix_frac=helix_frac,
        sheet_frac=sheet_frac,
        loop_frac=loop_frac,
        lengths=lengths,
        rel_paths=rel_paths_for_loader,
        cath_full=cath_full_for_loader,
        ckpt=args.ckpt,
        config=args.config,
        seed=int(args.seed),
        perplexity=float(args.perplexity),
        latent_rep=str(args.latent_rep),
        latent_n_tokens=latent_tokens,
        code_dim=code_dim,
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        cath_kept_classes=np.array(KEPT_CLASSES, dtype=np.int64),
        n_neighbors=int(args.n_neighbors),
        min_dist=float(args.min_dist),
    )
    print("[Cache] saved t-SNE + UMAP cache to", cache_path)

    # -------------------------------------------------------
    # Full PNG plots (THIS is what your pasted script lacked)
    # -------------------------------------------------------
    pt_size = float(args.plot_size)
    pt_alpha = float(args.plot_alpha)

    cath_colors, cath_legend = _colors_from_labels(labels)
    ss_simplex_colors = mix_three_colors_simplex(
        HELIX_COLOR, SHEET_COLOR, LOOP_COLOR,
        helix_frac, sheet_frac, loop_frac
    )

    # t-SNE plots
    _save_scatter(
        os.path.join(out_dir, f"tsne_by_cath_{args.latent_rep}.png"),
        tsne_2d,
        f"t-SNE (AE {args.latent_rep}) colored by CATH",
        colors=cath_colors,
        point_size=pt_size,
        alpha=pt_alpha,
        legend=cath_legend,
    )
    _save_scatter(
        os.path.join(out_dir, f"tsne_by_length_{args.latent_rep}.png"),
        tsne_2d,
        f"t-SNE (AE {args.latent_rep}) colored by length",
        cvals=lengths,
        cmap="viridis",
        point_size=pt_size,
        alpha=pt_alpha,
        colorbar=True,
    )
    _save_scatter(
        os.path.join(out_dir, f"tsne_by_ss_simplex_{args.latent_rep}.png"),
        tsne_2d,
        f"t-SNE (AE {args.latent_rep}) colored by SS simplex",
        colors=ss_simplex_colors,
        point_size=pt_size,
        alpha=pt_alpha,
    )
    _save_scatter(
        os.path.join(out_dir, f"tsne_by_helix_frac_{args.latent_rep}.png"),
        tsne_2d,
        f"t-SNE (AE {args.latent_rep}) colored by helix fraction",
        cvals=helix_frac,
        cmap="viridis",
        point_size=pt_size,
        alpha=pt_alpha,
        colorbar=True,
    )
    _save_scatter(
        os.path.join(out_dir, f"tsne_by_sheet_frac_{args.latent_rep}.png"),
        tsne_2d,
        f"t-SNE (AE {args.latent_rep}) colored by sheet fraction",
        cvals=sheet_frac,
        cmap="viridis",
        point_size=pt_size,
        alpha=pt_alpha,
        colorbar=True,
    )
    _save_scatter(
        os.path.join(out_dir, f"tsne_by_loop_frac_{args.latent_rep}.png"),
        tsne_2d,
        f"t-SNE (AE {args.latent_rep}) colored by loop fraction",
        cvals=loop_frac,
        cmap="viridis",
        point_size=pt_size,
        alpha=pt_alpha,
        colorbar=True,
    )
    _save_hexbin(
        os.path.join(out_dir, f"tsne_density_hexbin_{args.latent_rep}.png"),
        tsne_2d,
        f"t-SNE density (AE {args.latent_rep})",
    )

    # UMAP plots
    _save_scatter(
        os.path.join(out_dir, f"umap_by_cath_{args.latent_rep}.png"),
        umap_2d,
        f"UMAP (AE {args.latent_rep}) colored by CATH",
        colors=cath_colors,
        point_size=pt_size,
        alpha=pt_alpha,
        legend=cath_legend,
    )
    _save_scatter(
        os.path.join(out_dir, f"umap_by_length_{args.latent_rep}.png"),
        umap_2d,
        f"UMAP (AE {args.latent_rep}) colored by length",
        cvals=lengths,
        cmap="viridis",
        point_size=pt_size,
        alpha=pt_alpha,
        colorbar=True,
    )
    _save_scatter(
        os.path.join(out_dir, f"umap_by_ss_simplex_{args.latent_rep}.png"),
        umap_2d,
        f"UMAP (AE {args.latent_rep}) colored by SS simplex",
        colors=ss_simplex_colors,
        point_size=pt_size,
        alpha=pt_alpha,
    )
    _save_scatter(
        os.path.join(out_dir, f"umap_by_helix_frac_{args.latent_rep}.png"),
        umap_2d,
        f"UMAP (AE {args.latent_rep}) colored by helix fraction",
        cvals=helix_frac,
        cmap="viridis",
        point_size=pt_size,
        alpha=pt_alpha,
        colorbar=True,
    )
    _save_scatter(
        os.path.join(out_dir, f"umap_by_sheet_frac_{args.latent_rep}.png"),
        umap_2d,
        f"UMAP (AE {args.latent_rep}) colored by sheet fraction",
        cvals=sheet_frac,
        cmap="viridis",
        point_size=pt_size,
        alpha=pt_alpha,
        colorbar=True,
    )
    _save_scatter(
        os.path.join(out_dir, f"umap_by_loop_frac_{args.latent_rep}.png"),
        umap_2d,
        f"UMAP (AE {args.latent_rep}) colored by loop fraction",
        cvals=loop_frac,
        cmap="viridis",
        point_size=pt_size,
        alpha=pt_alpha,
        colorbar=True,
    )
    _save_hexbin(
        os.path.join(out_dir, f"umap_density_hexbin_{args.latent_rep}.png"),
        umap_2d,
        f"UMAP density (AE {args.latent_rep})",
    )

    # Stats plots
    _save_hist(
        os.path.join(out_dir, f"hist_length_{args.latent_rep}.png"),
        lengths,
        f"Length histogram (AE {args.latent_rep})",
        bins=60,
    )
    _save_hist(
        os.path.join(out_dir, f"hist_helix_frac_{args.latent_rep}.png"),
        helix_frac,
        f"Helix fraction histogram (AE {args.latent_rep})",
        bins=60,
    )
    _save_hist(
        os.path.join(out_dir, f"hist_sheet_frac_{args.latent_rep}.png"),
        sheet_frac,
        f"Sheet fraction histogram (AE {args.latent_rep})",
        bins=60,
    )
    _save_hist(
        os.path.join(out_dir, f"hist_loop_frac_{args.latent_rep}.png"),
        loop_frac,
        f"Loop fraction histogram (AE {args.latent_rep})",
        bins=60,
    )

    # CATH-full frequency topK
    uniq, cnt = np.unique(cath_full_for_loader.astype(str), return_counts=True)
    order = np.argsort(-cnt)
    topk = min(30, uniq.shape[0])
    uniq_top = uniq[order][:topk]
    cnt_top = cnt[order][:topk]

    fig = plt.figure(figsize=(10.0, 6.0), dpi=150)
    ax = fig.add_subplot(111)
    ax.barh(np.arange(topk)[::-1], cnt_top[::-1])
    ax.set_yticks(np.arange(topk)[::-1])
    ax.set_yticklabels(uniq_top[::-1], fontsize=7)
    ax.set_title(f"CATH-full Top{topk} frequency (AE {args.latent_rep})", fontsize=11)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"cath_full_top{topk}_{args.latent_rep}.png"))
    plt.close(fig)

    print("[Plots] saved PNGs to", out_dir)
    print("[Done]")


if __name__ == "__main__":
    main()
