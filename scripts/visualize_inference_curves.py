#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust curve visualizer for VQ-VAE project.

Supported inputs per .npy:
  1) dict: {"curve_coords":[L,3], "ss_one_hot":[L,3]}  (includes 0-d object ndarray wrapping a dict)
  2) ndarray [L,6]: [xyz, ss_one_hot]
  3) ndarray [L,4]: [xyz, label] with label mapping: -1->H (helix), 1->E (sheet), 0->C (loop)

It draws a 3D polyline with segment colors by SS (H/E/C) and saves PNGs.

Usage:
  python /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/scripts/visualize_inference_curves.py \
  --dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/prior/diffusion_prior_recons_step190000 \
  --out_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/prior/diffusion_prior_recons_step190000/visualizations \
  --yaml /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/configs/stage2_vq.yaml

"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import yaml

INDEX_TO_SS = {0: "h", 1: "s", 2: "l"}        # H/E/C -> helix/sheet/loop
COLOR_MAP   = {"h": "red", "s": "green", "l": "blue"}

def set_equal_aspect_3d(ax, X: np.ndarray):
    # Ensure equal aspect for 3D axes
    x_max, y_max, z_max = X.max(axis=0)
    x_min, y_min, z_min = X.min(axis=0)
    ranges = np.array([x_max - x_min, y_max - y_min, z_max - z_min], dtype=np.float32)
    if not np.all(np.isfinite(ranges)) or float(np.max(ranges)) == 0.0:
        # fallback in degenerate cases
        ax.set_box_aspect((1, 1, 1))
        return
    max_range = float(ranges.max())
    x_mid = 0.5 * (x_max + x_min)
    y_mid = 0.5 * (y_max + y_min)
    z_mid = 0.5 * (z_max + z_min)
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

def _unwrap_object0d(obj):
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        try:
            return obj.item()
        except Exception:
            return obj
    return obj

def _labels_to_one_hot(labels: np.ndarray) -> np.ndarray:
    labels = np.rint(labels).astype(np.int32)
    L = labels.shape[0]
    ss = np.zeros((L, 3), dtype=np.float32)  # H,E,C
    ss[labels == -1, 0] = 1.0  # H
    ss[labels ==  1, 1] = 1.0  # E
    ss[labels ==  0, 2] = 1.0  # C
    return ss

def _to_one_hot_from_any(ss: np.ndarray) -> np.ndarray:
    ss = ss.astype(np.float32)
    if ss.ndim != 2 or ss.shape[1] != 3:
        raise ValueError(f"SS must be [L,3], got {ss.shape}")
    sums = ss.sum(axis=-1)
    is_binary = np.all((ss == 0.0) | (ss == 1.0))
    if is_binary and np.allclose(sums, 1.0):
        return ss
    idx = np.argmax(ss, axis=-1)
    L = ss.shape[0]
    oh = np.zeros((L, 3), dtype=np.float32)
    oh[np.arange(L), idx] = 1.0
    return oh

def load_curve_any(path: str):
    """
    Returns (xyz [L,3], ss_one_hot [L,3]) or (None, None) if unsupported.
    """
    try:
        obj = np.load(path, allow_pickle=True)
    except Exception:
        return None, None

    obj = _unwrap_object0d(obj)

    # Case 1: our dataset dict
    if isinstance(obj, dict) and "curve_coords" in obj and "ss_one_hot" in obj:
        xyz = np.asarray(obj["curve_coords"], dtype=np.float32)
        ss  = np.asarray(obj["ss_one_hot"],   dtype=np.float32)
        if xyz.ndim == 2 and xyz.shape[1] == 3 and ss.ndim == 2 and ss.shape[1] == 3 and xyz.shape[0] == ss.shape[0]:
            return xyz, _to_one_hot_from_any(ss)
        return None, None

    # Case 2: [L,6] ndarray
    if isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.shape[1] == 6:
        xyz = obj[:, :3].astype(np.float32)
        ss  = obj[:, 3:].astype(np.float32)
        return xyz, _to_one_hot_from_any(ss)

    # Case 3: [L,4] ndarray (xyz + label)
    if isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.shape[1] == 4:
        xyz = obj[:, :3].astype(np.float32)
        labels = obj[:, 3]
        ss = _labels_to_one_hot(labels)
        return xyz, ss

    # Single row flattened
    if isinstance(obj, np.ndarray) and obj.ndim == 1 and obj.size in (4, 6):
        arr = obj.reshape(1, -1)
        return load_curve_any_from_array(arr)

    return None, None

def load_curve_any_from_array(arr: np.ndarray):
    if arr.ndim != 2 or arr.shape[1] not in (4, 6):
        return None, None
    if arr.shape[1] == 4:
        xyz = arr[:, :3].astype(np.float32)
        ss  = _labels_to_one_hot(arr[:, 3])
        return xyz, ss
    xyz = arr[:, :3].astype(np.float32)
    ss  = _to_one_hot_from_any(arr[:, 3:])
    return xyz, ss

def read_max_seq_len_from_yaml(yaml_path: str):
    try:
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        return int(cfg.get("model_params", {}).get("max_seq_len", 0)) or None
    except Exception:
        return None

def plot_one(fname: str, in_dir: str, out_dir: str, max_seq_len: int = None):
    path = os.path.join(in_dir, fname)
    xyz, ss_onehot = load_curve_any(path)
    if xyz is None:
        print(f"[Skip] unsupported format for {fname}")
        return False

    L = xyz.shape[0]
    if max_seq_len is not None and L > max_seq_len:
        print(f"[Warn] {fname}: length {L} exceeds max_seq_len {max_seq_len} (yaml). Rendering anyway.")

    ss_idx = np.argmax(ss_onehot, axis=-1)               # [L] in {0,1,2}
    ss_labels = [INDEX_TO_SS.get(int(i), "l") for i in ss_idx]
    ss_colors = [COLOR_MAP.get(lbl, "blue") for lbl in ss_labels]

    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax  = fig.add_subplot(111, projection="3d")

    # colored segments
    for i in range(L - 1):
        ax.plot(xyz[i:i+2, 0], xyz[i:i+2, 1], xyz[i:i+2, 2], color=ss_colors[i], linewidth=2)
    # scatter
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=ss_colors, s=18, alpha=0.9)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Curve: {fname}")

    legend_elements = [
        Line2D([0], [0], color="red",   lw=2, label="Helix (h)"),
        Line2D([0], [0], color="green", lw=2, label="Sheet (s)"),
        Line2D([0], [0], color="blue",  lw=2, label="Loop (l)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    set_equal_aspect_3d(ax, xyz)
    plt.tight_layout()

    out_name = os.path.splitext(fname)[0] + ".png"
    dst = os.path.join(out_dir, out_name)
    plt.savefig(dst, dpi=300)
    plt.close(fig)
    print(f"Saved: {dst}")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir",      type=str, required=False,
                    default="/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/test_new_curve",
                    help="Directory containing .npy files to visualize.")
    ap.add_argument("--out_dir",  type=str, required=False, default=None,
                    help="Output directory for PNGs (default: <dir>/visualizations).")
    ap.add_argument("--yaml",     type=str, required=False, default=None,
                    help="Optional yaml to check max_seq_len and warn if exceeded.")
    ap.add_argument("--pattern",  type=str, required=False, default="*.npy",
                    help="Filename pattern to include (glob-like, but simple .endswith handled).")
    args = ap.parse_args()

    in_dir = args.dir
    out_dir = args.out_dir or os.path.join(in_dir, "visualizations")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # read yaml (optional)
    max_seq_len = None
    if args.yaml and os.path.exists(args.yaml):
        max_seq_len = read_max_seq_len_from_yaml(args.yaml)
        if max_seq_len:
            print(f"[Info] max_seq_len from yaml: {max_seq_len}")

    # list npy files
    npy_files = sorted([f for f in os.listdir(in_dir) if f.endswith(".npy")])
    print(f"Found {len(npy_files)} .npy files to visualize.")

    cnt = 0
    for fname in npy_files:
        ok = plot_one(fname, in_dir, out_dir, max_seq_len)
        if ok:
            cnt += 1
    print(f"Done. Rendered {cnt}/{len(npy_files)} files to {out_dir}.")

if __name__ == "__main__":
    main()
