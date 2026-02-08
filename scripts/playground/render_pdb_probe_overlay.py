#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render overlay plot for unified PDB probe cache produced by probe_pdb_unified.py.

Expected npz keys (from your probe_pdb_unified.py):
  base_tsne_2d, base_umap_2d (optional), base_lengths (optional), ...
  probe_tsne_2d (optional), probe_umap_2d (optional),
  probe_groups, probe_paths (optional), probe_lengths (optional), ...

Usage examples:
  python scripts/playground/render_pdb_probe_overlay.py \
  --npz /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/latent_analysis_ae_sigmoid/class_pdb/probe_cache_ae_tokens_mean_class1_len_between_1_80_multi_4370_items.npz \
  --proj umap \
  --legend \
  --max_base 250000 \
  --out /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/latent_analysis_ae_sigmoid/class_pdb/overlay_umap.png


  python scripts/playground/render_pdb_probe_overlay.py \
    --npz /path/to/probe_cache_ae_tokens_mean_class1_len_between_1_80_XXXX.npz \
    --proj tsne \
    --out /path/to/overlay_tsne.png \
    --max_base 200000
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser("Render base+probe overlay from unified PDB probe cache (.npz)")
    p.add_argument("--npz", type=str, required=True, help="Unified probe cache .npz from probe_pdb_unified.py")
    p.add_argument("--proj", type=str, default="umap", choices=["umap", "tsne"], help="Which 2D embedding to plot")
    p.add_argument("--out", type=str, default="", help="Output image path (.png/.pdf). Default: alongside npz.")
    p.add_argument("--title", type=str, default="", help="Optional figure title")

    # Plot style
    p.add_argument("--figsize", type=float, nargs=2, default=(9.5, 8.0), help="Figure size, e.g. 10 8")
    p.add_argument("--dpi", type=int, default=250, help="Save DPI")

    # Base scatter
    p.add_argument("--max_base", type=int, default=250000, help="Max base points to draw (subsample for speed). <=0 means all")
    p.add_argument("--base_size", type=float, default=2.0, help="Base point size")
    p.add_argument("--base_alpha", type=float, default=0.12, help="Base point alpha")

    # Probe scatter
    p.add_argument("--probe_size", type=float, default=10.0, help="Probe point size")
    p.add_argument("--probe_alpha", type=float, default=0.90, help="Probe point alpha")

    # Legend / grouping
    p.add_argument("--legend", action="store_true", help="Show legend (recommended)")
    p.add_argument("--legend_max_groups", type=int, default=30, help="If too many groups, show at most N groups in legend")
    p.add_argument("--group_prefix_strip", action="store_true",
                   help="Strip common long prefixes in group names (best-effort)")

    # Repro
    p.add_argument("--seed", type=int, default=42, help="Seed for subsampling base points")
    return p.parse_args()


def _load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    cache = np.load(str(npz_path), allow_pickle=True)
    out = {k: cache[k] for k in cache.files}
    return out


def _pick_2d_arrays(cache: Dict[str, np.ndarray], proj: str) -> Tuple[np.ndarray, np.ndarray]:
    proj = proj.lower().strip()
    if proj == "umap":
        base_key = "base_umap_2d"
        probe_key = "probe_umap_2d"
    else:
        base_key = "base_tsne_2d"
        probe_key = "probe_tsne_2d"

    if base_key not in cache:
        raise KeyError(f"Missing key '{base_key}' in npz. Available: {list(cache.keys())[:20]} ...")
    if probe_key not in cache:
        raise KeyError(f"Missing key '{probe_key}' in npz. Available: {list(cache.keys())[:20]} ...")

    base_2d = np.asarray(cache[base_key], dtype=np.float32)
    probe_2d = np.asarray(cache[probe_key], dtype=np.float32)

    if base_2d.ndim != 2 or base_2d.shape[1] != 2:
        raise ValueError(f"{base_key} must be [N,2], got {base_2d.shape}")
    if probe_2d.ndim != 2 or probe_2d.shape[1] != 2:
        raise ValueError(f"{probe_key} must be [M,2], got {probe_2d.shape}")

    return base_2d, probe_2d


def _subsample(arr: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if max_n <= 0 or arr.shape[0] <= max_n:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=max_n, replace=False)
    return arr[idx]


def _best_effort_strip_prefix(names: np.ndarray) -> np.ndarray:
    # Heuristic: if all strings share a long common prefix, strip it.
    s = [str(x) for x in names.tolist()]
    if not s:
        return names
    pref = s[0]
    for t in s[1:]:
        # compute common prefix
        i = 0
        m = min(len(pref), len(t))
        while i < m and pref[i] == t[i]:
            i += 1
        pref = pref[:i]
        if len(pref) == 0:
            break
    # strip if prefix is "too long"
    if len(pref) >= 12:
        s2 = [x[len(pref):].lstrip("/_- .") for x in s]
        return np.array(s2, dtype=object)
    return names


def main():
    args = parse_args()
    np.random.seed(args.seed)

    npz_path = Path(args.npz).resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"npz not found: {npz_path}")

    cache = _load_npz(npz_path)
    base_2d, probe_2d = _pick_2d_arrays(cache, args.proj)

    # groups (required for your use-case)
    if "probe_groups" not in cache:
        raise KeyError("Missing key 'probe_groups' in npz. Your probe_pdb_unified.py should save it.")
    probe_groups = np.asarray(cache["probe_groups"], dtype=object).reshape(-1)
    if probe_groups.shape[0] != probe_2d.shape[0]:
        raise ValueError(f"probe_groups length mismatch: {probe_groups.shape[0]} vs probe_2d {probe_2d.shape[0]}")

    if args.group_prefix_strip:
        probe_groups = _best_effort_strip_prefix(probe_groups)

    # subsample base for speed
    base_plot = _subsample(base_2d, args.max_base, args.seed)

    # decide output path
    if args.out.strip():
        out_path = Path(args.out).resolve()
    else:
        out_name = f"overlay_{args.proj.lower()}.png"
        out_path = npz_path.parent / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # group -> indices
    uniq = np.unique(probe_groups)
    # stable order (as strings)
    uniq = np.array(sorted([str(u) for u in uniq]), dtype=object)

    # plot
    fig = plt.figure(figsize=tuple(args.figsize))
    ax = fig.add_subplot(111)

    ax.scatter(
        base_plot[:, 0], base_plot[:, 1],
        s=float(args.base_size),
        alpha=float(args.base_alpha),
        linewidths=0.0,
        label="base" if not args.legend else None,
    )

    # probe by group
    # Matplotlib will use default color cycle; we only control sizes/alpha.
    shown_groups = 0
    for g in uniq:
        m = (probe_groups.astype(str) == str(g))
        pts = probe_2d[m]
        if pts.size == 0:
            continue

        label = None
        if args.legend and shown_groups < int(args.legend_max_groups):
            label = f"{g} (n={pts.shape[0]})"
            shown_groups += 1

        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=float(args.probe_size),
            alpha=float(args.probe_alpha),
            linewidths=0.3,
            label=label,
        )

    # cosmetics
    if args.title.strip():
        ax.set_title(args.title.strip())
    else:
        # best-effort: use stored base_cache_path if present
        base_cache_path = cache.get("base_cache_path", None)
        if base_cache_path is not None:
            try:
                ax.set_title(f"Overlay ({args.proj.upper()}): base + probe")
            except Exception:
                pass

    ax.set_xlabel(args.proj.upper() + "-1")
    ax.set_ylabel(args.proj.upper() + "-2")
    ax.grid(False)

    if args.legend:
        ax.legend(
            loc="best",
            frameon=True,
            fontsize=8,
            markerscale=1.0,
            handletextpad=0.4,
            borderpad=0.3,
        )

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=int(args.dpi))
    plt.close(fig)

    print("[OK] Saved overlay figure:")
    print(" ", str(out_path))
    print(f"[Info] base points drawn: {base_plot.shape[0]} / {base_2d.shape[0]}")
    print(f"[Info] probe points: {probe_2d.shape[0]}, groups: {len(uniq)}")


if __name__ == "__main__":
    main()
