#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
python scripts/playground/render_base_probe_overlay.py \
  --npz /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/latent_analysis_ae_sigmoid/class1/probe_cache_ae_tokens_mean_class1_len_between_1_80_decoded_curves_120_filtered.npz \
  --proj umap \
  --density \
  --out_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/latent_analysis_ae_sigmoid/class1 \
  --prefix decoded122_mean \
  --base_alpha 0.35 \
  --probe_size 12 \
  --probe_alpha 0.7


python scripts/playground/render_base_probe_overlay.py \
  --npz /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/latent_analysis_ae/class1_mean/probe_cache_ae_tokens_mean_class1_len_between_1_80_decoded_curves_120_filtered.npz \
  --proj tsne \
  --density \
  --out_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/latent_analysis_ae/class1_mean \
  --prefix decoded120_mean \
  --base_alpha 0.12 \
  --probe_size 12 \
  --probe_alpha 0.9

'''
import os
import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser("Render base/probe overlay plots from unified probe cache (.npz)")

    p.add_argument("--npz", type=str, required=True, help="Unified probe cache produced by probe_pdb_unified.py")
    p.add_argument("--proj", type=str, default="umap", choices=["umap", "tsne"], help="Which 2D projection to plot")

    p.add_argument("--out_dir", type=str, default="", help="Output directory (default: alongside npz)")
    p.add_argument("--prefix", type=str, default="", help="Optional filename prefix")

    # style
    p.add_argument("--base_alpha", type=float, default=0.15)
    p.add_argument("--probe_alpha", type=float, default=0.85)
    p.add_argument("--base_size", type=float, default=3.0)
    p.add_argument("--probe_size", type=float, default=10.0)

    p.add_argument("--base_color", type=str, default="#94a3b8", help="Base points color (hex)")
    p.add_argument("--probe_color", type=str, default="#ef4444", help="Probe points color (hex)")

    # options
    p.add_argument("--by_group", action="store_true",
                   help="Color probe points by probe_groups (instead of single probe_color)")
    p.add_argument("--legend", action="store_true", help="Show legend (useful with --by_group)")

    # density background
    p.add_argument("--density", action="store_true",
                   help="Also plot base density (hexbin) + probe points")
    p.add_argument("--hex_gridsize", type=int, default=70)

    p.add_argument("--dpi", type=int, default=180)

    return p.parse_args()


def _load_2d(cache: np.lib.npyio.NpzFile, name: str):
    """
    Read 2D embedding from npz; returns None if missing or saved as object(None).
    """
    if name not in cache.files:
        return None
    arr = cache[name]
    # handle object(None)
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        v = arr.item()
        if v is None:
            return None
        arr = v
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    return arr.astype(np.float32, copy=False)


def _ensure_obj_array(x):
    if x is None:
        return None
    x = np.asarray(x)
    if x.dtype == object:
        return x
    return x.astype(object)


def _group_color_map(groups):
    """
    Deterministic colors for groups using matplotlib tab10/tab20 cycle.
    """
    uniq = []
    seen = set()
    for g in groups.tolist():
        if g not in seen:
            uniq.append(g)
            seen.add(g)

    cmap = plt.get_cmap("tab20")
    color_map = {}
    for i, g in enumerate(uniq):
        color_map[g] = cmap(i % cmap.N)
    return color_map, uniq


def render_overlay(X_base, X_probe, out_png, title,
                   base_color, probe_color,
                   base_alpha, probe_alpha,
                   base_size, probe_size,
                   probe_groups=None,
                   by_group=False,
                   show_legend=False,
                   dpi=180):
    fig = plt.figure(figsize=(7.6, 6.4), dpi=dpi)
    ax = fig.add_subplot(111)

    # base
    ax.scatter(
        X_base[:, 0], X_base[:, 1],
        s=base_size, c=base_color, alpha=base_alpha,
        linewidths=0,
        label="Base"
    )

    # probe
    if by_group and probe_groups is not None:
        probe_groups = _ensure_obj_array(probe_groups)
        color_map, uniq = _group_color_map(probe_groups)

        for g in uniq:
            mask = (probe_groups == g)
            ax.scatter(
                X_probe[mask, 0], X_probe[mask, 1],
                s=probe_size, alpha=probe_alpha,
                linewidths=0,
                c=[color_map[g]],
                label=str(g)
            )
    else:
        ax.scatter(
            X_probe[:, 0], X_probe[:, 1],
            s=probe_size, c=probe_color, alpha=probe_alpha,
            linewidths=0,
            label="Probe"
        )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Dim 1", fontsize=9)
    ax.set_ylabel("Dim 2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(False)

    if show_legend:
        ax.legend(loc="best", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def render_density_base_with_probe(X_base, X_probe, out_png, title,
                                  probe_color,
                                  probe_alpha, probe_size,
                                  gridsize=70,
                                  probe_groups=None,
                                  by_group=False,
                                  show_legend=False,
                                  dpi=180):
    fig = plt.figure(figsize=(7.6, 6.4), dpi=dpi)
    ax = fig.add_subplot(111)

    hb = ax.hexbin(
        X_base[:, 0], X_base[:, 1],
        gridsize=int(gridsize),
        mincnt=1
    )
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)

    if by_group and probe_groups is not None:
        probe_groups = _ensure_obj_array(probe_groups)
        color_map, uniq = _group_color_map(probe_groups)
        for g in uniq:
            mask = (probe_groups == g)
            ax.scatter(
                X_probe[mask, 0], X_probe[mask, 1],
                s=probe_size, alpha=probe_alpha,
                linewidths=0,
                c=[color_map[g]],
                label=str(g)
            )
    else:
        ax.scatter(
            X_probe[:, 0], X_probe[:, 1],
            s=probe_size, alpha=probe_alpha,
            linewidths=0,
            c=probe_color,
            label="Probe"
        )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Dim 1", fontsize=9)
    ax.set_ylabel("Dim 2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(False)

    if show_legend:
        ax.legend(loc="best", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    args = parse_args()

    npz_path = Path(args.npz).resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"npz not found: {npz_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix.strip()
    if prefix:
        prefix = prefix + "_"

    cache = np.load(str(npz_path), allow_pickle=True)

    # pick projection
    proj = args.proj.lower()
    if proj == "umap":
        X_base = _load_2d(cache, "base_umap_2d")
        X_probe = _load_2d(cache, "probe_umap_2d")
        proj_name = "umap"
    else:
        X_base = _load_2d(cache, "base_tsne_2d")
        X_probe = _load_2d(cache, "probe_tsne_2d")
        proj_name = "tsne"

    if X_base is None or X_probe is None:
        missing = []
        if X_base is None:
            missing.append(f"base_{proj_name}_2d")
        if X_probe is None:
            missing.append(f"probe_{proj_name}_2d")
        raise RuntimeError(
            f"Missing required 2D arrays in npz: {missing}\n"
            f"Tip: if probe_tsne_2d is missing, you likely ran probe with --only_umap."
        )

    probe_groups = cache["probe_groups"] if "probe_groups" in cache.files else None

    # title info
    latent_rep = ""
    if "latent_rep" in cache.files:
        try:
            latent_rep = str(cache["latent_rep"].item())
        except Exception:
            latent_rep = str(cache["latent_rep"])
    title = f"{proj_name.upper()} overlay (latent_rep={latent_rep}) | base={X_base.shape[0]} probe={X_probe.shape[0]}"

    out_png1 = out_dir / f"{prefix}{proj_name}_overlay.png"
    render_overlay(
        X_base=X_base,
        X_probe=X_probe,
        out_png=str(out_png1),
        title=title,
        base_color=args.base_color,
        probe_color=args.probe_color,
        base_alpha=float(args.base_alpha),
        probe_alpha=float(args.probe_alpha),
        base_size=float(args.base_size),
        probe_size=float(args.probe_size),
        probe_groups=probe_groups,
        by_group=bool(args.by_group),
        show_legend=bool(args.legend),
        dpi=int(args.dpi),
    )
    print("[Saved]", str(out_png1))

    if args.density:
        out_png2 = out_dir / f"{prefix}{proj_name}_density_base_plus_probe.png"
        render_density_base_with_probe(
            X_base=X_base,
            X_probe=X_probe,
            out_png=str(out_png2),
            title=f"{proj_name.upper()} base density + probe (latent_rep={latent_rep})",
            probe_color=args.probe_color,
            probe_alpha=float(args.probe_alpha),
            probe_size=float(args.probe_size),
            gridsize=int(args.hex_gridsize),
            probe_groups=probe_groups,
            by_group=bool(args.by_group),
            show_legend=bool(args.legend),
            dpi=int(args.dpi),
        )
        print("[Saved]", str(out_png2))

    print("[Done]")


if __name__ == "__main__":
    main()
