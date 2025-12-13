#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example:

python scripts/plot_probe_cache_example.py \
  --cache latent_analysis/class1/probe_cache_class1_len_between_1_80_example.npz

This script:
  - Loads a probe cache produced by probe_pdb_unified.py
  - The cache is expected to contain:
      tsne_2d           [N_bg, 2]  background TSNE coords
      umap_2d           [N_bg, 2]  background UMAP coords
      probe_tsne_2d     [N_probe, 2]  probe TSNE coords (optional)
      probe_umap_2d     [N_probe, 2]  probe UMAP coords (optional)
      probe_groups      [N_probe]  group name per probe (optional)
      probe_pdb_paths   [N_probe]  original pdb paths (optional)

  - Generates simple example plots:
      plots/tsne_probes_by_group.png
      plots/umap_probes_by_group.png

Examples:
python scripts/plot_probe_cache_example.py \
  --cache latent_analysis/class1/probe_cache/probe_cache_class1_len_between_1_80_multi_4370_pdbs.npz

"""

import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_OUT_DIR = "/public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/examples/bb-crv_sp-probe-res/plot"


def parse_args():
    p = argparse.ArgumentParser("Plot probe cache with TSNE and UMAP")
    p.add_argument(
        "--cache",
        type=str,
        required=True,
        help="Path to probe cache npz file produced by probe_pdb_unified.py",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory for plots",
    )
    p.add_argument(
        "--no_tsne",
        action="store_true",
        help="Disable TSNE plots",
    )
    p.add_argument(
        "--no_umap",
        action="store_true",
        help="Disable UMAP plots",
    )
    return p.parse_args()


def get_array(cache, key, required=False):
    if key in cache.files:
        return cache[key]
    if required:
        raise KeyError("Key '{}' not found in cache".format(key))
    return None


def get_unique_groups(groups):
    uniq = []
    for g in groups:
        if g not in uniq:
            uniq.append(g)
    return uniq


def make_group_style(groups):
    color_cycle = [
        "#ef4444",
        "#3b82f6",
        "#22c55e",
        "#f97316",
        "#a855f7",
        "#eab308",
        "#06b6d4",
        "#ec4899",
        "#4b5563",
    ]
    marker_cycle = ["o", "s", "D", "^", "v", "P", "X", "*", "h"]

    unique = get_unique_groups(groups)
    group_color = {}
    group_marker = {}
    for i, g in enumerate(unique):
        group_color[g] = color_cycle[i % len(color_cycle)]
        group_marker[g] = marker_cycle[i % len(marker_cycle)]
    return unique, group_color, group_marker


def plot_embedding_all(
    bg_2d,
    probe_2d,
    probe_groups,
    out_png,
    title,
    x_label,
    y_label,
):
    groups = list(probe_groups)
    unique_groups, group_color, group_marker = make_group_style(groups)

    fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=200)

    if bg_2d is not None and bg_2d.size > 0:
        ax.scatter(
            bg_2d[:, 0],
            bg_2d[:, 1],
            s=4,
            c="#d1d5db",
            alpha=0.7,
            edgecolors="none",
            label="_background",
        )

    for coord, g in zip(probe_2d, groups):
        c = group_color.get(g, "#000000")
        m = group_marker.get(g, "o")
        ax.scatter(
            coord[0],
            coord[1],
            s=40,
            c=c,
            marker=m,
            alpha=0.95,
            edgecolors="none",
        )

    handles = []
    for g in unique_groups:
        c = group_color.get(g, "#000000")
        m = group_marker.get(g, "o")
        handles.append(
            Line2D(
                [0],
                [0],
                marker=m,
                linestyle="None",
                markerfacecolor=c,
                markeredgecolor="none",
                markersize=7,
                label=str(g),
            )
        )
    if handles:
        ax.legend(
            handles=handles,
            title="Probe groups",
            loc="best",
            fontsize=7,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_embedding_per_group(
    bg_2d,
    probe_2d,
    probe_groups,
    out_dir,
    prefix,
    x_label,
    y_label,
):
    groups = np.asarray(probe_groups)
    unique_groups, group_color, group_marker = make_group_style(groups)

    for g in unique_groups:
        mask = (groups == g)
        if not np.any(mask):
            continue
        coords_g = probe_2d[mask]

        safe_g = str(g).replace(os.sep, "_").replace(" ", "_")
        out_png = os.path.join(out_dir, "{}_group_{}.png".format(prefix, safe_g))

        fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=200)

        if bg_2d is not None and bg_2d.size > 0:
            ax.scatter(
                bg_2d[:, 0],
                bg_2d[:, 1],
                s=4,
                c="#d1d5db",
                alpha=0.7,
                edgecolors="none",
                label="_background",
            )

        c = group_color.get(g, "#000000")
        m = group_marker.get(g, "o")
        ax.scatter(
            coords_g[:, 0],
            coords_g[:, 1],
            s=40,
            c=c,
            marker=m,
            alpha=0.95,
            edgecolors="none",
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("{} (group = {})".format(prefix, g))

        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)


def main():
    args = parse_args()

    cache_path = os.path.abspath(args.cache)
    if not os.path.isfile(cache_path):
        raise FileNotFoundError("Cache file not found: {}".format(cache_path))

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("[Plot] cache:", cache_path)
    print("[Plot] out_dir:", out_dir)

    cache = np.load(cache_path, allow_pickle=True)

    # Required probe arrays
    probe_tsne_2d = get_array(cache, "probe_tsne_2d", required=False)
    probe_umap_2d = get_array(cache, "probe_umap_2d", required=False)
    probe_groups = get_array(cache, "probe_groups", required=True)

    probe_groups = np.asarray(probe_groups)

    print("[Plot] num probe points:", probe_groups.shape[0])
    print("[Plot] keys in cache:", cache.files)

    # Background embeddings
    base_tsne_2d = get_array(cache, "base_tsne_2d", required=False)
    base_umap_2d = get_array(cache, "base_umap_2d", required=False)

    # TSNE plots
    if not args.no_tsne:
        if probe_tsne_2d is None:
            print("[TSNE] probe_tsne_2d not found in cache; skip TSNE plots")
        else:
            print("[TSNE] probe points:", probe_tsne_2d.shape[0])
            if base_tsne_2d is None:
                print("[TSNE] base_tsne_2d not found; plotting probes without background")
            else:
                print("[TSNE] background points:", base_tsne_2d.shape[0])

            out_all_tsne = os.path.join(out_dir, "tsne_all_groups.png")
            plot_embedding_all(
                bg_2d=base_tsne_2d,
                probe_2d=probe_tsne_2d,
                probe_groups=probe_groups,
                out_png=out_all_tsne,
                title="TSNE background with all probe groups",
                x_label="TSNE dim-1",
                y_label="TSNE dim-2",
            )

            plot_embedding_per_group(
                bg_2d=base_tsne_2d,
                probe_2d=probe_tsne_2d,
                probe_groups=probe_groups,
                out_dir=out_dir,
                prefix="tsne",
                x_label="TSNE dim-1",
                y_label="TSNE dim-2",
            )

    # UMAP plots
    if not args.no_umap:
        if probe_umap_2d is None:
            print("[UMAP] probe_umap_2d not found in cache; skip UMAP plots")
        else:
            print("[UMAP] probe points:", probe_umap_2d.shape[0])
            if base_umap_2d is None:
                print("[UMAP] base_umap_2d not found; plotting probes without background")
            else:
                print("[UMAP] background points:", base_umap_2d.shape[0])

            out_all_umap = os.path.join(out_dir, "umap_all_groups.png")
            plot_embedding_all(
                bg_2d=base_umap_2d,
                probe_2d=probe_umap_2d,
                probe_groups=probe_groups,
                out_png=out_all_umap,
                title="UMAP background with all probe groups",
                x_label="UMAP dim-1",
                y_label="UMAP dim-2",
            )

            plot_embedding_per_group(
                bg_2d=base_umap_2d,
                probe_2d=probe_umap_2d,
                probe_groups=probe_groups,
                out_dir=out_dir,
                prefix="umap",
                x_label="UMAP dim-1",
                y_label="UMAP dim-2",
            )

    print("[Plot] done.")


if __name__ == "__main__":
    main()