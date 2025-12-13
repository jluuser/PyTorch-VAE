#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze a t-SNE cache produced by visualize_latent_and_codebook*.py.

Inputs (from cache .npz):
  - tsne_2d:       [N,2] t-SNE coordinates
  - latents:       [N,D] encoder sequence latents (continuous space)
  - lengths:       [N]   sequence lengths
  - helix_frac:    [N]
  - sheet_frac:    [N]
  - loop_frac:     [N]
  - labels:        [N]   optional top-level CATH class (1,2,3,4,6)
  - cath_full:     [N]   optional full CATH id string (e.g. "1.10.420.10")
  - rel_paths:     [N]   optional relative curve paths under data_dir

Outputs (saved under <cache_dir>/analysis/):
  - tsne_plain.png
  - tsne_len_continuous.png
  - tsne_helix_frac.png
  - tsne_sheet_frac.png
  - tsne_loop_frac.png
  - tsne_ss_argmax.png
  - tsne_cath_topclass.png (if labels exist)
  - tsne_fold_topK.png (if cath_full exists)
  - tsne_clusters.png
  - hist_*_cluster_<id>.png (per cluster)
  - cluster_summary.txt
  - island_curves.txt              (all non-main clusters, if rel_paths exists)
  - cluster_<id>_curves.txt        (per cluster, if rel_paths exists)

Additionally, if "latents" is present in the cache, this script will also:
  - run PCA on latents and save:
      pca_2d.npy
      pca_plain.png
      pca_len_continuous.png
      pca_helix_frac.png
      pca_sheet_frac.png
      pca_loop_frac.png
      pca_ss_argmax.png
      pca_cath_topclass.png
  - run UMAP (if the "umap-learn" package is installed) and save analogous
    files with prefix "umap_*.png" and "umap_2d.npy".

Example:

python scripts/analyze_tsne_cache2.py \
  --cache latent_analysis/class1/tsne_cache_vq_class1_len_between_1_80.npz \
  --n_clusters 6 \
  --top_k_folds 5
"""

import os
import argparse
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def load_cache(cache_path):
    """Load npz cache and return the NpzFile object."""
    cache = np.load(cache_path, allow_pickle=True)
    return cache


def ensure_analysis_dir(cache_path):
    """Create an 'analysis' directory next to the cache file."""
    cache_dir = os.path.dirname(os.path.abspath(cache_path))
    analysis_dir = os.path.join(cache_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    return analysis_dir


def get_array_or_none(cache, key):
    """Safely read an array from npz; return None if not present."""
    if key in cache.files:
        return cache[key]
    return None


# ---------------------------------------------------------------------
# 2D scatter helpers (generic, not t-SNE specific)
# ---------------------------------------------------------------------


def plot_tsne_plain(lat2d, out_png, title="2D scatter (plain)"):
    """Basic shape-only scatter with a single color."""
    fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=140)
    ax.scatter(
        lat2d[:, 0],
        lat2d[:, 1],
        s=4,
        c="#4b5563",
        alpha=0.8,
        edgecolors="none",
    )
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_tsne_continuous(lat2d, values, out_png, title, cbar_label):
    """Scatter colored by a continuous scalar (e.g. length, helix_frac)."""
    vals = np.asarray(values, dtype=np.float32)
    assert lat2d.shape[0] == vals.shape[0]

    fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=140)
    sc = ax.scatter(
        lat2d[:, 0],
        lat2d[:, 1],
        s=4,
        c=vals,
        cmap="viridis",
        alpha=0.8,
        edgecolors="none",
    )
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_tsne_ss_argmax(lat2d, helix_frac, sheet_frac, loop_frac, out_png, title):
    """Scatter colored by argmax of (helix_frac, sheet_frac, loop_frac)."""
    h = np.asarray(helix_frac, dtype=np.float32)
    s = np.asarray(sheet_frac, dtype=np.float32)
    l = np.asarray(loop_frac, dtype=np.float32)
    assert lat2d.shape[0] == h.shape[0] == s.shape[0] == l.shape[0]

    w = np.stack([h, s, l], axis=1)  # [N,3]
    winner = np.argmax(w, axis=1)    # 0=helix,1=sheet,2=loop

    colors = {
        0: "tab:red",
        1: "tab:green",
        2: "tab:blue",
    }
    labels = {
        0: "Helix-dominant",
        1: "Sheet-dominant",
        2: "Loop-dominant",
    }

    point_colors = [colors[int(c)] for c in winner]

    fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=140)
    ax.scatter(
        lat2d[:, 0],
        lat2d[:, 1],
        s=4,
        c=point_colors,
        alpha=0.8,
        edgecolors="none",
    )
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.set_title(title)

    from matplotlib.lines import Line2D
    handles = []
    for k in [0, 1, 2]:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=labels[k],
                markerfacecolor=colors[k],
                markersize=6,
            )
        )
    ax.legend(handles=handles, title="Dominant SS", loc="best")

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_tsne_cath_topclass(lat2d, labels, out_png, title):
    """Scatter colored by top-level CATH class (1,2,3,4,6)."""
    if labels is None:
        print("[Warn] labels is None, skip CATH topclass plot")
        return

    lab = np.asarray(labels, dtype=np.int64)
    mask = lab >= 0
    if not np.any(mask):
        print("[Warn] no valid labels (>=0), skip CATH topclass plot")
        return

    lat2d = lat2d[mask]
    lab = lab[mask]

    class_names = {
        1: "1 Mainly Alpha",
        2: "2 Mainly Beta",
        3: "3 Alpha-Beta",
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

    point_colors = [class_colors.get(int(c), "gray") for c in lab]

    fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=140)
    ax.scatter(
        lat2d[:, 0],
        lat2d[:, 1],
        s=4,
        c=point_colors,
        alpha=0.8,
        edgecolors="none",
    )
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.set_title(title)

    from matplotlib.lines import Line2D
    unique_classes = sorted(set(lab.tolist()))
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
    ax.legend(handles=handles, title="CATH top class", loc="best")

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def fold_prefix_from_cath_full(cath_id):
    """
    Extract fold prefix from a full CATH id string, e.g.
      "1.10.420.10" -> "1.10"
    If parsing fails, return "unknown".
    """
    if cath_id is None:
        return "unknown"
    if not isinstance(cath_id, str):
        try:
            cath_id = str(cath_id)
        except Exception:
            return "unknown"
    parts = cath_id.split(".")
    if len(parts) < 2:
        return "unknown"
    return parts[0] + "." + parts[1]


def plot_tsne_fold_topk(lat2d, cath_full, out_png, title, top_k=5):
    """Scatter colored by top-K most frequent fold prefixes."""
    if cath_full is None:
        print("[Warn] cath_full is None, skip fold_topK plot")
        return

    cath_full = np.asarray(cath_full)
    if cath_full.shape[0] != lat2d.shape[0]:
        print("[Warn] cath_full length mismatch, skip fold_topK plot")
        return

    fold_prefixes = [fold_prefix_from_cath_full(c) for c in cath_full]
    counts = Counter(fold_prefixes)

    if "unknown" in counts:
        unknown_count = counts.pop("unknown")
    else:
        unknown_count = 0

    if not counts:
        print("[Warn] only unknown folds, skip fold_topK plot")
        return

    most_common = counts.most_common(top_k)
    keep_folds = [k for k, _ in most_common]
    keep_set = set(keep_folds)

    fold_to_index = {f: i for i, f in enumerate(keep_folds)}
    other_index = len(keep_folds)

    num_groups = len(keep_folds) + 1
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(num_groups)]

    point_colors = []
    group_labels = []
    for fp in fold_prefixes:
        if fp in keep_set:
            gid = fold_to_index[fp]
        else:
            gid = other_index
        point_colors.append(colors[gid])
        group_labels.append(gid)

    group_labels = np.array(group_labels, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=140)
    ax.scatter(
        lat2d[:, 0],
        lat2d[:, 1],
        s=4,
        c=point_colors,
        alpha=0.8,
        edgecolors="none",
    )
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.set_title(title)

    from matplotlib.lines import Line2D
    handles = []
    for i, f in enumerate(keep_folds):
        label = "{} (n={})".format(f, counts[f])
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=label,
                markerfacecolor=colors[i],
                markersize=6,
            )
        )
    others_count = sum(v for k, v in counts.items() if k not in keep_set) + unknown_count
    label_others = "others (n={})".format(others_count)
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label_others,
            markerfacecolor=colors[other_index],
            markersize=6,
        )
    )

    ax.legend(handles=handles, title="Fold prefix (top-{})".format(top_k), loc="best")

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ---------------------------------------------------------------------
# Clustering on 2D coordinates (KMeans) and cluster-level analysis
# ---------------------------------------------------------------------


def cluster_tsne_kmeans(lat2d, n_clusters=6, random_state=42):
    """Run KMeans on 2D coordinates, return cluster labels [N]."""
    N = lat2d.shape[0]
    if N < n_clusters:
        n_clusters = max(1, N)
    if n_clusters <= 1:
        print("[Cluster] N={}, using a single cluster".format(N))
        return np.zeros(N, dtype=np.int64)

    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = km.fit_predict(lat2d)
    return labels.astype(np.int64)


def plot_tsne_clusters(lat2d, cluster_labels, out_png, title):
    """2D scatter colored by KMeans cluster id."""
    lab = np.asarray(cluster_labels, dtype=np.int64)
    assert lab.shape[0] == lat2d.shape[0]

    num_clusters = int(lab.max()) + 1
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(num_clusters)]

    point_colors = [colors[int(c)] for c in lab]

    fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=140)
    ax.scatter(
        lat2d[:, 0],
        lat2d[:, 1],
        s=4,
        c=point_colors,
        alpha=0.8,
        edgecolors="none",
    )
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.set_title(title)

    from matplotlib.lines import Line2D
    handles = []
    counts = Counter(lab.tolist())
    for c in range(num_clusters):
        label = "cluster {} (n={})".format(c, counts.get(c, 0))
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=label,
                markerfacecolor=colors[c],
                markersize=6,
            )
        )
    ax.legend(handles=handles, title="Clusters", loc="best", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_cluster_hist(values, out_png, title, xlabel):
    """Simple histogram plot for one cluster."""
    vals = np.asarray(values, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=140)
    ax.hist(vals, bins=40, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def write_cluster_summary(
    analysis_dir,
    cluster_labels,
    lengths,
    helix_frac,
    sheet_frac,
    loop_frac,
    labels,
    cath_full,
    rel_paths,
):
    """
    Write cluster_summary.txt and island_curves.txt (if rel_paths exist).
    Largest cluster is considered "mainland", others "islands".
    """
    lab = np.asarray(cluster_labels, dtype=np.int64)
    N = lab.shape[0]
    counts = Counter(lab.tolist())
    sorted_clusters = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    main_cluster = sorted_clusters[0][0]

    print(
        "[Cluster] total points = {}, clusters = {}, main cluster = {} (n={})".format(
            N, len(counts), main_cluster, counts[main_cluster]
        )
    )

    summary_path = os.path.join(analysis_dir, "cluster_summary.txt")
    with open(summary_path, "w") as f:
        f.write("# Cluster summary\n")
        f.write("# Total points: {}\n".format(N))
        f.write("# Number of clusters: {}\n".format(len(counts)))
        f.write("# Main cluster (largest): {}\n".format(main_cluster))
        f.write("#\n")

        for cid, size in sorted_clusters:
            frac = size / float(N)
            idx = np.where(lab == cid)[0]

            len_sub = (
                np.asarray(lengths, dtype=np.float32)[idx]
                if lengths is not None
                else None
            )
            h_sub = (
                np.asarray(helix_frac, dtype=np.float32)[idx]
                if helix_frac is not None
                else None
            )
            s_sub = (
                np.asarray(sheet_frac, dtype=np.float32)[idx]
                if sheet_frac is not None
                else None
            )
            l_sub = (
                np.asarray(loop_frac, dtype=np.float32)[idx]
                if loop_frac is not None
                else None
            )

            f.write("Cluster {}:\n".format(cid))
            f.write("  size = {} ({:.4f} of total)\n".format(size, frac))

            if len_sub is not None:
                f.write(
                    "  length: mean={:.2f}, median={:.2f}, min={:.2f}, max={:.2f}\n".format(
                        float(len_sub.mean()),
                        float(np.median(len_sub)),
                        float(len_sub.min()),
                        float(len_sub.max()),
                    )
                )
            if h_sub is not None and s_sub is not None and l_sub is not None:
                f.write(
                    "  helix_frac: mean={:.3f}, sheet_frac: mean={:.3f}, loop_frac: mean={:.3f}\n".format(
                        float(h_sub.mean()),
                        float(s_sub.mean()),
                        float(l_sub.mean()),
                    )
                )

            if labels is not None:
                lab_sub = np.asarray(labels, dtype=np.int64)[idx]
                c_counts = Counter(lab_sub.tolist())
                f.write("  CATH top-class distribution:\n")
                for k, v in sorted(c_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(
                        "    class {}: n={} ({:.4f})\n".format(
                            k, v, v / float(size)
                        )
                    )

            if cath_full is not None:
                cf_sub = np.asarray(cath_full)[idx]
                fp_sub = [fold_prefix_from_cath_full(c) for c in cf_sub]
                fp_counts = Counter(fp_sub)
                f.write("  Fold prefix distribution (top 10):\n")
                for k, v in fp_counts.most_common(10):
                    f.write(
                        "    {}: n={} ({:.4f})\n".format(
                            k, v, v / float(size)
                        )
                    )

            f.write("\n")

    print("[Cluster] summary written to {}".format(summary_path))

    if rel_paths is not None:
        rel_paths_arr = np.asarray(rel_paths)
        island_idx = np.where(lab != main_cluster)[0]
        island_path = os.path.join(analysis_dir, "island_curves.txt")
        with open(island_path, "w") as f:
            f.write("# Island curves (all clusters except the largest)\n")
            f.write("# Columns: global_index\tcluster_id\trel_path\n")
            for i in island_idx:
                if i < rel_paths_arr.shape[0]:
                    name = str(rel_paths_arr[i])
                else:
                    name = "idx_{}".format(i)
                f.write("{}\t{}\t{}\n".format(i, int(lab[i]), name))
        print("[Cluster] island curve list written to {}".format(island_path))

        for cid, size in sorted_clusters:
            idx = np.where(lab == cid)[0]
            out_txt = os.path.join(
                analysis_dir, "cluster_{}_curves.txt".format(cid)
            )
            with open(out_txt, "w") as f:
                f.write("# Curves in cluster {}\n".format(cid))
                f.write("# Columns: global_index\trel_path\n")
                for i in idx:
                    if i < rel_paths_arr.shape[0]:
                        name = str(rel_paths_arr[i])
                    else:
                        name = "idx_{}".format(i)
                    f.write("{}\t{}\n".format(i, name))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser("Analyze a t-SNE cache (latent_analysis/*.npz)")
    p.add_argument(
        "--cache",
        type=str,
        required=True,
        help="Path to tsne_cache_*.npz",
    )
    p.add_argument(
        "--n_clusters",
        type=int,
        default=6,
        help="Number of KMeans clusters on t-SNE coordinates",
    )
    p.add_argument(
        "--top_k_folds",
        type=int,
        default=5,
        help="Top-K fold prefixes to visualize in fold-topK plot",
    )
    p.add_argument(
        "--no_pca",
        action="store_true",
        help="Disable PCA analysis even if latents are present.",
    )
    p.add_argument(
        "--no_umap",
        action="store_true",
        help="Disable UMAP analysis even if the package is installed.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cache_path = os.path.abspath(args.cache)
    if not os.path.isfile(cache_path):
        raise FileNotFoundError("Cache file not found: {}".format(cache_path))

    print("[Analyze] cache:", cache_path)

    cache = load_cache(cache_path)
    analysis_dir = ensure_analysis_dir(cache_path)
    print("[Analyze] outputs will be saved under:", analysis_dir)

    tsne_2d = cache["tsne_2d"]
    lengths = get_array_or_none(cache, "lengths")
    helix_frac = get_array_or_none(cache, "helix_frac")
    sheet_frac = get_array_or_none(cache, "sheet_frac")
    loop_frac = get_array_or_none(cache, "loop_frac")
    labels = get_array_or_none(cache, "labels")
    cath_full = get_array_or_none(cache, "cath_full")
    rel_paths = get_array_or_none(cache, "rel_paths")
    latents = get_array_or_none(cache, "latents")

    N = tsne_2d.shape[0]
    print("[Analyze] N points in cache:", N)

    # -------------------------------------------------
    # PCA / UMAP on continuous latents (if available)
    # -------------------------------------------------
    if latents is not None:
        X = np.asarray(latents, dtype=np.float32)
        if X.shape[0] != N:
            print(
                "[Warn] latents.shape[0] != tsne_2d.shape[0]; "
                "using min(N_latents, N_tsne)."
            )
            M = min(X.shape[0], N)
            X = X[:M]
            tsne_2d = tsne_2d[:M]
            if lengths is not None:
                lengths = lengths[:M]
            if helix_frac is not None:
                helix_frac = helix_frac[:M]
            if sheet_frac is not None:
                sheet_frac = sheet_frac[:M]
            if loop_frac is not None:
                loop_frac = loop_frac[:M]
            if labels is not None:
                labels = labels[:M]
            if cath_full is not None:
                cath_full = cath_full[:M]
            if rel_paths is not None:
                rel_paths = rel_paths[:M]
            N = M
            print("[Analyze] trimmed to N =", N)

        print("[Analyze] latents shape:", X.shape)

        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-6
        X_norm = (X - X_mean) / X_std

        # PCA 2D
        if not args.no_pca:
            print("[PCA] running PCA on latents...")
            pca = PCA(n_components=2, random_state=42)
            X_pca2 = pca.fit_transform(X_norm)
            np.save(os.path.join(analysis_dir, "pca_2d.npy"), X_pca2)
            print(
                "[PCA] explained_var_ratio (PC1, PC2):",
                pca.explained_variance_ratio_[:2],
            )

            plot_tsne_plain(
                X_pca2,
                os.path.join(analysis_dir, "pca_plain.png"),
                title="PCA (plain)",
            )
            if lengths is not None:
                plot_tsne_continuous(
                    X_pca2,
                    lengths,
                    os.path.join(analysis_dir, "pca_len_continuous.png"),
                    title="PCA colored by length",
                    cbar_label="sequence length",
                )
            if helix_frac is not None:
                plot_tsne_continuous(
                    X_pca2,
                    helix_frac,
                    os.path.join(analysis_dir, "pca_helix_frac.png"),
                    title="PCA colored by helix_frac",
                    cbar_label="helix fraction",
                )
            if sheet_frac is not None:
                plot_tsne_continuous(
                    X_pca2,
                    sheet_frac,
                    os.path.join(analysis_dir, "pca_sheet_frac.png"),
                    title="PCA colored by sheet_frac",
                    cbar_label="sheet fraction",
                )
            if loop_frac is not None:
                plot_tsne_continuous(
                    X_pca2,
                    loop_frac,
                    os.path.join(analysis_dir, "pca_loop_frac.png"),
                    title="PCA colored by loop_frac",
                    cbar_label="loop fraction",
                )
            if (
                helix_frac is not None
                and sheet_frac is not None
                and loop_frac is not None
            ):
                plot_tsne_ss_argmax(
                    X_pca2,
                    helix_frac,
                    sheet_frac,
                    loop_frac,
                    os.path.join(analysis_dir, "pca_ss_argmax.png"),
                    title="PCA colored by dominant SS",
                )
            if labels is not None:
                plot_tsne_cath_topclass(
                    X_pca2,
                    labels,
                    os.path.join(analysis_dir, "pca_cath_topclass.png"),
                    title="PCA colored by CATH top class",
                )

        # UMAP 2D
        if HAS_UMAP and not args.no_umap:
            print("[UMAP] running UMAP on latents...")
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=30,
                min_dist=0.1,
                random_state=42,
            )
            X_umap2 = reducer.fit_transform(X_norm)
            np.save(os.path.join(analysis_dir, "umap_2d.npy"), X_umap2)

            plot_tsne_plain(
                X_umap2,
                os.path.join(analysis_dir, "umap_plain.png"),
                title="UMAP (plain)",
            )
            if lengths is not None:
                plot_tsne_continuous(
                    X_umap2,
                    lengths,
                    os.path.join(analysis_dir, "umap_len_continuous.png"),
                    title="UMAP colored by length",
                    cbar_label="sequence length",
                )
            if helix_frac is not None:
                plot_tsne_continuous(
                    X_umap2,
                    helix_frac,
                    os.path.join(analysis_dir, "umap_helix_frac.png"),
                    title="UMAP colored by helix_frac",
                    cbar_label="helix fraction",
                )
            if sheet_frac is not None:
                plot_tsne_continuous(
                    X_umap2,
                    sheet_frac,
                    os.path.join(analysis_dir, "umap_sheet_frac.png"),
                    title="UMAP colored by sheet_frac",
                    cbar_label="sheet fraction",
                )
            if loop_frac is not None:
                plot_tsne_continuous(
                    X_umap2,
                    loop_frac,
                    os.path.join(analysis_dir, "umap_loop_frac.png"),
                    title="UMAP colored by loop_frac",
                    cbar_label="loop fraction",
                )
            if (
                helix_frac is not None
                and sheet_frac is not None
                and loop_frac is not None
            ):
                plot_tsne_ss_argmax(
                    X_umap2,
                    helix_frac,
                    sheet_frac,
                    loop_frac,
                    os.path.join(analysis_dir, "umap_ss_argmax.png"),
                    title="UMAP colored by dominant SS",
                )
            if labels is not None:
                plot_tsne_cath_topclass(
                    X_umap2,
                    labels,
                    os.path.join(analysis_dir, "umap_cath_topclass.png"),
                    title="UMAP colored by CATH top class",
                )
        elif not HAS_UMAP and not args.no_umap:
            print(
                "[UMAP] umap-learn is not installed; "
                "install it or use --no_umap to suppress this message."
            )

    # -------------------------------------------------
    # Original t-SNE based analysis
    # -------------------------------------------------

    plot_tsne_plain(
        tsne_2d,
        os.path.join(analysis_dir, "tsne_plain.png"),
        title="t-SNE (plain)",
    )

    if lengths is not None:
        plot_tsne_continuous(
            tsne_2d,
            lengths,
            os.path.join(analysis_dir, "tsne_len_continuous.png"),
            title="t-SNE colored by length",
            cbar_label="sequence length",
        )
    if helix_frac is not None:
        plot_tsne_continuous(
            tsne_2d,
            helix_frac,
            os.path.join(analysis_dir, "tsne_helix_frac.png"),
            title="t-SNE colored by helix_frac",
            cbar_label="helix fraction",
        )
    if sheet_frac is not None:
        plot_tsne_continuous(
            tsne_2d,
            sheet_frac,
            os.path.join(analysis_dir, "tsne_sheet_frac.png"),
            title="t-SNE colored by sheet_frac",
            cbar_label="sheet fraction",
        )
    if loop_frac is not None:
        plot_tsne_continuous(
            tsne_2d,
            loop_frac,
            os.path.join(analysis_dir, "tsne_loop_frac.png"),
            title="t-SNE colored by loop_frac",
            cbar_label="loop fraction",
        )

    if helix_frac is not None and sheet_frac is not None and loop_frac is not None:
        plot_tsne_ss_argmax(
            tsne_2d,
            helix_frac,
            sheet_frac,
            loop_frac,
            os.path.join(analysis_dir, "tsne_ss_argmax.png"),
            title="t-SNE colored by dominant SS",
        )

    if labels is not None:
        plot_tsne_cath_topclass(
            tsne_2d,
            labels,
            os.path.join(analysis_dir, "tsne_cath_topclass.png"),
            title="t-SNE colored by CATH top class",
        )

    if cath_full is not None:
        plot_tsne_fold_topk(
            tsne_2d,
            cath_full,
            os.path.join(analysis_dir, "tsne_fold_topK.png"),
            title="t-SNE colored by fold prefix (top-K)",
            top_k=int(args.top_k_folds),
        )

    cluster_labels = cluster_tsne_kmeans(
        tsne_2d,
        n_clusters=int(args.n_clusters),
        random_state=42,
    )
    plot_tsne_clusters(
        tsne_2d,
        cluster_labels,
        os.path.join(analysis_dir, "tsne_clusters.png"),
        title="t-SNE colored by KMeans clusters",
    )

    lab = np.asarray(cluster_labels, dtype=np.int64)
    cluster_ids = sorted(set(lab.tolist()))
    for cid in cluster_ids:
        idx = np.where(lab == cid)[0]
        if lengths is not None:
            plot_cluster_hist(
                np.asarray(lengths)[idx],
                os.path.join(
                    analysis_dir,
                    "hist_length_cluster_{}.png".format(cid),
                ),
                title="Length histogram (cluster {})".format(cid),
                xlabel="sequence length",
            )
        if helix_frac is not None:
            plot_cluster_hist(
                np.asarray(helix_frac)[idx],
                os.path.join(
                    analysis_dir,
                    "hist_helix_frac_cluster_{}.png".format(cid),
                ),
                title="Helix fraction (cluster {})".format(cid),
                xlabel="helix fraction",
            )
        if sheet_frac is not None:
            plot_cluster_hist(
                np.asarray(sheet_frac)[idx],
                os.path.join(
                    analysis_dir,
                    "hist_sheet_frac_cluster_{}.png".format(cid),
                ),
                title="Sheet fraction (cluster {})".format(cid),
                xlabel="sheet fraction",
            )
        if loop_frac is not None:
            plot_cluster_hist(
                np.asarray(loop_frac)[idx],
                os.path.join(
                    analysis_dir,
                    "hist_loop_frac_cluster_{}.png".format(cid),
                ),
                title="Loop fraction (cluster {})".format(cid),
                xlabel="loop fraction",
            )

    write_cluster_summary(
        analysis_dir=analysis_dir,
        cluster_labels=cluster_labels,
        lengths=lengths,
        helix_frac=helix_frac,
        sheet_frac=sheet_frac,
        loop_frac=loop_frac,
        labels=labels,
        cath_full=cath_full,
        rel_paths=rel_paths,
    )

    print("[Analyze] done.")


if __name__ == "__main__":
    main()
