#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from sklearn.cluster import KMeans
'''
python scripts/extract_umap_island_paths.py \
  --cache latent_analysis/class1/tsne_cache_class1_len_between_1_80.npz \
  --n_clusters 2

'''



def parse_args():
    p = argparse.ArgumentParser("Extract left island paths from UMAP (1D x-split)")
    p.add_argument("--cache", type=str, required=True,
                   help="tsne_cache_*.npz path")
    p.add_argument("--analysis_dir", type=str, default="",
                   help="analysis dir; if empty, infer from cache path")
    p.add_argument("--n_clusters", type=int, default=2,
                   help="KMeans clusters on UMAP x coordinate")
    return p.parse_args()


def main():
    args = parse_args()
    cache_path = os.path.abspath(args.cache)
    cache = np.load(cache_path, allow_pickle=True)

    if "rel_paths" not in cache.files:
        raise RuntimeError("rel_paths not found in cache")

    rel_paths = cache["rel_paths"]

    if args.analysis_dir:
        analysis_dir = args.analysis_dir
    else:
        cache_dir = os.path.dirname(cache_path)
        analysis_dir = os.path.join(cache_dir, "analysis")

    umap_npy = os.path.join(analysis_dir, "umap_2d.npy")
    if not os.path.isfile(umap_npy):
        raise FileNotFoundError("umap_2d.npy not found: {}".format(umap_npy))

    umap_2d = np.load(umap_npy)
    x = umap_2d[:, 0:1]  # shape [N,1]
    N = x.shape[0]
    print("Loaded UMAP x coords:", x.shape)

    n_clusters = min(max(args.n_clusters, 2), N)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(x)
    centers = km.cluster_centers_.reshape(-1)

    left_cluster = int(np.argmin(centers))
    print("Left-most cluster id:", left_cluster, "center_x=", centers[left_cluster])

    idx = np.where(labels == left_cluster)[0]
    print("Island size:", idx.shape[0])

    out_txt = os.path.join(analysis_dir, "umap_left_island_curves.txt")
    with open(out_txt, "w") as f:
        f.write("# Left island curves from UMAP (1D x-split)\n")
        f.write("# global_index\trel_path\n")
        for i in idx:
            name = str(rel_paths[i]) if i < rel_paths.shape[0] else "idx_{}".format(i)
            f.write("{}\t{}\n".format(i, name))

    print("Saved island list to:", out_txt)


if __name__ == "__main__":
    main()
