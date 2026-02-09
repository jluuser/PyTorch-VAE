#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
python scripts/playground/analyze_latent_regions.py \
  --curve_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/results/aeot_runs/test_run_random_02/filtered_npy \
  --config configs/stage1_ae.yaml \
  --ckpt /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/checkpoints/aeot_sigmoid/epochepoch=epoch=089.ckpt \
  --base_cache /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/latent_analysis_ae_sigmoid/class1/tsne_cache_ae_tokens_mean_class1_len_between_1_80.npz \
  --umap_model /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/latent_analysis_ae_sigmoid/class1/umap_reducer_ae_tokens_mean_class1_len_between_1_80.pkl \
  --out_dir results/analysis_regions_02 \
  --n_clusters 12 \
  --latent_rep tokens_mean
'''
import os
import sys
import json
import argparse
import joblib
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Repo Path Setup
# -----------------------------
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import build_experiment_from_yaml

# -----------------------------
# Helper Functions
# -----------------------------
def strip_prefixes(state_dict):
    out = {}
    for k, v in state_dict.items():
        name = k
        for p in ("model.", "module.", "net."):
            if name.startswith(p):
                name = name[len(p):]
                break
        out[name] = v
    return out

@torch.no_grad()
def encode_single_curve(model, x_np, device, latent_rep):
    """
    Directly encode a single [L, 6] numpy array to a latent vector.
    """
    model.eval()
    # x_np shape: [L, 6] -> tensor [1, L, 6]
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)
    mask = torch.ones((1, x.size(1)), dtype=torch.bool).to(device)
    
    h_fuse, _, _ = model.encode(x, mask=mask)
    z_tok = model._tokenize_to_codes(h_fuse, mask) # [1, N, D]
    
    if latent_rep == "tokens_mean":
        z = z_tok.mean(dim=1) # [1, D]
    else:
        z = z_tok.reshape(z_tok.size(0), -1) # [1, N*D]
        
    return z.cpu().numpy().astype(np.float32)

# -----------------------------
# Main Logic
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Integrated Latent Region Analyzer")
    # Paths
    parser.add_argument("--curve_dir", type=str, required=True, help="Generated .npy curves directory")
    parser.add_argument("--config", type=str, required=True, help="stage1_ae.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="AE checkpoint")
    parser.add_argument("--base_cache", type=str, required=True, help="Base .npz file (AFDB)")
    parser.add_argument("--umap_model", type=str, required=True, help="UMAP .pkl model")
    
    # Analysis Params
    parser.add_argument("--n_clusters", type=int, default=12, help="Number of regions to partition")
    parser.add_argument("--latent_rep", type=str, default="tokens_mean")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Load Base Data & Partition (K-Means on UMAP Space)
    print(f"[1/5] Loading base cache and partitioning into {args.n_clusters} regions...")
    base_data = np.load(args.base_cache, allow_pickle=True)
    
    X_base = base_data["umap_2d"] if "umap_2d" in base_data.files else base_data["base_umap_2d"]
    
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    base_regions = kmeans.fit_predict(X_base)
    centroids = kmeans.cluster_centers_

    # 2. Load AE and Encode Probe Curves
    print("[2/5] Loading AE model and encoding generated curves...")
    exp, _ = build_experiment_from_yaml(args.config)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = strip_prefixes(ckpt.get("state_dict", ckpt))
    exp.model.load_state_dict(state_dict, strict=False)
    model = exp.model.to(device).eval()

    # Collect .npy files directly
    curve_files = sorted(list(Path(args.curve_dir).rglob("*.npy")))
    if not curve_files:
        print(f"Error: No .npy files found in {args.curve_dir}")
        return

    probe_latents = []
    print(f"Processing {len(curve_files)} files...")
    for p in tqdm(curve_files):
        arr = np.load(p).astype(np.float32)
        
        if arr.dtype == object:
            arr = arr.item()['curve_coords']
            
        z = encode_single_curve(model, arr, device, args.latent_rep)
        probe_latents.append(z)
        
    probe_latents = np.concatenate(probe_latents, axis=0)

    # 3. Project to UMAP and Assign Regions
    print("[3/5] Projecting to UMAP and mapping to regions...")
    reducer = joblib.load(args.umap_model)
    X_probe = reducer.transform(probe_latents)
    probe_regions = kmeans.predict(X_probe)

    # 4. Generate JSON Manifest
    print("[4/5] Exporting JSON manifest...")
    region_manifest = {f"Region_{i}": [] for i in range(args.n_clusters)}
    for idx, r_id in enumerate(probe_regions):
        region_manifest[f"Region_{r_id}"].append(curve_files[idx].name)

    with open(out_dir / "region_manifest.json", "w") as f:
        json.dump(region_manifest, f, indent=4)

    # 5. Render Partitioned Overlay Plot
    print("[5/5] Rendering overlay plot...")
    fig = plt.figure(figsize=(10, 8), dpi=180)
    ax = fig.add_subplot(111)

    # Plot base points colored by cluster
    ax.scatter(X_base[:, 0], X_base[:, 1], c=base_regions, cmap='tab20', s=1, alpha=0.15)
    
    # Plot probe points (Generated)
    ax.scatter(X_probe[:, 0], X_probe[:, 1], c='red', s=10, alpha=0.8, edgecolors='white', linewidths=0.3, label="Generated")
    
    # Add labels to centroids
    for i, (cx, cy) in enumerate(centroids):
        ax.text(cx, cy, f"R{i}", fontsize=10, weight='bold', color='black', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.set_title(f"Latent Space Partitioning | Generated: {len(X_probe)} curves")
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(out_dir / "region_overlay_analysis.png")
    plt.close(fig)
    
    print(f"\n[Done] Outputs saved to {out_dir}")
    print(f"- Plot: region_overlay_analysis.png")
    print(f"- JSON: region_manifest.json")

if __name__ == "__main__":
    main()