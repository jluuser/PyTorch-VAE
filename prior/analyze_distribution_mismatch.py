#!/usr/bin/env python3
# coding: utf-8
"""
Diagnostic Script: Prior Distribution Mismatch Analyzer

What it does:
1. Loads Ground Truth indices (Train Manifest) and Prior Sampled indices (Sample Manifest).
2. Computes Code Usage Histograms (Are we using the whole vocab?).
3. Reconstructs Latent Vectors (z_q) using the VQ-VAE Codebook.
4. Computes Latent Norm Distributions (Is the signal energy correct?).

Usage:
  python prior/analyze_distribution_mismatch.py \
  --vq_ckpt /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/checkpoints/vq_token64_K1024_D512_ResidualVQ_fromscratch/epochepoch=139.ckpt \
  --vq_yaml /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/configs/stage2_vq.yaml \
  --train_manifest /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/prior/out_prior_token64_K1024_D512_Residual_VQ_data/train/manifest.jsonl \
  --sample_manifest /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/prior/prior_samples_1_8/samples_manifest.jsonl \
  --out_dir prior/analysis_result_1_11s \
  --device cuda
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import sys

# Add path to import your experiment utils
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from experiment import build_experiment_from_yaml
except ImportError:
    sys.path.append(os.getcwd())
    try:
        from experiment import build_experiment_from_yaml
    except ImportError:
        print("[Error] Could not import 'experiment'. Make sure you are in the project root.")
        sys.exit(1)

# ================= Utils to Reuse VQVAE Logic =================
# We need these to correctly map Indices -> z_q (Summing residuals)

@torch.no_grad()
def get_codebook_and_quantizers(model):
    # Adapt this if your model structure is different
    core = model.model if hasattr(model, "model") else model
    q = getattr(core, "quantizer", None)
    if q is None: raise RuntimeError("Quantizer not found")
    
    emb = getattr(q, "embedding", None) # Expect [K, D]
    num_q = int(getattr(q, "num_quantizers", 1))
    
    return emb, num_q

@torch.no_grad()
def indices_to_latent(indices, emb, num_q):
    # indices: [B, N_flat] -> z_q: [B, N_tokens, D]
    B, N_flat = indices.shape
    D = emb.shape[1]
    
    # RVQ Flatten Logic: [B, N_tokens * num_q]
    N_tokens = N_flat // num_q
    
    # Lookup embeddings
    # inds: [B, N_flat] -> z_all: [B, N_flat, D]
    z_all = F.embedding(indices, emb)
    
    # Reshape to [B, N_tokens, num_q, D]
    z_all = z_all.view(B, N_tokens, num_q, D)
    
    # Sum residuals: [B, N_tokens, D]
    z_q = z_all.sum(dim=2)
    return z_q

# ================= Analysis Functions =================

def load_data(manifest_path, limit=500):
    indices_list = []
    print(f"[Info] Loading {limit} records from {manifest_path}...")
    with open(manifest_path, 'r') as f:
        lines = [l for l in f if l.strip()]
        
    # Shuffle or just take first N? Let's take first N for speed
    if len(lines) > limit:
        lines = lines[:limit]
        
    for line in tqdm(lines):
        rec = json.loads(line)
        path = rec["indices_path"]
        if Path(path).exists():
            # Load and flatten
            idx = np.load(path, allow_pickle=False).reshape(-1)
            indices_list.append(idx)
            
    if not indices_list:
        return None
    # Return as [Total_Samples, Seq_Len]
    return torch.from_numpy(np.stack(indices_list)).long()

def analyze_code_frequency(train_idx, sample_idx, out_dir, vocab_size=4096):
    """
    Compare how 'flat' or 'peaked' the code usage is.
    """
    print("[Analysis] Computing Code Frequencies...")
    
    # Flatten everything
    t_flat = train_idx.view(-1).numpy()
    s_flat = sample_idx.view(-1).numpy()
    
    # Count
    t_counts = Counter(t_flat)
    s_counts = Counter(s_flat)
    
    # Convert to probabilities
    t_probs = np.array([t_counts.get(i, 0) for i in range(vocab_size)])
    s_probs = np.array([s_counts.get(i, 0) for i in range(vocab_size)])
    
    t_probs = t_probs / (t_probs.sum() + 1e-8)
    s_probs = s_probs / (s_probs.sum() + 1e-8)
    
    # Sort by Train Frequency to visualize misalignment
    sorted_indices = np.argsort(-t_probs) # Descending order of GT usage
    t_sorted = t_probs[sorted_indices]
    s_sorted = s_probs[sorted_indices]
    
    # Plot Top 200 codes (The "Head" of the distribution)
    plt.figure(figsize=(12, 6))
    x = np.arange(200)
    plt.bar(x, t_sorted[:200], alpha=0.6, label='Train (Ground Truth)', color='blue')
    plt.bar(x, s_sorted[:200], alpha=0.6, label='Prior Samples', color='orange')
    plt.title("Top 200 Most Frequent Codes (Sorted by Train Freq)")
    plt.xlabel("Rank")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig(out_dir / "code_frequency_head.png")
    plt.close()
    
    # Plot Entropy / Coverage
    used_t = np.sum(t_probs > 0)
    used_s = np.sum(s_probs > 0)
    print(f"  > Unique Codes Used - Train: {used_t}/{vocab_size}, Sample: {used_s}/{vocab_size}")
    
    # Coverage Plot (Cumulative)
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(t_sorted), label='Train CDF', color='blue')
    plt.plot(np.cumsum(np.sort(s_probs)[::-1]), label='Sample CDF', color='orange')
    plt.title("Code Coverage (Cumulative Distribution)")
    plt.xlabel("Num Codes")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "code_coverage.png")
    plt.close()

def analyze_latent_norms(train_idx, sample_idx, emb, num_q, out_dir, device):
    """
    Reconstruct z_q and calculate L2 norms.
    """
    print("[Analysis] Computing Latent Norms...")
    emb = emb.to(device)
    
    # Process in batches to avoid OOM
    def get_norms(indices):
        norms = []
        batch_size = 100
        for i in range(0, len(indices), batch_size):
            batch = indices[i:i+batch_size].to(device)
            z_q = indices_to_latent(batch, emb, num_q) # [B, N_tokens, D]
            # Calculate norm per token: [B, N_tokens]
            token_norms = torch.norm(z_q, p=2, dim=-1).view(-1).cpu().numpy()
            norms.append(token_norms)
        return np.concatenate(norms)

    t_norms = get_norms(train_idx)
    s_norms = get_norms(sample_idx)
    
    print(f"  > Mean Norm - Train: {t_norms.mean():.4f} +/- {t_norms.std():.4f}")
    print(f"  > Mean Norm - Sample: {s_norms.mean():.4f} +/- {s_norms.std():.4f}")
    
    # Plot Histograms
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, max(t_norms.max(), s_norms.max()), 100)
    plt.hist(t_norms, bins=bins, alpha=0.5, label='Train Latents', density=True, color='blue')
    plt.hist(s_norms, bins=bins, alpha=0.5, label='Sample Latents', density=True, color='orange')
    plt.title("Latent Vector Norm Distribution (||z_q||)")
    plt.xlabel("L2 Norm")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(out_dir / "latent_norms.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vq_ckpt", type=str, required=True)
    parser.add_argument("--vq_yaml", type=str, required=True)
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--sample_manifest", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="debug_distribution")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    train_indices = load_data(args.train_manifest, limit=1000)
    sample_indices = load_data(args.sample_manifest, limit=1000)
    
    if train_indices is None or sample_indices is None:
        print("Error loading data.")
        return

    # 2. Analyze Code Frequency (Does not need VQ model)
    # Assume Vocab Size is max index found + padding room, or hardcode 4096
    vocab_size = 4096
    analyze_code_frequency(train_indices, sample_indices, out_dir, vocab_size)
    
    # 3. Load VQ Model (For Embedding lookup)
    print(f"[Info] Loading VQ Model from {args.vq_ckpt}...")
    device = torch.device(args.device)
    exp, _ = build_experiment_from_yaml(args.vq_yaml)
    sd = torch.load(args.vq_ckpt, map_location="cpu")
    exp.load_state_dict(sd["state_dict"], strict=False)
    exp.to(device).eval()
    
    emb, num_q = get_codebook_and_quantizers(exp)
    
    # 4. Analyze Latent Norms
    analyze_latent_norms(train_indices, sample_indices, emb, num_q, out_dir, device)
    
    print(f"[Done] Results saved to {out_dir}")

if __name__ == "__main__":
    main()