#!/usr/bin/env python3
"""
Build a VQ codebook via K-means on encoder latents (z_e) from a Stage-A pretrained model.

Usage:
  python scripts/build_codebook_kmeans.py \
    --config configs/vq_vae.yaml \
    --ckpt 902_checkpoints/last.ckpt \
    --out scripts/kmeans_centroids_256x64.npy \
    --n-samples 200000 \
    --batch-size 64

Notes:
  - Ensure Stage-A training was done with `use_vq: false`.
  - This script forces `use_vq=false` when constructing the model to extract continuous z_e.
  - Centroids are L2-normalized before saving, matching the quantizer's cosine metric.
"""

import os
import sys
import yaml
import math
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Project-local imports
# Assumes repository layout similar to: models/, dataset.py, etc.
from models import vae_models  # mapping name -> class
from dataset import CurveDataModule


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def make_model(cfg: dict, device: torch.device) -> nn.Module:
    mp = dict(cfg["model_params"])
    # Force continuous AE mode for feature extraction
    mp["use_vq"] = False
    model = vae_models[mp["name"]](**mp)
    model.to(device)
    model.eval()
    return model


def load_stage_a_weights(model: nn.Module, ckpt_path: str) -> None:
    """
    Loads Lightning checkpoint saved from VAEXperiment into bare model.
    Tries to strip the 'model.' prefix if present.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    # First try: direct load (in case keys match)
    try:
        model.load_state_dict(state, strict=False)
        return
    except Exception:
        pass

    # Second try: strip "model." prefix
    stripped = {}
    for k, v in state.items():
        if k.startswith("model."):
            stripped[k[len("model."):]] = v
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if len(missing) > 0:
        print(f"[Warn] Missing keys after load: {len(missing)} (showing first 10): {missing[:10]}")
    if len(unexpected) > 0:
        print(f"[Warn] Unexpected keys after load: {len(unexpected)} (showing first 10): {unexpected[:10]}")


@torch.no_grad()
def collect_latents(model: nn.Module,
                    datamodule: CurveDataModule,
                    device: torch.device,
                    target_tokens: int,
                    batch_size: int,
                    num_workers: int) -> np.ndarray:
    """
    Streams batches from the train dataloader, encodes to z_e, collects valid tokens by mask,
    and returns up to `target_tokens` latent vectors as a NumPy array of shape (N, D).
    """
    # Use the datamodule's train_dataloader to inherit its collate_fn and settings
    dl = DataLoader(
        datamodule.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=datamodule.train_dataloader().collate_fn,  # reuse its pad_collate
    )

    collected = []
    total = 0
    t0 = time.time()

    for step, (x, mask) in enumerate(dl, start=1):
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # Encode to z_e (B, L, D)
        z_e = model.encode(x, mask=mask)

        # Keep only valid positions
        # mask: (B, L) bool -> index into z_e
        valid = mask.reshape(-1)
        if valid.any():
            z = z_e.reshape(-1, z_e.size(-1))[valid]  # (M, D)
            collected.append(z.detach().cpu())
            total += z.size(0)

        if step % 50 == 0:
            dt = time.time() - t0
            print(f"[Info] Batches: {step:04d}, collected tokens: {total}, elapsed: {dt:.1f}s")

        if total >= target_tokens:
            break

    if len(collected) == 0:
        raise RuntimeError("No valid tokens collected. Check masks and dataset.")

    X = torch.cat(collected, dim=0)
    if X.size(0) > target_tokens:
        X = X[:target_tokens]
    print(f"[Info] Collected latents: {tuple(X.shape)}")
    return X.numpy()


def run_kmeans(X: np.ndarray, n_clusters: int, batch_size: int = 4096, max_iter: int = 100) -> np.ndarray:
    """
    Runs MiniBatchKMeans on X (N, D) and returns L2-normalized centroids (K, D).
    """
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as e:
        raise RuntimeError("scikit-learn is required: pip install scikit-learn") from e

    print(f"[Info] Running MiniBatchKMeans: K={n_clusters}, N={X.shape[0]}, D={X.shape[1]}")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=max_iter,
        verbose=1,
        n_init=3,
        init="k-means++",
        reassignment_ratio=0.01,
    )
    kmeans.fit(X)
    C = kmeans.cluster_centers_.astype(np.float32)
    # L2 normalize to match cosine/dot product usage
    denom = np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    C /= denom
    return C


def main():
    parser = argparse.ArgumentParser(description="Build VQ codebook via K-means over encoder latents.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config used for Stage-A model.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Stage-A checkpoint (Lightning .ckpt).")
    parser.add_argument("--out", type=str, required=True, help="Path to save centroids .npy.")
    parser.add_argument("--n-samples", type=int, default=200000, help="Number of latent tokens to collect.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for latent extraction.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for encoding.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Resolve K from config
    mp = cfg.get("model_params", {})
    codebook_size = int(mp.get("codebook_size", 256))
    model_name = mp.get("name", "VQVAE")
    print(f"[Info] Model={model_name}, codebook_size={codebook_size}")

    # DataModule
    dm = CurveDataModule(**cfg["data_params"])
    dm.setup()

    # Device
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # Model
    model = make_model(cfg, device)
    load_stage_a_weights(model, args.ckpt)
    model.eval()

    # Collect latents
    X = collect_latents(
        model=model,
        datamodule=dm,
        device=device,
        target_tokens=int(args.n_samples),
        batch_size=int(args.batch_size),
        num_workers=int(args.workers),
    )

    # K-means
    C = run_kmeans(X, n_clusters=codebook_size, batch_size=4096, max_iter=100)

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, C)
    print(f"[Done] Saved centroids to: {out_path} shape={C.shape}")


if __name__ == "__main__":
    main()
