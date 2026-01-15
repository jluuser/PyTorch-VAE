#!/usr/bin/env python3
# coding: utf-8

"""
Compute per-dimension mean and std of VQ-VAE latent z_q from an indices manifest.

- Input:  JSONL manifest with "indices_path" (same format as prior train_manifest)
- VQVAE:  loaded via prior.utils.vq_adapter.load_vq_experiment
- Output: .npz file with mean, std, count, code_dim, num_quantizers

python scripts/compute_zq_stats.py \
  --manifest prior/out_prior_token64_K1024_D512_Residual_VQ_data/train/manifest.jsonl \
  --vq_ckpt checkpoints/vq_token64_K1024_D512_ResidualVQ_fromscratch/epochepoch=139.ckpt \
  --vq_yaml configs/stage2_vq.yaml \
  --out prior/zq_stats_token64_K1024_D512.npz \
  --device cuda \
  --max_samples 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

# repo root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prior.utils.vq_adapter import (
    load_vq_experiment,
    core_model,
    get_vq_info,
    indices_to_latent_sum,
)


def get_device(device_str: str) -> torch.device:
    """Resolve device string to a torch.device."""
    device_str = str(device_str)
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="JSONL manifest used for diffusion prior training (contains indices_path)",
    )
    ap.add_argument(
        "--vq_ckpt",
        type=str,
        required=True,
        help="VQ-VAE checkpoint path",
    )
    ap.add_argument(
        "--vq_yaml",
        type=str,
        required=True,
        help="VQ-VAE config yaml path",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output npz path to save z_q statistics",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string, e.g., 'cuda' or 'cpu'",
    )
    ap.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Optional limit on number of sequences to process (0 = all)",
    )
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    device = get_device(args.device)
    print(f"[stats] device = {device}")

    # Load VQ-VAE experiment and extract core model + codebook info
    vq_exp = load_vq_experiment(args.vq_ckpt, args.vq_yaml, device)
    vq_core = core_model(vq_exp)
    vq_info = get_vq_info(vq_core)

    code_dim = int(vq_info.code_dim)
    num_q = int(vq_info.num_quantizers)
    vq_codebook = vq_info.codebook.to(device)

    # Pad id is only used if sequences contain padding indices.
    # Using K_total (out-of-vocab) is safe when there is no padding.
    pad_id = int(vq_info.K_total)

    print(
        f"[stats] code_dim={code_dim}, num_quantizers={num_q}, "
        f"K_total={int(vq_info.K_total)}, pad_id={pad_id}"
    )

    # Running statistics: sum, squared sum, count
    running_sum: Optional[torch.Tensor] = None
    running_sq_sum: Optional[torch.Tensor] = None
    running_count: int = 0

    num_sequences = 0
    max_samples = int(args.max_samples)

    with manifest_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            idx_path = rec.get("indices_path", "")
            if not idx_path:
                continue

            idx_path = Path(idx_path)
            if not idx_path.is_file():
                print(f"[warn] missing indices file: {idx_path}")
                continue

            arr = np.load(str(idx_path), allow_pickle=False).reshape(-1)
            if arr.size == 0:
                continue

            indices = torch.from_numpy(arr.astype(np.int64, copy=False)).unsqueeze(0).to(device)  # [1, L_flat]

            # Convert indices -> latent z_q and token mask
            z_q, token_mask = indices_to_latent_sum(
                vq_codebook,
                indices,
                num_quantizers=num_q,
                pad_id=pad_id,
                return_token_mask=True,
            )  # z_q: [1, M, D], token_mask: [1, M] or None

            # Remove batch dimension
            z_q = z_q[0]  # [M, D]
            if token_mask is not None:
                mask = token_mask[0].to(torch.bool)  # [M]
                z_q = z_q[mask]

            if z_q.numel() == 0:
                continue

            # Initialize running stats on first valid batch
            if running_sum is None:
                running_sum = z_q.sum(dim=0)  # [D]
                running_sq_sum = (z_q ** 2).sum(dim=0)  # [D]
                running_count = int(z_q.size(0))
            else:
                running_sum += z_q.sum(dim=0)
                running_sq_sum += (z_q ** 2).sum(dim=0)
                running_count += int(z_q.size(0))

            num_sequences += 1
            if max_samples > 0 and num_sequences >= max_samples:
                break

            if num_sequences % 1000 == 0:
                print(
                    f"[stats] processed {num_sequences} sequences, "
                    f"tokens so far: {running_count}"
                )

    if running_sum is None or running_sq_sum is None or running_count == 0:
        raise RuntimeError("No valid z_q tokens found in manifest")

    mean = running_sum / float(running_count)
    var = running_sq_sum / float(running_count) - mean ** 2
    var = torch.clamp(var, min=1e-12)
    std = torch.sqrt(var)

    mean_np = mean.detach().cpu().numpy().astype(np.float32, copy=False)
    std_np = std.detach().cpu().numpy().astype(np.float32, copy=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(out_path),
        mean=mean_np,
        std=std_np,
        count=np.int64(running_count),
        num_sequences=np.int64(num_sequences),
        code_dim=np.int64(code_dim),
        num_quantizers=np.int64(num_q),
    )

    print(f"[stats] done.")
    print(f"[stats] sequences processed: {num_sequences}")
    print(f"[stats] tokens counted: {running_count}")
    print(f"[stats] saved stats to: {out_path}")


if __name__ == "__main__":
    main()
