#!/usr/bin/env python3
# coding: utf-8

"""
Compute per-dimension mean and std of encoder latent z_e from a manifest.

- Input:  JSONL manifest with "latent_path" (each .npy is [M, D] float32/float64)
- Output: .npz file with:
    - mean: [D] float32
    - std:  [D] float32
    - count: total number of tokens (sum of M)
    - num_sequences: number of sequences used
    - code_dim: D

Example:

python scripts/compute_ze_stats.py \
  --manifest prior/out_prior_token64_K1024_D512_Residual_VQ_data/train/manifest.jsonl \
  --out prior/ze_stats_token64_K1024_D512.npz \
  --max_samples 0


"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="JSONL manifest containing 'latent_path' entries (z_e latents)",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output npz path to save z_e statistics",
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
            lat_path = rec.get("latent_path", "")
            if not lat_path:
                continue

            lat_path = Path(lat_path)
            if not lat_path.is_file():
                print(f"[warn] missing latent file: {lat_path}")
                continue

            arr = np.load(str(lat_path), allow_pickle=False)
            if arr.ndim == 1:
                # Cannot infer code_dim from 1D latent
                continue
            arr = arr.astype(np.float32, copy=False)
            arr = arr.reshape(-1, arr.shape[-1])  # [M, D]
            if arr.size == 0:
                continue

            x = torch.from_numpy(arr)  # [M, D]

            if running_sum is None:
                running_sum = x.sum(dim=0)          # [D]
                running_sq_sum = (x ** 2).sum(dim=0)
                running_count = int(x.size(0))
            else:
                running_sum += x.sum(dim=0)
                running_sq_sum += (x ** 2).sum(dim=0)
                running_count += int(x.size(0))

            num_sequences += 1
            if max_samples > 0 and num_sequences >= max_samples:
                break

            if num_sequences % 1000 == 0:
                print(
                    f"[stats] processed {num_sequences} sequences, "
                    f"tokens so far: {running_count}"
                )

    if running_sum is None or running_sq_sum is None or running_count == 0:
        raise RuntimeError("No valid z_e tokens found in manifest")

    mean = running_sum / float(running_count)
    var = running_sq_sum / float(running_count) - mean ** 2
    var = torch.clamp(var, min=1e-12)
    std = torch.sqrt(var)

    mean_np = mean.detach().cpu().numpy().astype(np.float32, copy=False)
    std_np = std.detach().cpu().numpy().astype(np.float32, copy=False)

    code_dim = int(mean_np.shape[0])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(out_path),
        mean=mean_np,
        std=std_np,
        count=np.int64(running_count),
        num_sequences=np.int64(num_sequences),
        code_dim=np.int64(code_dim),
    )

    print(f"[stats] done.")
    print(f"[stats] sequences processed: {num_sequences}")
    print(f"[stats] tokens counted: {running_count}")
    print(f"[stats] saved stats to: {out_path}")


if __name__ == "__main__":
    main()
