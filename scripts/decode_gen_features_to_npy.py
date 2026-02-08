#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Decode generated latent features (gen_features.pt) into curves and save
ONE curve per .npy file.

Inputs:
  - YAML config for VQVAE (e.g., configs/stage2_vq.yaml)
  - Trained VQVAE checkpoint (.ckpt)
  - gen_features.pt: Tensor [N, 64, 512], generated ze tokens

Outputs:
  - For each sample i in [0, N):
      out_dir/curve_{i:05d}.npy  with shape [target_len, 6]

Example:

python scripts/decode_gen_features_to_npy.py \
  --config configs/stage2_vq.yaml \
  --ckpt checkpoints/vq_token64_K1024_D512_ResidualVQ_fromscratch/epochepoch=139.ckpt \
  --features_pt /public/home/zhangyangroup/chengshiz/keyuan.zhou/gen_features.pt \
  --out_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/prp-dataset/gen_from_ze_npy_len40 \
  --target_len 40 \
  --batch_size 64
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

# Add repo root to sys.path so that "from models.vq_vae import VQVAE" works
THIS_DIR = Path(__file__).resolve().parent          # .../PyTorch-VAE/scripts
REPO_ROOT = THIS_DIR.parent                         # .../PyTorch-VAE
sys.path.append(str(REPO_ROOT))

from models.vq_vae import VQVAE  # your VQVAE class


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model_from_config(config_path: str, ckpt_path: str, device: torch.device) -> VQVAE:
    """
    Build a VQVAE model from YAML config and load weights from a checkpoint.
    Only "model.*" keys are loaded, following run.py behavior.
    """
    cfg = load_yaml(config_path)
    model_params = cfg["model_params"].copy()

    # VQVAE __init__ does not need "name"
    model_params.pop("name", None)

    model = VQVAE(**model_params).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    # Strip the "model." prefix used by Lightning
    stripped = {}
    for k, v in state.items():
        if k.startswith("model."):
            stripped[k[len("model."):]] = v

    missing, unexpected = model.load_state_dict(stripped, strict=False)
    print(f"[Load] from {ckpt_path}")
    print(f"[Load] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("  missing keys (first 5):", missing[:5])
    if unexpected:
        print("  unexpected keys (first 5):", unexpected[:5])

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser("Decode gen_features.pt to per-curve .npy files")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (e.g., configs/stage2_vq.yaml)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained VQVAE checkpoint (.ckpt)")
    parser.add_argument("--features_pt", type=str, required=True,
                        help="Path to gen_features.pt (Tensor [N,64,512])")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save per-curve .npy files")
    parser.add_argument("--target_len", type=int, default=80,
                        help="Output curve length L (mask length for decoder)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Decode batch size")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu' (default: cuda if available)")
    args = parser.parse_args()

    # Device selection
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("[Device]", device)

    # Build model and load weights
    model = build_model_from_config(args.config, args.ckpt, device)

    # Load generated features
    feat_path = Path(args.features_pt)
    if not feat_path.is_file():
        raise FileNotFoundError(f"features_pt not found: {feat_path}")

    obj = torch.load(str(feat_path), map_location="cpu")
    if not torch.is_tensor(obj):
        raise RuntimeError(f"Expected Tensor in {feat_path}, got {type(obj)}")

    # Cast to float32 and move to target device
    ze_all = obj.to(dtype=torch.float32, device=device)  # [N, 64, 512]
    N, Ntok, D = ze_all.shape
    print(f"[Features] shape={ze_all.shape}, dtype={ze_all.dtype}")

    # Prepare output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Output dir] {out_dir}")

    B = int(args.batch_size)
    L_out = int(args.target_len)

    model.eval()
    with torch.no_grad():
        idx_global = 0
        for start in range(0, N, B):
            end = min(N, start + B)
            ze_batch = ze_all[start:end]  # [b, 64, 512]
            bsz = ze_batch.size(0)

            # Decoder mask: all-True, only used to set output length
            mask = torch.ones(bsz, L_out, dtype=torch.bool, device=device)

            # Directly use ze_batch as z_for_decode
            out = model.decode(ze_batch, mask=mask)  # [b, L_out, 6]
            out_np = out.cpu().numpy()               # [b, L_out, 6]

            # Save one curve per .npy file
            for i in range(bsz):
                curve = out_np[i]  # [L_out, 6]
                fname = out_dir / f"curve_{idx_global:05d}.npy"
                np.save(fname, curve)
                idx_global += 1

            print(f"[Decode] processed {end}/{N}")

    print(f"[Done] total curves saved: {N} to {out_dir}")


if __name__ == "__main__":
    main()
