#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage example:

export CUDA_VISIBLE_DEVICES=0,1,2,3
python inference.py \
  --config configs/vae.yaml \
  --ckpt 815_checkpoints/epochepoch=99.ckpt \
  --output_dir generated_curves \
  --num_curves 5 \
  --min_len 30 \
  --max_len 100 \
  --device cuda
"""

import os
import yaml
import argparse
import torch
import numpy as np
from pathlib import Path
from models import vae_models

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='VAE Curve Inference')
    parser.add_argument('--config', type=str, default='configs/vae.yaml',
                        help='Path to the YAML config file')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to the Lightning checkpoint (.ckpt)')
    parser.add_argument('--output_dir', type=str, default='generated_curves',
                        help='Directory to save generated .npy files')
    parser.add_argument('--num_curves', type=int, default=100,
                        help='Number of curves to generate')
    parser.add_argument('--min_len', type=int, default=10,
                        help='Minimum number of points per curve')
    parser.add_argument('--max_len', type=int, default=None,
                        help='Maximum number of points per curve; if None, uses model.max_seq_len')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (cpu or cuda)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    model_name = config['model_params']['name']
    model_cls = vae_models[model_name]

    # Load normalization stats
    mean_xyz = np.array(config['data_params']['mean_xyz'], dtype=np.float32)
    std_xyz = np.array(config['data_params']['std_xyz'], dtype=np.float32)

    # Instantiate and load model
    model = model_cls(**config['model_params'])
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    # Prepare output
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Determine maximum length
    default_max = config['model_params'].get('max_seq_len', 350)
    max_len = args.max_len or default_max

    latent_dim = config['model_params']['latent_dim']

    with torch.no_grad():
        for i in range(args.num_curves):
            length = np.random.randint(args.min_len, max_len + 1)
            z = torch.randn(1, latent_dim, device=device)
            lengths_tensor = torch.tensor([length], device=device)
            out = model.decode(z, enc_out=None, lengths=lengths_tensor)

            arr = out[0, :length].cpu().numpy()  # (L, 6)
            xyz = arr[:, :3] * std_xyz + mean_xyz  # de-normalize
            ss_logits = arr[:, 3:]
            ss_idx = np.argmax(ss_logits, axis=-1)
            ss_onehot = np.eye(3)[ss_idx]

            full_curve = np.concatenate([xyz, ss_onehot], axis=-1)  # (L, 6)

            filename = os.path.join(args.output_dir, f"curve_{i}.npy")
            np.save(filename, full_curve)

    print(f"Generated {args.num_curves} curves in '{args.output_dir}'")

if __name__ == '__main__':
    main()
