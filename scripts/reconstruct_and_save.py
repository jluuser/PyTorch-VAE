#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import shutil
import yaml
from models import vae_models

# ==== Paths ====
npy_dir = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/prp-dataset/filtered_curves_npy/"
ckpt_path = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/812_checkpoints/epochepoch=04.ckpt"
yaml_path = "configs/vae.yaml"
output_root = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/test/"
original_dir = os.path.join(output_root, "original")
reconstructed_dir = os.path.join(output_root, "reconstructed7")

# ==== Ensure output directories ====
os.makedirs(original_dir, exist_ok=True)
os.makedirs(reconstructed_dir, exist_ok=True)

# ==== Load mean/std from yaml ====
with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

mean = np.array(config["data_params"]["mean_xyz"], dtype=np.float32).reshape(1, 3)
std = np.array(config["data_params"]["std_xyz"], dtype=np.float32).reshape(1, 3)

# ==== Collect first 3 .npy files ====
all_npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith(".npy")])
selected_npy = all_npy_files[:3]

# ==== Load model ====
model_name = config["model_params"]["name"]
model_cls = vae_models[model_name]
model = model_cls(**config["model_params"])

ckpt = torch.load(ckpt_path, map_location="cpu")
state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
model.load_state_dict(state_dict)

# ==== Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# ==== Process each file ====
for fname in selected_npy:
    input_path = os.path.join(npy_dir, fname)
    print(f"Processing: {fname}")

    # Load and normalize data
    raw_data = np.load(input_path, allow_pickle=True).item()
    coords = raw_data["curve_coords"].astype(np.float32)          # [L, 3]
    coords_norm = (coords - mean) / std                           # normalized [L, 3]
    ss_onehot = raw_data["ss_one_hot"].astype(np.float32)         # [L, 3]
    input_array = np.concatenate([coords_norm, ss_onehot], axis=-1)  # [L, 6]

    input_tensor = torch.from_numpy(input_array).unsqueeze(0).to(device)  # [1, L, 6]
    mask = torch.any(input_tensor[:, :, :3] != 0, dim=-1).to(device)      # [1, L]

    # Deterministic reconstruction: z = mu
    with torch.no_grad():
        mu, logvar, enc_out, _ = model.encode(input_tensor, mask=mask)
        lengths = mask.sum(dim=1)  # [1]
        recon = model.decode(mu, enc_out=enc_out, mask=mask, lengths=lengths)[0]  # [L, 6]

    # Denormalize xyz
    xyz = (recon[:, :3].detach().cpu().numpy() * std) + mean
    ss_logits = recon[:, 3:]
    ss_idx = torch.argmax(ss_logits, dim=-1)
    ss_onehot_out = torch.nn.functional.one_hot(ss_idx, num_classes=3).float().cpu().numpy()
    final = np.concatenate([xyz, ss_onehot_out], axis=-1)

    # Save reconstructed
    np.save(os.path.join(reconstructed_dir, fname), final)

    # Save original (denormalized) for comparison
    orig_curve = np.concatenate([coords, ss_onehot], axis=-1)
    np.save(os.path.join(original_dir, fname), orig_curve)

    print(f"Saved reconstructed: {os.path.join(reconstructed_dir, fname)}")
    print(f"Saved original: {os.path.join(original_dir, fname)}")
