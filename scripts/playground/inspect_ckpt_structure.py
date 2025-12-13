#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_full_weights_to_txt.py

Dump all encoder and decoder weights (full tensors) from a Lightning checkpoint
to a UTF-8 text file.

This can be large (~hundreds of MB), so ensure you have enough disk space.
"""

import os
import torch
import numpy as np
# ======== User settings ========
CKPT_PATH = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/new_checkpoints2/epochepoch=079.ckpt"
OUT_TXT_PATH = "encoder_decoder_full_weights2.txt"
# ================================

assert os.path.isfile(CKPT_PATH), f"Checkpoint not found: {CKPT_PATH}"
ckpt = torch.load(CKPT_PATH, map_location="cpu")

# Handle Lightning checkpoints
state_dict = ckpt.get("state_dict", ckpt)

encoder_items = {k: v for k, v in state_dict.items() if k.startswith("model.encoder")}
decoder_items = {k: v for k, v in state_dict.items() if k.startswith("model.decoder")}

print(f"Loaded checkpoint: {CKPT_PATH}")
print(f"Encoder params: {len(encoder_items)} | Decoder params: {len(decoder_items)}")

with open(OUT_TXT_PATH, "w", encoding="utf-8") as f:
    f.write("=" * 100 + "\n")
    f.write(f"Checkpoint: {CKPT_PATH}\n")
    f.write(f"Total encoder params: {len(encoder_items)} | decoder params: {len(decoder_items)}\n")
    f.write("=" * 100 + "\n\n")

    # ---- Encoder ----
    f.write("[ENCODER WEIGHTS]\n")
    for k, v in encoder_items.items():
        arr = v.detach().cpu().numpy()
        f.write(f"{k}\n")
        f.write(f"  shape: {arr.shape}\n")
        np.set_printoptions(threshold=np.inf, linewidth=180, precision=6, suppress=True)
        f.write(f"{arr}\n\n")

    # ---- Decoder ----
    f.write("[DECODER WEIGHTS]\n")
    for k, v in decoder_items.items():
        arr = v.detach().cpu().numpy()
        f.write(f"{k}\n")
        f.write(f"  shape: {arr.shape}\n")
        np.set_printoptions(threshold=np.inf, linewidth=180, precision=6, suppress=True)
        f.write(f"{arr}\n\n")

print(f"✅ All encoder/decoder weights dumped to {OUT_TXT_PATH}")
