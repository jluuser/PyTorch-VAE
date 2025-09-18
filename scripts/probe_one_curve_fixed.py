#!/usr/bin/env python3
# coding: utf-8
"""
ASCII-only script. Generates reconstructions in memory mode with per-alpha single sample.
Outputs per picked curve:
  - orig_curve.npy/.png
  - recon_memory_a{alpha}.npy/.png  (for each alpha in ALPHAS)

Memory mode means: decode(z, enc_out=enc_out, mask=mask, lengths=lengths)
with z = mu + alpha * std * eps.
"""

import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# ======= USER SETTINGS =======
CKPT_PATH = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/815_checkpoints/epochepoch=14.ckpt"
YAML_PATH = "configs/vae.yaml"
NPY_DIR   = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/prp-dataset/curves_npy-sheet/"
OUT_DIR   = "probe_epoch74_ascii"

# alphas for z = mu + alpha * std * eps
ALPHAS = [0.0, 1.0, 2.0, 3.0]

# epsilon control: "random" | "zero" | "manual"
EPS_MODE = "random"
EPS_MANUAL_SEED = 12345

# scan up to this many dir entries to pick a .npy file
MAX_SCAN = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======= PROJECT IMPORTS =======
from models import vae_models


# ======= UTILS =======
def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_model(cfg, device):
    Model = vae_models[cfg["model_params"]["name"]]
    model = Model(**cfg["model_params"]).to(device)
    model.eval()
    return model

def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    new_state = {}
    for k, v in state.items():
        nk = k[6:] if k.startswith("model.") else k
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print("[Warn] missing keys (truncated):", missing[:6])
    if unexpected:
        print("[Warn] unexpected keys (truncated):", unexpected[:6])

def pick_random_npy_fast(dir_path: str, cap: int = 50) -> Optional[str]:
    candidates, scanned = [], 0
    with os.scandir(dir_path) as it:
        for entry in it:
            scanned += 1
            if entry.is_file() and entry.name.endswith(".npy"):
                candidates.append(entry.path)
            if scanned >= cap:
                break
    if not candidates:
        return None
    choice = np.random.choice(candidates)
    print(f"[Info] scanned {scanned} entries, picked: {os.path.basename(choice)}")
    return choice

def load_and_preprocess_single_curve(npy_path: str, mean_xyz, std_xyz):
    """
    Load one dict npy with keys: 'curve_coords' [L,3], 'ss_one_hot' [L,3].
    Normalize xyz using global mean/std, then concat with ss one-hot -> [L,6].
    Returns x:[1,L,6], mask:[1,L], and mean/std for denorm.
    """
    data = np.load(npy_path, allow_pickle=True).item()
    coords = data["curve_coords"].astype(np.float32)
    ss_one = data["ss_one_hot"].astype(np.float32)

    mean = np.array(mean_xyz, dtype=np.float32).reshape(1, 3)
    std  = np.array(std_xyz,  dtype=np.float32).reshape(1, 3)

    xyz_norm = (coords - mean) / std
    full = np.concatenate([xyz_norm, ss_one], axis=-1)  # [L,6]
    x = torch.from_numpy(full).unsqueeze(0)              # [1,L,6]
    L = x.size(1)
    mask = torch.ones(1, L, dtype=torch.bool)
    return x, mask, mean, std

def denorm_xyz(xyz, mean, std):
    # xyz: (L,3), mean/std: (1,3) numpy
    return xyz * std.reshape(1, 3) + mean.reshape(1, 3)

def plot_xyz_proj(xyz, out_png, title=""):
    fig = plt.figure(figsize=(8, 3), dpi=140)
    plt.subplot(1, 2, 1)
    plt.plot(xyz[:, 0], xyz[:, 1], linewidth=1)
    plt.title("XY"); plt.xlabel("x"); plt.ylabel("y")
    plt.subplot(1, 2, 2)
    plt.plot(xyz[:, 0], xyz[:, 2], linewidth=1)
    plt.title("XZ"); plt.xlabel("x"); plt.ylabel("z")
    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ======= MAIN =======
@torch.no_grad()
def main():
    # enforce ascii-only locale for plot text
    import locale
    try:
        locale.setlocale(locale.LC_ALL, "C")
    except Exception:
        pass

    device = torch.device(DEVICE)
    cfg = load_config(YAML_PATH)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) pick a random .npy
    picked = pick_random_npy_fast(NPY_DIR, cap=MAX_SCAN)
    if picked is None:
        raise FileNotFoundError(
            "No .npy files found within the scan cap. Increase MAX_SCAN or check directory."
        )

    # 2) build model and load checkpoint
    model = build_model(cfg, device)
    load_checkpoint(model, CKPT_PATH, device)
    model.eval()

    # 3) load curve and preprocess
    x, mask, mean_xyz, std_xyz = load_and_preprocess_single_curve(
        picked, cfg["data_params"]["mean_xyz"], cfg["data_params"]["std_xyz"]
    )
    x, mask = x.to(device), mask.to(device)
    lengths = mask.sum(dim=1)

    # 4) encode to get mu/logvar/enc_out
    mu, logvar, enc_out, _ = model.encode(x, mask=mask)  # mu/logvar: (1,D), enc_out: (1,L,H)
    std_latent = torch.exp(0.5 * logvar)                 # (1,D)

    # 5) save and plot original
    orig = x[0].detach().cpu().numpy()
    orig_xyz = denorm_xyz(orig[:, :3], mean_xyz, std_xyz)
    orig_ss  = orig[:, 3:]
    np.save(out_dir / "orig_curve.npy", np.concatenate([orig_xyz, orig_ss], axis=-1))
    plot_xyz_proj(orig_xyz, out_dir / "orig_curve.png", title=f"Original | {os.path.basename(picked)}")

    # 6) choose eps once
    if EPS_MODE == "random":
        eps = torch.randn_like(std_latent)
    elif EPS_MODE == "zero":
        eps = torch.zeros_like(std_latent)
    elif EPS_MODE == "manual":
        gen = torch.Generator(device=device).manual_seed(EPS_MANUAL_SEED)
        eps = torch.randn_like(std_latent, generator=gen)
    else:
        raise ValueError(f"Unknown EPS_MODE: {EPS_MODE}")

    # 7) per-alpha memory-mode reconstructions: decode(z, enc_out=enc_out, ...)
    #    z = mu + alpha * std * eps
    for alpha in ALPHAS:
        z = mu + float(alpha) * std_latent * eps  # (1,D)
        y = model.decode(z, enc_out=enc_out, mask=mask, lengths=lengths)  # (1,L,6)
        y_np = y[0].detach().cpu().numpy()
        y_xyz = denorm_xyz(y_np[:, :3], mean_xyz, std_xyz)

        y_ss_logits = torch.from_numpy(y_np[:, 3:])
        y_ss_idx = y_ss_logits.argmax(dim=-1)
        y_ss_oh = F.one_hot(y_ss_idx, num_classes=3).float().numpy()

        full = np.concatenate([y_xyz, y_ss_oh], axis=-1)
        tag = f"a{str(alpha).replace('.', 'p')}"
        np.save(out_dir / f"recon_memory_{tag}.npy", full)
        plot_xyz_proj(y_xyz, out_dir / f"recon_memory_{tag}.png", title=f"memory | alpha={alpha}")

    print(f"[Done] Picked file: {picked}")
    print(f"[Done] Outputs saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
