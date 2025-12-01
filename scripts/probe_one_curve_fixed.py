#!/usr/bin/env python3
# coding: utf-8
"""
Minimal reconstruction probe for your VQ-VAE (pure VQ).
Encode -> Tokenize (L->N) -> z_e -> Quantize (no EMA) -> Decode.
ASCII-only, short comments.
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import yaml
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== User paths =====
CKPT_PATH = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/vq_s_gradient_ckpt_test11_15/epochepoch=499.ckpt"
YAML_PATH = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/configs/stage2_vq.yaml"
NPY_DIR   = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/prp-dataset/filtered_curves_npy/"
OUT_DIR   = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/probe_epoch84_newarch/"

# ===== Probe knobs =====
EPOCH_HINT: Optional[int] = None
STEPS_PER_EPOCH = 420
ALPHAS = [0.0, 0.5, 1.0, 2.0]
EPS_MODE = "random"   # "random" | "zero" | "manual"
EPS_MANUAL_SEED = 12345
MAX_SCAN = 50
SEED = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Repo root on sys.path =====
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# now we can safely import experiment builder
from experiment import build_experiment_from_yaml


# ===== Utils =====
def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_epoch_from_ckpt(path: str, fallback: Optional[int]) -> int:
    bname = os.path.basename(path)
    m = re.search(r"epoch(?:epoch)?=0*([0-9]+)", bname)
    if m:
        return int(m.group(1))
    return int(fallback) if fallback is not None else 0

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

def extract_curve_dict(obj) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(obj, np.lib.npyio.NpzFile):
        obj = {k: obj[k] for k in obj.files}
    if isinstance(obj, dict):
        coords_keys = ("curve_coords", "coords", "xyz", "curve_xyz")
        ss_keys = ("ss_one_hot", "ss", "ss_oh")
        coords, ss = None, None
        for k in coords_keys:
            if k in obj:
                coords = obj[k]; break
        for k in ss_keys:
            if k in obj:
                ss = obj[k]; break
        if coords is None or ss is None:
            raise ValueError("missing keys like 'curve_coords' and 'ss_one_hot'")
        return np.asarray(coords, np.float32), np.asarray(ss, np.float32)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 2 and obj.shape[1] == 6:
            return obj[:, :3].astype(np.float32), obj[:, 3:].astype(np.float32)
        raise ValueError("raw ndarray must be shape [L,6]")
    raise ValueError("unsupported .npy content")

def load_curve_center_only(npy_path: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    raw = np.load(npy_path, allow_pickle=True)
    data = raw.item() if hasattr(raw, "item") and not isinstance(raw, np.lib.npyio.NpzFile) else raw
    xyz, ss_oh = extract_curve_dict(data)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("coords must be [L,3]")
    if ss_oh.ndim != 2 or ss_oh.shape[1] != 3:
        raise ValueError("ss_one_hot must be [L,3]")
    mean = xyz.mean(axis=0, keepdims=True)
    xyz_center = xyz - mean
    full = np.concatenate([xyz_center, ss_oh], axis=-1).astype(np.float32)
    x = torch.from_numpy(full).unsqueeze(0)  # [1,L,6]
    mask = torch.ones(1, x.size(1), dtype=torch.bool)
    return x, mask, mean.astype(np.float32)

def de_center(xyz_center: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return xyz_center + mean.reshape(1, 3)

def plot_proj(xyz: np.ndarray, out_png: Path, title: str = ""):
    fig = plt.figure(figsize=(8, 3), dpi=140)
    plt.subplot(1, 2, 1); plt.plot(xyz[:, 0], xyz[:, 1], linewidth=1)
    plt.title("XY"); plt.xlabel("x"); plt.ylabel("y")
    plt.subplot(1, 2, 2); plt.plot(xyz[:, 0], xyz[:, 2], linewidth=1)
    plt.title("XZ"); plt.xlabel("x"); plt.ylabel("z")
    plt.suptitle(title); plt.tight_layout()
    fig.savefig(str(out_png)); plt.close(fig)

def robust_quantize(q, z_e):
    # quantize in latent space (B,N,D); do not pass L-mask
    try:
        return q(z_e, mask=None, do_ema_update=False, allow_reinit=False)
    except TypeError:
        try:
            return q(z_e, mask=None)
        except TypeError:
            return q(z_e)

def get_alpha_from_model(model) -> float:
    if hasattr(model, "_phase_and_alpha") and callable(model._phase_and_alpha):
        try:
            _, alpha = model._phase_and_alpha()
            return float(alpha)
        except Exception:
            pass
    if hasattr(model, "current_mix_alpha"):
        try:
            return float(getattr(model, "current_mix_alpha"))
        except Exception:
            pass
    return 1.0


# ===== Main =====
def main():
    if SEED is not None:
        np.random.seed(SEED); torch.manual_seed(SEED)

    device = torch.device(DEVICE)
    _ = load_yaml(YAML_PATH)
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    # build Lightning experiment and load ckpt
    exp, _cfg = build_experiment_from_yaml(YAML_PATH)
    state = torch.load(CKPT_PATH, map_location="cpu")
    sd = state.get("state_dict", state)
    exp.load_state_dict(sd, strict=False)
    exp.to(device).eval()
    model = exp.model if hasattr(exp, "model") else exp

    epoch = parse_epoch_from_ckpt(CKPT_PATH, EPOCH_HINT)
    try:
        if hasattr(model, "set_epoch_context"):
            model.set_epoch_context(epoch=epoch, steps_per_epoch=int(STEPS_PER_EPOCH))
        print(f"[Align] set_epoch_context: epoch={epoch}, steps_per_epoch={STEPS_PER_EPOCH}")
    except Exception as e:
        print(f"[Align] set_epoch_context failed: {e}")

    print(f"[Info] alpha(from model)={get_alpha_from_model(model):.4f}")

    picked = pick_random_npy_fast(NPY_DIR, cap=MAX_SCAN)
    if picked is None:
        raise FileNotFoundError("No .npy files found in NPY_DIR.")
    x, mask, curve_mean = load_curve_center_only(picked)
    x = x.to(device); mask = mask.to(device)

    # ensure L <= max_seq_len
    L = x.size(1)
    maxL = getattr(model, "max_seq_len", L)
    if L > maxL:
        x = x[:, :maxL, :]
        mask = mask[:, :maxL]
        L = maxL

    # encode -> tokenize -> z_e [B,N,D]
    with torch.no_grad():
        enc_out = model.encode(x, mask=mask)
        h_tokens = enc_out[0] if isinstance(enc_out, (tuple, list)) else enc_out
        z_e = model._tokenize_to_codes(h_tokens, mask)

    # epsilon on z_e
    if EPS_MODE == "random":
        eps = torch.randn_like(z_e)
    elif EPS_MODE == "zero":
        eps = torch.zeros_like(z_e)
    elif EPS_MODE == "manual":
        gen = torch.Generator(device=device).manual_seed(EPS_MANUAL_SEED)
        eps = torch.randn_like(z_e, generator=gen)
    else:
        raise ValueError("Unknown EPS_MODE")
    ze_std = float(z_e.flatten().std(unbiased=False).item())
    if not np.isfinite(ze_std) or ze_std <= 0.0:
        ze_std = 1e-3
    eps = eps.mul(ze_std)

    # save original (de-centered)
    orig = x[0].detach().cpu().numpy()
    orig_xyz = de_center(orig[:, :3], curve_mean)
    orig_ss = orig[:, 3:]
    np.save(out_dir / "orig_curve.npy", np.concatenate([orig_xyz, orig_ss], axis=-1))
    plot_proj(orig_xyz, out_dir / "orig_curve.png", title=f"Original | {os.path.basename(picked)}")

    # quantizer
    q = getattr(model, "quantizer", None)
    if q is None:
        raise RuntimeError("Model has no 'quantizer' attribute.")

    # recon for each alpha (pure VQ)
    for a in ALPHAS:
        z_e_noisy = z_e + float(a) * eps
        q_out = robust_quantize(q, z_e_noisy)
        if isinstance(q_out, (tuple, list)):
            z_q_st = q_out[0]
        elif torch.is_tensor(q_out):
            z_q_st = q_out
        else:
            raise RuntimeError("Unsupported quantizer output type.")

        with torch.no_grad():
            y = model.decode(z_q_st, mask=mask)  # [B,L,6]
            y = y[0].detach()

        y_xyz = de_center(y[:, :3].cpu().numpy(), curve_mean)
        y_ss_idx = y[:, 3:].argmax(dim=-1)
        y_ss_oh = F.one_hot(y_ss_idx, num_classes=y[:, 3:].size(-1)).float().cpu().numpy()

        full = np.concatenate([y_xyz, y_ss_oh], axis=-1)
        tag = f"a{str(a).replace('.', 'p')}"
        np.save(out_dir / f"recon_{tag}.npy", full)
        plot_proj(y_xyz, out_dir / f"recon_{tag}.png",
                  title=f"epoch={epoch} | pure_vq | eps_alpha={a}")

    print(f"[Done] picked: {picked}")
    print(f"[Done] outputs -> {out_dir.resolve()}")

if __name__ == "__main__":
    main()
