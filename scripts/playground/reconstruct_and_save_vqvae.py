#!/usr/bin/env python3
# coding: utf-8
"""
ASCII-only reconstruction script (no noise, single NPY output) for a specific curve .npy using VQVAE.

- Fixed absolute NPY_PATH; no CLI args.
- Saves ONLY ONE .npy reconstruction file into the SAME directory as NPY_PATH.
- Optionally aligns model.skip_scale from yaml schedule at the checkpoint epoch.
- Reproduces training-time z_e <-> z_q_st warmup interpolation using
  vq_forward_warmup_steps and an estimated training step s = epoch * STEPS_PER_EPOCH.
- Center-only normalization for xyz; denormalize back by adding curve mean.

Edit constants (CKPT_PATH, YAML_PATH, NPY_PATH, EPOCH_HINT, STEPS_PER_EPOCH) as needed.
"""

import sys
import re
from pathlib import Path

# Optional: add project root if needed
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../PyTorch-VAE
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import locale
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# ======= USER SETTINGS =======
CKPT_PATH = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/vq_new_checkpoints/epochepoch=029.ckpt"
YAML_PATH = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/configs/stage2_vq.yaml"
NPY_PATH  = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/test_new_curve/6knm_binder_curve_std.npy"

# If your ckpt name lacks epoch number, set EPOCH_HINT manually (int).
EPOCH_HINT = None  # e.g., 79

# Steps per epoch during training (for z_e <-> z_q_st warmup interpolation)
STEPS_PER_EPOCH = 420  # set to your actual steps/epoch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======= HELPERS =======

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_model(cfg: dict, device: torch.device):
    from models.vq_vae import VQVAE
    model = VQVAE(**cfg["model_params"]).to(device)
    return model

def robust_load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> str:
    ck = torch.load(ckpt_path, map_location=device)
    if not isinstance(ck, dict):
        raise RuntimeError(f"Checkpoint at {ckpt_path} is not a dict or state_dict.")
    if "state_dict" in ck and isinstance(ck["state_dict"], dict):
        sd = ck["state_dict"]
    elif "model_state_dict" in ck and isinstance(ck["model_state_dict"], dict):
        sd = ck["model_state_dict"]
    else:
        sd = ck

    def strip_prefixes(state_dict):
        prefixes = ("module.", "model.", "net.")
        any_pref = any(k.startswith(prefixes) for k in state_dict.keys())
        if not any_pref:
            return state_dict
        new = {}
        for k, v in state_dict.items():
            name = k
            for p in prefixes:
                if name.startswith(p):
                    name = name[len(p):]
            new[name] = v
        return new

    try:
        model.load_state_dict(sd)
        return "loaded_exact"
    except Exception:
        pass
    sd_stripped = strip_prefixes(sd)
    try:
        model.load_state_dict(sd_stripped)
        return "loaded_stripped_prefixes"
    except Exception:
        pass

    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd_stripped.items()
                if k in model_sd and getattr(v, "shape", None) == getattr(model_sd[k], "shape", None)}
    if filtered:
        missing = set(model_sd.keys()) - set(filtered.keys())
        model.load_state_dict(filtered, strict=False)
        return f"partial_loaded_{len(filtered)}_matched_{len(missing)}_missing"
    raise RuntimeError(f"Failed to load checkpoint {ckpt_path}: no matching params found.")

def _extract_curve_dict(data_obj) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(data_obj, np.lib.npyio.NpzFile):
        data_obj = {k: data_obj[k] for k in data_obj.files}
    if isinstance(data_obj, dict):
        coords_keys = ("curve_coords", "coords", "xyz", "curve_xyz")
        ss_keys = ("ss_one_hot", "ss", "ss_oh")
        coords, ss = None, None
        for k in coords_keys:
            if k in data_obj:
                coords = data_obj[k]; break
        for k in ss_keys:
            if k in data_obj:
                ss = data_obj[k]; break
        if coords is None or ss is None:
            raise ValueError("Loaded .npy dict missing keys like 'curve_coords' and 'ss_one_hot'.")
        return np.asarray(coords, dtype=np.float32), np.asarray(ss, dtype=np.float32)
    if isinstance(data_obj, np.ndarray):
        if data_obj.ndim == 2 and data_obj.shape[1] == 6:
            return data_obj[:, :3].astype(np.float32), data_obj[:, 3:].astype(np.float32)
        raise ValueError("Loaded .npy is ndarray but not shape [L,6].")
    raise ValueError("Unsupported .npy content. Expected dict or ndarray or npz.")

def load_and_preprocess_single_curve(npy_path: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    raw = np.load(npy_path, allow_pickle=True)
    data_obj = raw.item() if hasattr(raw, "item") and not isinstance(raw, np.lib.npyio.NpzFile) else raw
    coords, ss_one = _extract_curve_dict(data_obj)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be [L,3]..")
    if ss_one.ndim != 2 or ss_one.shape[1] != 3:
        raise ValueError("ss_one_hot must be [L,3].")
    curve_mean = coords.mean(axis=0, keepdims=True)
    coords_centered = coords - curve_mean
    full = np.concatenate([coords_centered, ss_one], axis=-1).astype(np.float32)
    x = torch.from_numpy(full).unsqueeze(0)
    L = x.size(1)
    mask = torch.ones(1, L, dtype=torch.bool)
    return x, mask, curve_mean.astype(np.float32)

def denorm_xyz_center_only(xyz_centered: np.ndarray, curve_mean: np.ndarray) -> np.ndarray:
    return xyz_centered + curve_mean.reshape(1, 3)

# ---------- schedule helpers ----------

def parse_epoch_from_ckpt(path: str, fallback: Optional[int]) -> int:
    bname = os.path.basename(path)
    m = re.search(r"epoch(?:epoch)?=0*([0-9]+)", bname)
    if m:
        return int(m.group(1))
    return int(fallback) if fallback is not None else 0

def interp_schedule(pairs, epoch: int) -> float:
    """
    pairs: [[e0, v0], [e1, v1], ...] in ascending epoch order
    returns linearly interpolated value at 'epoch'
    """
    if not pairs:
        raise ValueError("Empty schedule pairs")
    if epoch <= pairs[0][0]:
        return float(pairs[0][1])
    if epoch >= pairs[-1][0]:
        return float(pairs[-1][1])
    for i in range(1, len(pairs)):
        e0, v0 = pairs[i-1]
        e1, v1 = pairs[i]
        if e0 <= epoch <= e1:
            if e1 == e0:
                return float(v1)
            t = (epoch - e0) / float(e1 - e0)
            return float(v0 + t * (v1 - v0))
    return float(pairs[-1][1])

# ======= MAIN =======

@torch.no_grad()
def main():
    try:
        locale.setlocale(locale.LC_ALL, "C")
    except Exception:
        pass

    device = torch.device(DEVICE)
    cfg = load_config(YAML_PATH)

    # Resolve paths
    npy_path = Path(NPY_PATH).resolve()
    if not npy_path.exists():
        raise FileNotFoundError(f"NPY file not found: {npy_path}")
    out_dir = npy_path.parent
    base_stem = npy_path.stem  # e.g., "6knm_binder_curve_std"

    # 0) epoch from ckpt (or hint) — kept for logging only
    epoch = parse_epoch_from_ckpt(CKPT_PATH, EPOCH_HINT)
    print(f"[Info] Using epoch={epoch} for schedule alignment.")

    # 1) build model and load checkpoint
    model = build_model(cfg, device)
    status = robust_load_checkpoint(model, CKPT_PATH, device)
    print(f"[Info] checkpoint load status: {status}")
    model.eval()

    # 1.1) align skip_scale from schedule (kept for visibility)
    try:
        schedule = cfg.get("exp_params", {}).get("schedules", {}).get("skip_scale", None)
        if schedule:
            ss_val = interp_schedule(schedule, epoch)
            if hasattr(model, "skip_scale"):
                model.skip_scale = float(ss_val)
            print(f"[Align] skip_scale set to {getattr(model, 'skip_scale', ss_val):.6f} at epoch {epoch}.")
        else:
            print("[Align] no skip_scale schedule found; using model default.")
    except Exception as e:
        print(f"[Align] failed to set skip_scale from schedule: {e}")

    # 2) load curve and preprocess (center-only)
    x, mask, curve_mean = load_and_preprocess_single_curve(str(npy_path))
    x, mask = x.to(device), mask.to(device)

    # 3) encode -> quantize (no EMA)
    z_e, h_enc = model.encode(x, mask=mask)  # [1,L,D], [1,L,H]
    z_q_st, z_q_raw, indices, stats = model.quantizer(
        z_e, do_ema_update=False, allow_reinit=False, mask=mask
    )

    # ---- PURE CODEBOOK RECONSTRUCTION CHANGES START ----
    # Force pure codebook path: use z_q_st only, no z_e mixing.
    z_for_decode = z_q_st
    print("[Align] pure codebook path: use z_q_st only (no z_e mix).")

    # Temporarily disable skip bypass: set skip_scale=0 and do not pass skip_memory.
    prev_skip = float(getattr(model, "skip_scale", 0.0))
    model.skip_scale = 0.0
    try:
        y_t = model.decode(z_for_decode, mask=mask, drop_history=False, skip_memory=None)[0].detach()
    finally:
        model.skip_scale = prev_skip
    # ---- PURE CODEBOOK RECONSTRUCTION CHANGES END ----

    # Denorm xyz back by adding curve mean; ss -> one-hot
    y_xyz_center = y_t[:, :3].cpu().numpy()
    y_ss_logits  = y_t[:, 3:]
    y_xyz = denorm_xyz_center_only(y_xyz_center, curve_mean)
    y_ss_idx = y_ss_logits.argmax(dim=-1)
    y_ss_oh = F.one_hot(y_ss_idx, num_classes=y_ss_logits.size(-1)).float().cpu().numpy()

    # Save ONLY ONE npy: pure codebook recon
    # Use a distinct suffix to avoid confusion with training-like recon outputs.
    recon = np.concatenate([y_xyz, y_ss_oh], axis=-1)
    out_path = out_dir / f"{base_stem}_recon_codebook.npy"
    np.save(out_path, recon)

    print(f"[Done] Output saved: {out_path}")

if __name__ == "__main__":
    main()
