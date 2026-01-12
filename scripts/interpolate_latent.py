#!/usr/bin/env python3
# coding: utf-8
"""
Latent interpolation probe for current VQ-VAE (supports residual VQ).

For each pair of curves:
  - Load two original curves A and B from NPY_DIR.
  - Center coordinates and build [1, L, 6] inputs.
  - Encode to pre-VQ latents z_e_A, z_e_B.
  - Quantize and decode each individually to get A_recon and B_recon.
  - For a list of alphas, compute:
        z_e_mix = alpha * z_e_A + (1 - alpha) * z_e_B
        z_q_mix = quantizer(z_e_mix)
        decode(z_q_mix, mask=target_mask)
    and save the mixed curves.

Outputs:
  OUT_DIR/pair_0000/
      A_orig.npy
      B_orig.npy
      A_recon.npy
      B_recon.npy
      mix_alpha0.10.npy
      mix_alpha0.30.npy
      ...
"""

import os
import re
import sys
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import yaml
import torch
import torch.nn.functional as F

# ===================== User configuration ===================== #

CKPT_PATH = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/checkpoints/vq_token64_K1024_D512_ResidualVQ_fromscratch/epochepoch=139.ckpt"
YAML_PATH = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/configs/stage2_vq.yaml"
NPY_DIR   = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/prp-dataset/filtered_curves_npy/"
OUT_DIR   = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/probe_vq_interpolate/"

STEPS_PER_EPOCH = 420
EPOCH_HINT: Optional[int] = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_PAIRS = 5
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]

SEED: Optional[int] = 1234

# =============================================================== #

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment import build_experiment_from_yaml  # type: ignore


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_epoch_from_ckpt(path: str, fallback: Optional[int]) -> int:
    bname = os.path.basename(path)
    m = re.search(r"epoch(?:epoch)?=0*([0-9]+)", bname)
    if m:
        return int(m.group(1))
    return int(fallback) if fallback is not None else 0


def list_npy_files(dir_path: str) -> List[str]:
    p = Path(dir_path)
    if not p.is_dir():
        raise FileNotFoundError(f"NPY_DIR not found: {dir_path}")
    files = [str(x) for x in p.iterdir() if x.is_file() and x.suffix == ".npy"]
    files.sort()
    return files


def extract_curve_dict(obj) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(obj, np.lib.npyio.NpzFile):
        obj = {k: obj[k] for k in obj.files}
    if isinstance(obj, dict):
        coords_keys = ("curve_coords", "coords", "xyz", "curve_xyz")
        ss_keys = ("ss_one_hot", "ss", "ss_oh")
        coords, ss = None, None
        for k in coords_keys:
            if k in obj:
                coords = obj[k]
                break
        for k in ss_keys:
            if k in obj:
                ss = obj[k]
                break
        if coords is None or ss is None:
            raise ValueError("missing keys like 'curve_coords' and 'ss_one_hot'")
        return np.asarray(coords, np.float32), np.asarray(ss, np.float32)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 2 and obj.shape[1] == 6:
            return obj[:, :3].astype(np.float32), obj[:, 3:].astype(np.float32)
        raise ValueError("raw ndarray must be shape [L,6]")
    raise ValueError("unsupported .npy content")


def load_curve_center_only(npy_path: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    raw = np.load(npy_path, allow_pickle=True)
    if isinstance(raw, np.lib.npyio.NpzFile):
        data = {k: raw[k] for k in raw.files}
    elif isinstance(raw, dict):
        data = raw
    else:
        try:
            data = raw.item()
        except Exception:
            data = raw

    xyz, ss_oh = extract_curve_dict(data)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("coords must be [L,3]")
    if ss_oh.ndim != 2 or ss_oh.shape[1] != 3:
        raise ValueError("ss_one_hot must be [L,3]")

    mean = xyz.mean(axis=0, keepdims=True)
    xyz_center = xyz - mean
    full_center = np.concatenate([xyz_center, ss_oh], axis=-1).astype(np.float32)
    full_orig = np.concatenate([xyz, ss_oh], axis=-1).astype(np.float32)

    x = torch.from_numpy(full_center).unsqueeze(0)
    mask = torch.ones(1, x.size(1), dtype=torch.bool)
    return x, mask, mean.astype(np.float32), full_orig


def de_center(xyz_center: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return xyz_center + mean.reshape(1, 3)


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


@torch.no_grad()
def _fallback_tokenize(core, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not hasattr(core, "tokenizer") or not hasattr(core, "to_code"):
        raise RuntimeError("encode() returned [B,L,H] but model has no tokenizer/to_code.")
    kpm = (~mask) if mask is not None else None
    h_mem = core.tokenizer(features, key_padding_mask=kpm)
    z_e = core.to_code(h_mem)
    return z_e


@torch.no_grad()
def encode_to_ze(core, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dev = next(core.parameters()).device
    x = x.to(dev, non_blocking=True)
    mask = mask.to(dev, non_blocking=True)

    if not hasattr(core, "encode") or not callable(core.encode):
        raise RuntimeError("VQVAE model must implement encode(x, mask=mask).")

    enc_out = core.encode(x, mask=mask)

    latent_tokens = getattr(core, "latent_n_tokens", None)
    if latent_tokens is None:
        latent_tokens = getattr(core, "latent_tokens", None)

    if isinstance(enc_out, (tuple, list)):
        feat = None
        for item in enc_out:
            if torch.is_tensor(item) and item.dim() == 3:
                feat = item
                break
        if feat is None:
            raise RuntimeError("encode() did not return a 3D tensor as features.")
        B, T, _ = feat.shape
        if latent_tokens is not None and T == int(latent_tokens):
            z_e = feat
        else:
            z_e = _fallback_tokenize(core, feat, mask)
    else:
        feat = enc_out
        if not torch.is_tensor(feat) or feat.dim() != 3:
            raise RuntimeError("encode() output must be a 3D tensor.")
        B, T, _ = feat.shape
        if latent_tokens is not None and T == int(latent_tokens):
            z_e = feat
        else:
            z_e = _fallback_tokenize(core, feat, mask)

    return z_e  # [B, N_latent, D]


@torch.no_grad()
def get_quantizer(core):
    q = getattr(core, "quantizer", None)
    if q is None:
        raise RuntimeError("model.quantizer not found")
    return q


@torch.no_grad()
def quantize_latent(core, z_e: torch.Tensor) -> torch.Tensor:
    q = get_quantizer(core)
    q_out = q(z_e, do_ema_update=False, allow_reinit=False, mask=None)
    if isinstance(q_out, (tuple, list)):
        z_q = q_out[0]
    else:
        z_q = q_out
    return z_q  # [B, N_latent, D]


@torch.no_grad()
def decode_latent(core, z_q: torch.Tensor, target_len: int) -> np.ndarray:
    dev = z_q.device
    B = z_q.size(0)
    if B != 1:
        raise ValueError("decode_latent expects batch size 1.")
    target_mask = torch.ones(1, int(target_len), dtype=torch.bool, device=dev)

    try:
        y = core.decode(z_q, mask=target_mask)
    except TypeError:
        try:
            y = core.decode(z_q, target_len=int(target_len))
        except TypeError:
            y = core.decode(z_q)
    y = y[0].detach().cpu()
    y_xyz = y[:, :3].numpy()
    y_ss_logits = y[:, 3:]
    y_ss_idx = y_ss_logits.argmax(dim=-1)
    y_ss_oh = F.one_hot(y_ss_idx, num_classes=y_ss_logits.size(-1)).float().numpy()
    full = np.concatenate([y_xyz, y_ss_oh], axis=-1)
    return full  # [L, 6]


def pick_two_files(npy_files: List[str]) -> Tuple[str, str]:
    if len(npy_files) < 2:
        raise RuntimeError("Not enough .npy files to form a pair.")
    a, b = random.sample(npy_files, 2)
    return a, b


def main():
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    device = torch.device(DEVICE)
    _ = load_yaml(YAML_PATH)

    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

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

    npy_files = list_npy_files(NPY_DIR)
    print(f"[Info] found {len(npy_files)} curve files in {NPY_DIR}")

    pair_count = 0
    while pair_count < N_PAIRS:
        path_a, path_b = pick_two_files(npy_files)
        print(f"[Pair {pair_count}] A={os.path.basename(path_a)}, B={os.path.basename(path_b)}")

        try:
            x_a, mask_a, mean_a, full_orig_a = load_curve_center_only(path_a)
            x_b, mask_b, mean_b, full_orig_b = load_curve_center_only(path_b)
        except Exception as e:
            print(f"[Warn] failed to load pair: {e}")
            continue

        x_a = x_a.to(device)
        mask_a = mask_a.to(device)
        x_b = x_b.to(device)
        mask_b = mask_b.to(device)

        L_a = x_a.size(1)
        L_b = x_b.size(1)
        L_mix = min(L_a, L_b)

        try:
            z_e_a = encode_to_ze(model, x_a, mask_a)
            z_e_b = encode_to_ze(model, x_b, mask_b)
        except Exception as e:
            print(f"[Warn] encode_to_ze failed: {e}")
            continue

        if z_e_a.shape != z_e_b.shape:
            print(f"[Warn] z_e shapes differ: {z_e_a.shape} vs {z_e_b.shape}, skipping pair.")
            continue

        try:
            z_q_a = quantize_latent(model, z_e_a)
            z_q_b = quantize_latent(model, z_e_b)
        except Exception as e:
            print(f"[Warn] quantize_latent failed: {e}")
            continue

        try:
            recon_a_center = decode_latent(model, z_q_a, target_len=L_a)
            recon_b_center = decode_latent(model, z_q_b, target_len=L_b)
        except Exception as e:
            print(f"[Warn] decode_latent for A/B failed: {e}")
            continue

        recon_a_xyz = de_center(recon_a_center[:, :3], mean_a)
        recon_b_xyz = de_center(recon_b_center[:, :3], mean_b)
        recon_a = np.concatenate([recon_a_xyz, recon_a_center[:, 3:]], axis=-1)
        recon_b = np.concatenate([recon_b_xyz, recon_b_center[:, 3:]], axis=-1)

        pair_dir = out_root / f"pair_{pair_count:04d}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        np.save(pair_dir / "A_orig.npy", full_orig_a)
        np.save(pair_dir / "B_orig.npy", full_orig_b)
        np.save(pair_dir / "A_recon.npy", recon_a)
        np.save(pair_dir / "B_recon.npy", recon_b)

        for alpha in ALPHAS:
            alpha = float(alpha)
            beta = 1.0 - alpha

            z_e_mix = alpha * z_e_a + beta * z_e_b
            try:
                z_q_mix = quantize_latent(model, z_e_mix)
                recon_mix_center = decode_latent(model, z_q_mix, target_len=L_mix)
            except Exception as e:
                print(f"[Warn] decode_latent for mix alpha={alpha} failed: {e}")
                continue

            mix_mean = alpha * mean_a + beta * mean_b
            mix_xyz = de_center(recon_mix_center[:, :3], mix_mean)
            mix_full = np.concatenate([mix_xyz, recon_mix_center[:, 3:]], axis=-1)

            out_name = pair_dir / f"mix_alpha{alpha:.2f}.npy"
            np.save(out_name, mix_full)

        print(f"[Done] saved pair_{pair_count:04d} to {pair_dir.resolve()}")
        pair_count += 1

    print(f"[Finished] generated {pair_count} pairs in {OUT_DIR}")


if __name__ == "__main__":
    main()
