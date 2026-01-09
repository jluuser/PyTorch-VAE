#!/usr/bin/env python3
# coding: utf-8
"""
Reconstruction probe for current VQ-VAE (supports residual VQ).

Pipeline:
  raw curve .npy -> center coords -> encode + quantize -> discrete indices 
  -> indices to latent z_q -> decode(z_q, mask=target_mask_of_length_L)
  -> save original and reconstruction (.npy and XY/XZ plots).

This script simply passes a real sample through the VQ-VAE pipeline to verify reconstruction capability.
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import yaml
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== User paths (configured based on your previous input) =====
CKPT_PATH = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/checkpoints/vq_token64_K1024_D512_ResidualVQ_fromscratch/epochepoch=139.ckpt"
YAML_PATH = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/configs/stage2_vq.yaml"
NPY_DIR   = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/prp-dataset/filtered_curves_npy/"
OUT_DIR   = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/probe_vq_recon_residual/"

STEPS_PER_EPOCH = 420
MAX_SCAN = 50
EPOCH_HINT: Optional[int] = None
SEED = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Repo root on sys.path =====
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment import build_experiment_from_yaml


# ===== Basic utils =====
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


def load_curve_center_only(npy_path: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
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
    full = np.concatenate([xyz_center, ss_oh], axis=-1).astype(np.float32)
    x = torch.from_numpy(full).unsqueeze(0)  # [1,L,6]
    mask = torch.ones(1, x.size(1), dtype=torch.bool)
    return x, mask, mean.astype(np.float32)


def de_center(xyz_center: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return xyz_center + mean.reshape(1, 3)


def plot_proj(xyz: np.ndarray, out_png: Path, title: str = ""):
    fig = plt.figure(figsize=(8, 3), dpi=140)
    plt.subplot(1, 2, 1)
    plt.plot(xyz[:, 0], xyz[:, 1], linewidth=1)
    plt.title("XY")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(1, 2, 2)
    plt.plot(xyz[:, 0], xyz[:, 2], linewidth=1)
    plt.title("XZ")
    plt.xlabel("x")
    plt.ylabel("z")

    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(str(out_png))
    plt.close(fig)


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


# ===== RVQ / indices helpers =====
@torch.no_grad()
def _fallback_tokenize(core, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Fallback path: old-style encode -> tokenizer -> to_code.
    features: [B, L, H]
    """
    if not hasattr(core, "tokenizer") or not hasattr(core, "to_code"):
        raise RuntimeError("encode() returned [B,L,H] but model has no tokenizer/to_code.")
    kpm = (~mask) if mask is not None else None
    h_mem = core.tokenizer(features, key_padding_mask=kpm)  # [B, N, H]
    z_e = core.to_code(h_mem)  # [B, N, D]
    return z_e


@torch.no_grad()
def _ensure_batch_first_2d(
    indices: torch.Tensor,
    mask: torch.Tensor,
    num_quantizers: int = 1,
    latent_tokens: Optional[int] = None,
) -> torch.Tensor:
    """
    Normalize indices shape to [B, N] with batch_first.
    Handles Residual VQ (permute Q,B,M -> B,M,Q -> flatten).
    """
    indices = indices.long()
    B = mask.size(0)

    if indices.dim() == 1 and num_quantizers > 1:
        N_flat = indices.numel()
        base = B * num_quantizers
        if N_flat % base != 0:
            raise RuntimeError(
                f"RVQ indices length {N_flat} not divisible by B*num_quantizers={base}"
            )
        M = N_flat // base
        if latent_tokens is not None and M != int(latent_tokens):
            print(f"[warn] RVQ inferred tokens={M} != latent_tokens={latent_tokens}")
        idx = indices.view(num_quantizers, B, M)       # [Q, B, M]
        idx = idx.permute(1, 2, 0).contiguous()       # [B, M, Q]
        return idx.view(B, M * num_quantizers)        # [B, M*Q]

    if indices.dim() == 2:
        if indices.size(0) == B:
            return indices
        if indices.size(1) == B:
            return indices.transpose(0, 1)
        if indices.numel() % B != 0:
            raise RuntimeError(
                f"Cannot reshape indices of shape {tuple(indices.shape)} to [B,N] with B={B}"
            )
        return indices.reshape(B, -1)

    if indices.dim() == 1:
        N_flat = indices.numel()
        if N_flat % B != 0:
            raise RuntimeError(
                f"1D indices length {N_flat} not divisible by batch size {B}"
            )
        N = N_flat // B
        return indices.view(B, N)

    if indices.dim() == 3:
        if indices.size(0) == B:
            return indices.reshape(B, -1)
        if indices.size(1) == B:
            return indices.permute(1, 0, 2).reshape(B, -1)
        if indices.size(2) == B:
            return indices.permute(2, 0, 1).reshape(B, -1)
        if indices.numel() % B != 0:
            raise RuntimeError(
                f"Cannot reshape 3D indices {tuple(indices.shape)} to [B,N] with B={B}"
            )
        return indices.reshape(B, -1)

    raise RuntimeError(
        f"Unsupported indices dim={indices.dim()} with shape {tuple(indices.shape)}"
    )


@torch.no_grad()
def tokenize_to_indices(core, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Unified encode + quantize -> discrete indices [B, N_flat] (batch_first).
    """
    dev = next(core.parameters()).device
    x = x.to(dev, non_blocking=True)
    mask = mask.to(dev, non_blocking=True)

    if not hasattr(core, "encode") or not callable(core.encode):
        raise RuntimeError("VQVAE model must implement encode(x, mask=mask).")

    enc_out = core.encode(x, mask=mask)
    q = getattr(core, "quantizer", None)
    if q is None:
        raise RuntimeError("model.quantizer not found")

    latent_tokens = getattr(core, "latent_n_tokens", None)
    if latent_tokens is None:
        latent_tokens = getattr(core, "latent_tokens", None)

    num_quantizers = int(getattr(q, "num_quantizers", 1))

    indices: torch.Tensor

    # Handle various return types from encode()
    if isinstance(enc_out, (tuple, list)):
        cand_idx = None
        first_tensor = None
        for item in enc_out:
            if torch.is_tensor(item):
                if first_tensor is None:
                    first_tensor = item
                if item.dtype in (torch.int64, torch.int32) and item.dim() >= 1:
                    cand_idx = item
                    break
        if cand_idx is not None:
            indices = cand_idx.long()
        else:
            if not torch.is_tensor(enc_out[0]):
                raise RuntimeError("encode() output[0] must be a tensor.")
            feats = enc_out[0]
            B, T, _ = feats.shape
            if latent_tokens is not None and T == int(latent_tokens):
                z_e = feats
            else:
                z_e = _fallback_tokenize(core, feats, mask)
            q_out = q(z_e, do_ema_update=False, allow_reinit=False, mask=None)
            if isinstance(q_out, (tuple, list)) and len(q_out) >= 3:
                indices = q_out[2]
            elif torch.is_tensor(q_out):
                indices = q_out
            else:
                raise RuntimeError("Unsupported quantizer output type.")
    else:
        feats = enc_out
        B, T, _ = feats.shape
        if latent_tokens is not None and T == int(latent_tokens):
            z_e = feats
        else:
            z_e = _fallback_tokenize(core, feats, mask)
        q_out = q(z_e, do_ema_update=False, allow_reinit=False, mask=None)
        if isinstance(q_out, (tuple, list)) and len(q_out) >= 3:
            indices = q_out[2]
        elif torch.is_tensor(q_out):
            indices = q_out
        else:
            raise RuntimeError("Unsupported quantizer output type.")

    indices = _ensure_batch_first_2d(
        indices,
        mask,
        num_quantizers=num_quantizers,
        latent_tokens=int(latent_tokens) if latent_tokens is not None else None,
    )
    return indices  # [B, N_flat]


@torch.no_grad()
def get_codebook(core) -> torch.Tensor:
    q = getattr(core, "quantizer", None)
    if q is None:
        raise RuntimeError("model.quantizer missing")
    emb = getattr(q, "embedding", None)
    if emb is None:
        raise RuntimeError("quantizer.embedding missing")
    if torch.is_tensor(emb):
        return emb
    if hasattr(emb, "weight") and torch.is_tensor(emb.weight):
        return emb.weight
    raise RuntimeError("unsupported codebook type")


@torch.no_grad()
def get_num_quantizers(core) -> int:
    q = getattr(core, "quantizer", None)
    if q is None:
        return 1
    n_q = getattr(q, "num_quantizers", 1)
    try:
        return int(n_q)
    except Exception:
        return 1


@torch.no_grad()
def indices_to_latent(core, indices_np: np.ndarray) -> torch.Tensor:
    """
    Map discrete indices back to latent z_q.
    Handles summing up residual codes for RVQ.
    """
    if indices_np.ndim != 1:
        raise ValueError(f"indices must be 1D, got {indices_np.shape}")
    dev = next(core.parameters()).device
    inds = torch.from_numpy(indices_np).long().unsqueeze(0).to(dev)  # [1, N_flat]
    E = get_codebook(core).to(dev)  # [K_total, D]

    num_q = get_num_quantizers(core)
    # Simple single-codebook case
    if num_q <= 1:
        z_q = F.embedding(inds, E)  # [1, N_flat, D]
        return z_q

    # Residual VQ case
    N_flat = int(inds.shape[1])
    if N_flat % num_q != 0:
        raise ValueError(
            f"flattened indices length {N_flat} is not divisible by num_quantizers={num_q}"
        )
    N_tokens = N_flat // num_q

    z_all = F.embedding(inds, E)          # [1, N_flat, D]
    z_all = z_all.view(1, N_tokens, num_q, -1)  # [1, N_tokens, Q, D]
    z_q = z_all.sum(dim=2)                # [1, N_tokens, D]
    return z_q


# ===== Main =====
def main():
    if SEED is not None:
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    device = torch.device(DEVICE)
    _ = load_yaml(YAML_PATH)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build experiment and load checkpoint
    exp, _cfg = build_experiment_from_yaml(YAML_PATH)
    state = torch.load(CKPT_PATH, map_location="cpu")
    sd = state.get("state_dict", state)
    exp.load_state_dict(sd, strict=False)
    exp.to(device).eval()
    model = exp.model if hasattr(exp, "model") else exp

    # 2. Context setup
    epoch = parse_epoch_from_ckpt(CKPT_PATH, EPOCH_HINT)
    try:
        if hasattr(model, "set_epoch_context"):
            model.set_epoch_context(epoch=epoch, steps_per_epoch=int(STEPS_PER_EPOCH))
        print(f"[Align] set_epoch_context: epoch={epoch}, steps_per_epoch={STEPS_PER_EPOCH}")
    except Exception as e:
        print(f"[Align] set_epoch_context failed: {e}")

    print(f"[Info] alpha(from model)={get_alpha_from_model(model):.4f}")

    # 3. Load random data sample
    picked = pick_random_npy_fast(NPY_DIR, cap=MAX_SCAN)
    if picked is None:
        raise FileNotFoundError("No .npy files found in NPY_DIR.")
    x, src_mask, curve_mean = load_curve_center_only(picked)
    x = x.to(device)
    src_mask = src_mask.to(device)

    # Ensure length matches model constraint
    L = x.size(1)
    maxL = getattr(model, "max_seq_len", L)
    if L > maxL:
        x = x[:, :maxL, :]
        src_mask = src_mask[:, :maxL]
        L = maxL

    # 4. Save original
    orig = x[0].detach().cpu().numpy()
    orig_xyz = de_center(orig[:, :3], curve_mean)
    orig_ss = orig[:, 3:]
    np.save(out_dir / "orig_curve.npy", np.concatenate([orig_xyz, orig_ss], axis=-1))
    plot_proj(orig_xyz, out_dir / "orig_curve.png",
              title=f"Original | {os.path.basename(picked)}")

    # 5. Pass through VQ-VAE
    # Step A: Encode -> Quantize -> Indices
    indices_bt = tokenize_to_indices(model, x, src_mask)  # [1, N_flat]
    indices_np = indices_bt[0].detach().cpu().numpy()
    latent_len = int(indices_np.shape[0])
    print(f"[Info] latent_len (flattened) = {latent_len}")

    # Step B: Indices -> Embeddings (z_q)
    z_q = indices_to_latent(model, indices_np)  # [1, N_latent, D]
    B, N_latent, D = z_q.shape
    
    # [FIX] Use a mask that represents the ORIGINAL target length (L)
    # The decoder uses this mask length to determine the output sequence length.
    target_mask = torch.ones(B, int(L), dtype=torch.bool, device=z_q.device)

    # Step C: Decode
    with torch.no_grad():
        try:
            # Pass the target_mask so decoder knows to generate L points
            y = model.decode(z_q, mask=target_mask)
        except TypeError:
            try:
                # Fallback attempts
                y = model.decode(z_q, target_len=int(L))
            except TypeError:
                y = model.decode(z_q)
        
        y = y[0].detach()

    # 6. Save reconstruction
    y_xyz = de_center(y[:, :3].cpu().numpy(), curve_mean)
    y_ss_idx = y[:, 3:].argmax(dim=-1)
    y_ss_oh = F.one_hot(y_ss_idx, num_classes=y[:, 3:].size(-1)).float().cpu().numpy()
    full = np.concatenate([y_xyz, y_ss_oh], axis=-1)

    np.save(out_dir / "recon_vq.npy", full)
    plot_proj(y_xyz, out_dir / "recon_vq.png",
              title=f"epoch={epoch} | pure_vq_recon")

    print(f"[Done] picked: {picked}")
    print(f"[Done] outputs -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()