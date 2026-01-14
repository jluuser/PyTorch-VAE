# coding: utf-8

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


try:
    from experiment import build_experiment_from_yaml
except Exception:
    build_experiment_from_yaml = None


def load_vq_experiment(vq_ckpt: str, vq_yaml: str, device: torch.device):
    if build_experiment_from_yaml is None:
        raise RuntimeError("Cannot import build_experiment_from_yaml from experiment.py")
    exp, _ = build_experiment_from_yaml(vq_yaml)
    state = torch.load(vq_ckpt, map_location="cpu")
    sd = state.get("state_dict", state)
    exp.load_state_dict(sd, strict=False)
    exp.to(device)
    exp.eval()
    return exp


def core_model(exp_or_model):
    return exp_or_model.model if hasattr(exp_or_model, "model") else exp_or_model


@dataclass
class VQInfo:
    codebook: torch.Tensor
    num_quantizers: int
    code_dim: int
    K_total: int


@torch.no_grad()
def get_vq_info(core) -> VQInfo:
    q = getattr(core, "quantizer", None)
    if q is None:
        raise RuntimeError("model.quantizer not found")
    emb = getattr(q, "embedding", None)
    if emb is None or (not torch.is_tensor(emb)):
        raise RuntimeError("quantizer.embedding must be a Tensor buffer")
    num_q = int(getattr(q, "num_quantizers", 1))
    K_total, D = int(emb.shape[0]), int(emb.shape[1])
    return VQInfo(codebook=emb, num_quantizers=max(1, num_q), code_dim=D, K_total=K_total)


def _extended_codebook(codebook: torch.Tensor, pad_id: int) -> torch.Tensor:
    K = int(codebook.shape[0])
    if pad_id == K:
        pad = torch.zeros((1, codebook.shape[1]), dtype=codebook.dtype, device=codebook.device)
        return torch.cat([codebook, pad], dim=0)
    return codebook


@torch.no_grad()
def indices_to_latent_sum(
    codebook: torch.Tensor,
    indices_flat: torch.Tensor,
    num_quantizers: int,
    pad_id: Optional[int] = None,
    return_token_mask: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Map flat RVQ indices to token-level latent vectors by summing residual levels.

    indices_flat: [B, L_flat] where L_flat = M * num_quantizers (interleaved per token)
    returns:
      z0: [B, M, D]
      token_mask: [B, M] bool (True=valid) or None
    """
    if indices_flat.dim() != 2:
        raise ValueError(f"indices_flat must be [B,L], got {indices_flat.shape}")
    B, L = int(indices_flat.size(0)), int(indices_flat.size(1))
    num_q = max(1, int(num_quantizers))

    if pad_id is None:
        pad_id = -1
    emb = _extended_codebook(codebook, int(pad_id))

    if num_q == 1:
        z = F.embedding(indices_flat.clamp(min=0), emb)
        token_mask = None
        if return_token_mask:
            token_mask = (indices_flat != int(pad_id))
        return z, token_mask

    if L % num_q != 0:
        raise ValueError(f"L_flat={L} not divisible by num_quantizers={num_q}")
    M = L // num_q

    z_all = F.embedding(indices_flat.clamp(min=0), emb)  # [B, L, D]
    z_all = z_all.view(B, M, num_q, -1)
    z0 = z_all.sum(dim=2)

    token_mask = None
    if return_token_mask:
        idx = indices_flat.view(B, M, num_q)
        token_mask = (idx != int(pad_id)).all(dim=2)
    return z0, token_mask


def _rvq_indices_to_batch_first_interleaved(indices_1d: torch.Tensor, B: int, num_q: int) -> torch.Tensor:
    indices_1d = indices_1d.long()
    num_q = max(1, int(num_q))
    if num_q == 1:
        return indices_1d.view(B, -1)

    N_flat = int(indices_1d.numel())
    base = int(B * num_q)
    if N_flat % base != 0:
        raise RuntimeError(f"RVQ indices length {N_flat} not divisible by B*num_quantizers={base}")
    M = N_flat // base

    idx = indices_1d.view(num_q, B, M)
    idx = idx.permute(1, 2, 0).contiguous()
    return idx.view(B, M * num_q)


@torch.no_grad()
def latent_to_indices_flat(
    core,
    z0: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
    pad_id: Optional[int] = None,
) -> torch.Tensor:
    """Quantize token-level latents back to flat indices sequence.

    z0: [B, M, D]
    returns indices_flat: [B, M*num_quantizers] or [B,M] for single-level.
    """
    q = getattr(core, "quantizer", None)
    if q is None:
        raise RuntimeError("model.quantizer not found")
    num_q = int(getattr(q, "num_quantizers", 1))
    B, M = int(z0.size(0)), int(z0.size(1))

    if token_mask is not None:
        if token_mask.shape != (B, M):
            raise ValueError(f"token_mask must be [B,M]={B,M}, got {token_mask.shape}")
        z0 = z0 * token_mask[:, :, None].to(z0.dtype)

    q_out = q(z0, do_ema_update=False, allow_reinit=False, mask=token_mask)
    if isinstance(q_out, (tuple, list)) and len(q_out) >= 3:
        idx_out = q_out[2]
    else:
        idx_out = q_out

    if torch.is_tensor(idx_out) and idx_out.dim() == 2:
        indices_flat = idx_out.long()
    elif torch.is_tensor(idx_out) and idx_out.dim() == 1:
        indices_flat = _rvq_indices_to_batch_first_interleaved(idx_out, B=B, num_q=num_q)
    else:
        raise RuntimeError("unsupported quantizer output")

    if pad_id is not None and token_mask is not None:
        num_q_eff = max(1, int(num_q))
        if num_q_eff == 1:
            indices_flat = indices_flat.masked_fill(~token_mask, int(pad_id))
        else:
            mask_flat = token_mask[:, :, None].expand(B, M, num_q_eff).reshape(B, M * num_q_eff)
            indices_flat = indices_flat.masked_fill(~mask_flat, int(pad_id))

    return indices_flat
