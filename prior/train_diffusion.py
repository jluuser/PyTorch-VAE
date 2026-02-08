#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import yaml

from prior.datasets.diffusion_dataset import DiffusionIndexDataset, collate_pad
from prior.diffusion.schedule import DiffusionSchedule
from prior.diffusion.gaussian_diffusion import GaussianDiffusion
from prior.models.diffusion_denoiser_resnet1d import (
    DenoiserConfig,
    DiffusionDenoiserResNet1D,
)
from prior.utils.ema import EMA
from prior.utils.vq_adapter import (
    load_vq_experiment,
    core_model,
    get_vq_info,
    indices_to_latent_sum,
)


# -----------------------------------------------------------------------------#
# Distributed helpers
# -----------------------------------------------------------------------------#


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def barrier():
    if is_dist():
        dist.barrier()


def init_dist(backend: str = "nccl"):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend=backend, rank=rank, world_size=world)


def set_seed(seed: int):
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(cfg_runtime: Dict[str, Any]) -> torch.device:
    device_str = str(cfg_runtime.get("device", "cuda"))
    if device_str.startswith("cuda") and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


# -----------------------------------------------------------------------------#
# Conditioning helpers
# -----------------------------------------------------------------------------#


@torch.no_grad()
def _geo_to_token_cond(geo_flat: torch.Tensor, num_q: int) -> torch.Tensor:
    """Convert per-flat-position geo features to per-token features.

    Args:
        geo_flat: [B, L_flat, G] per-position geometry.
        num_q: number of quantizers in RVQ.

    Returns:
        geo_tok: [B, M, G] per-token geometry, averaged over residual levels.
    """
    if geo_flat.dim() != 3:
        raise ValueError(f"geo must be [B,L,G], got {geo_flat.shape}")
    B, L, G = geo_flat.shape
    num_q = max(1, int(num_q))
    if num_q == 1:
        return geo_flat
    if L % num_q != 0:
        raise ValueError(f"geo length {L} not divisible by num_q={num_q}")
    M = L // num_q
    geo = geo_flat.view(B, M, num_q, G).mean(dim=2)
    return geo


@torch.no_grad()
def _build_length_condition(
    target_len: torch.Tensor,
    num_tokens: int,
    max_target_len: int,
    cond_dim: int,
) -> Optional[torch.Tensor]:
    """Build per-token length conditioning tensor.

    Args:
        target_len: [B] integer sequence lengths in original curve space.
        num_tokens: M, latent token count.
        max_target_len: global max length used for normalization.
        cond_dim: feature dimension for length condition.

    Returns:
        Tensor of shape [B, M, cond_dim] or None if cond_dim <= 0.
    """
    cond_dim = int(cond_dim)
    if cond_dim <= 0:
        return None

    max_len = float(max(1, int(max_target_len)))
    len_clamped = target_len.clamp(min=1)
    len_norm = (len_clamped.float() / max_len).unsqueeze(-1)  # [B, 1]

    if cond_dim == 1:
        len_feat = len_norm  # [B, 1]
    else:
        len_feat = len_norm.repeat(1, cond_dim)  # [B, cond_dim]

    cond = len_feat.unsqueeze(1).expand(-1, num_tokens, -1)  # [B, M, cond_dim]
    return cond


# -----------------------------------------------------------------------------#
# Optim / LR helpers
# -----------------------------------------------------------------------------#


def build_optimizer(model: nn.Module, cfg_optim: Dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(cfg_optim.get("lr", 2e-4))
    betas = cfg_optim.get("betas", [0.9, 0.99])
    wd = float(cfg_optim.get("weight_decay", 0.0))
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=tuple(betas), weight_decay=wd)


def lr_warmup_cosine(step: int, warmup: int, total: int, base_lr: float) -> float:
    step = int(step)
    warmup = max(0, int(warmup))
    total = max(1, int(total))
    if warmup > 0 and step < warmup:
        return base_lr * float(step + 1) / float(warmup)
    progress = min(1.0, max(0.0, float(step - warmup) / float(max(1, total - warmup))))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_ckpt(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: Dict[str, Any],
    ema: Optional[EMA] = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj: Dict[str, Any] = {
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
    }
    if ema is not None:
        obj["ema"] = ema.state_dict()
    torch.save(obj, str(path))


def load_ckpt(path: str) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        return obj
    raise RuntimeError(f"invalid ckpt: {path}")


# -----------------------------------------------------------------------------#
# Latent normalization helpers
# -----------------------------------------------------------------------------#


def _normalize_latent(
    z: torch.Tensor,
    latent_mean: Optional[torch.Tensor],
    latent_std: Optional[torch.Tensor],
) -> torch.Tensor:
    """Normalize latent vectors using precomputed mean/std if available.

    Args:
        z: [B, M, D] latent vectors (z_q or z_e).
        latent_mean: [D] or None.
        latent_std: [D] or None.

    Returns:
        Normalized latent with same shape as input.
    """
    if latent_mean is None or latent_std is None:
        return z
    mean = latent_mean.view(1, 1, -1).to(z.device)
    std = torch.clamp(latent_std.view(1, 1, -1).to(z.device), min=1e-6)
    return (z - mean) / std


def _denormalize_latent(
    z_norm: torch.Tensor,
    latent_mean: Optional[torch.Tensor],
    latent_std: Optional[torch.Tensor],
) -> torch.Tensor:
    """Invert normalization of latent vectors using precomputed mean/std if available.

    Args:
        z_norm: [B, M, D] normalized latent vectors.
        latent_mean: [D] or None.
        latent_std: [D] or None.

    Returns:
        Denormalized latent with same shape as input.
    """
    if latent_mean is None or latent_std is None:
        return z_norm
    mean = latent_mean.view(1, 1, -1).to(z_norm.device)
    std = torch.clamp(latent_std.view(1, 1, -1).to(z_norm.device), min=1e-6)
    return z_norm * std + mean


# -----------------------------------------------------------------------------#
# Geometry regularization helpers
# -----------------------------------------------------------------------------#


def _collision_loss_single(
    coords: torch.Tensor,
    min_pairwise_dist: float,
    neighbor_exclude: int,
    max_token_samples: int,
    max_pairs_per_token: int,
) -> torch.Tensor:
    """Approximate self-collision loss for a single curve.

    We sample a subset of positions i and, for each i, a subset of positions j
    that are not close in sequence. Distances below min_pairwise_dist are
    penalized with a hinge loss.
    """
    device = coords.device
    L = int(coords.size(0))
    if L <= neighbor_exclude + 1:
        return coords.new_tensor(0.0)

    max_token_samples = int(max_token_samples)
    max_pairs_per_token = int(max_pairs_per_token)
    if max_token_samples <= 0 or max_pairs_per_token <= 0:
        return coords.new_tensor(0.0)

    K = min(L, max_token_samples)
    # Sample token indices i and partner indices j
    i_idx = torch.randint(0, L, (K,), device=device)
    J = max_pairs_per_token
    j_idx = torch.randint(0, L, (K, J), device=device)

    # Filter out near neighbors in sequence
    i_expand = i_idx.view(K, 1).expand(K, J)
    valid = (torch.abs(j_idx - i_expand) > int(neighbor_exclude))
    if not valid.any():
        return coords.new_tensor(0.0)

    i_valid = i_expand[valid]
    j_valid = j_idx[valid]
    if i_valid.numel() == 0:
        return coords.new_tensor(0.0)

    pts_i = coords[i_valid]  # [P, 3]
    pts_j = coords[j_valid]  # [P, 3]
    dists = torch.norm(pts_i - pts_j, dim=-1)  # [P]

    margin = F.relu(float(min_pairwise_dist) - dists)
    if margin.numel() == 0:
        return coords.new_tensor(0.0)
    return margin.mean()


def _bond_length_loss_single(
    coords: torch.Tensor,
    target_min: float,
    target_max: float,
) -> torch.Tensor:
    L = int(coords.size(0))
    if L < 2:
        return coords.new_tensor(0.0)

    diffs = coords[1:] - coords[:-1]  # [L-1, 3]
    dists = torch.norm(diffs, dim=-1)  # [L-1]

    loss_low = F.relu(float(target_min) - dists)
    loss_high = F.relu(dists - float(target_max))
    loss = loss_low + loss_high
    if loss.numel() == 0:
        return coords.new_tensor(0.0)
    return loss.mean()


def _bond_angle_loss_single(
    coords: torch.Tensor,
    min_deg: float,
    max_deg: float,
) -> torch.Tensor:
    L = int(coords.size(0))
    if L < 3:
        return coords.new_tensor(0.0)

    p0 = coords[:-2]      # [L-2, 3]
    p1 = coords[1:-1]     # [L-2, 3]
    p2 = coords[2:]       # [L-2, 3]

    v1 = p0 - p1          # [L-2, 3]
    v2 = p2 - p1          # [L-2, 3]

    v1_norm = torch.norm(v1, dim=-1)
    v2_norm = torch.norm(v2, dim=-1)
    denom = v1_norm * v2_norm

    mask = denom > 1e-6
    if not mask.any():
        return coords.new_tensor(0.0)

    cos_theta = (v1[mask] * v2[mask]).sum(dim=-1) / denom[mask]
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    angles = torch.acos(cos_theta) * (180.0 / math.pi)

    loss_low = F.relu(float(min_deg) - angles)
    loss_high = F.relu(angles - float(max_deg))
    loss = loss_low + loss_high
    if loss.numel() == 0:
        return coords.new_tensor(0.0)
    return loss.mean()


def _geometry_losses_from_coords(
    coords_batch: torch.Tensor,
    lengths: torch.Tensor,
    geom_cfg: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute collision / bond-length / angle losses over a batch of curves."""
    device = coords_batch.device
    B, L_max, _ = coords_batch.shape

    min_pairwise_dist = float(geom_cfg.get("min_pairwise_dist", 2.0))
    neighbor_exclude = int(geom_cfg.get("neighbor_exclude", 2))
    max_token_samples = int(geom_cfg.get("max_token_samples", 32))
    max_pairs_per_token = int(geom_cfg.get("max_pairs_per_token", 4))

    bond_target_min = float(geom_cfg.get("bond_target_min", 3.5))
    bond_target_max = float(geom_cfg.get("bond_target_max", 4.2))
    angle_min_deg = float(geom_cfg.get("angle_min_deg", 80.0))
    angle_max_deg = float(geom_cfg.get("angle_max_deg", 150.0))

    coll_losses = []
    bond_losses = []
    angle_losses = []

    for b in range(B):
        L_i = int(lengths[b].item())
        if L_i <= 1:
            continue
        c = coords_batch[b, :L_i]  # [L_i, 3]

        if min_pairwise_dist > 0.0 and max_token_samples > 0 and max_pairs_per_token > 0:
            coll_losses.append(
                _collision_loss_single(
                    c,
                    min_pairwise_dist=min_pairwise_dist,
                    neighbor_exclude=neighbor_exclude,
                    max_token_samples=max_token_samples,
                    max_pairs_per_token=max_pairs_per_token,
                )
            )

        if bond_target_max > 0.0:
            bond_losses.append(
                _bond_length_loss_single(
                    c,
                    target_min=bond_target_min,
                    target_max=bond_target_max,
                )
            )

        if angle_max_deg > 0.0 and L_i >= 3:
            angle_losses.append(
                _bond_angle_loss_single(
                    c,
                    min_deg=angle_min_deg,
                    max_deg=angle_max_deg,
                )
            )

    def _mean_or_zero(lst):
        if not lst:
            return coords_batch.new_tensor(0.0, device=device)
        return torch.stack(lst, dim=0).mean()

    loss_coll = _mean_or_zero(coll_losses)
    loss_bond = _mean_or_zero(bond_losses)
    loss_angle = _mean_or_zero(angle_losses)
    return loss_coll, loss_bond, loss_angle


def _training_loss_with_geom(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    vq_core: nn.Module,
    z0_norm: torch.Tensor,
    t: torch.Tensor,
    cond: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
    target_len: torch.Tensor,
    latent_mean: Optional[torch.Tensor],
    latent_std: Optional[torch.Tensor],
    geom_cfg: Dict[str, Any],
    step: int,
) -> torch.Tensor:
    """Diffusion training loss with additional geometry regularization.

    This mirrors GaussianDiffusion.training_loss (x-prediction on x0),
    and then decodes the predicted x0 through the frozen VQ-VAE decoder
    to apply collision / bond / angle penalties in coordinate space.
    """
    # ----- Base diffusion x0-pred loss -----
    noise = torch.randn_like(z0_norm)
    if mask is not None:
        noise = noise * mask[:, :, None].to(noise.dtype)

    x_t = diffusion.q_sample(z0_norm, t, noise=noise)
    x0_pred = model(x_t, t, cond=cond, mask=mask)

    if mask is not None:
        loss = (x0_pred - z0_norm) ** 2
        loss = loss * mask[:, :, None].to(loss.dtype)
        denom = torch.clamp(mask.sum() * z0_norm.size(-1), min=1.0)
        loss = loss.sum() / denom
    else:
        loss = F.mse_loss(x0_pred, z0_norm)

    # ----- Geometry regularization (optional) -----
    lambda_collision = float(geom_cfg.get("lambda_collision", 0.0))
    lambda_bond = float(geom_cfg.get("lambda_bond", 0.0))
    lambda_angle = float(geom_cfg.get("lambda_angle", 0.0))
    geom_step_interval = int(geom_cfg.get("step_interval", 1))

    if (
        (lambda_collision > 0.0 or lambda_bond > 0.0 or lambda_angle > 0.0)
        and geom_step_interval > 0
        and (step % geom_step_interval == 0)
    ):
        # Decode only a subset of the batch for efficiency
        z0_pred = _denormalize_latent(x0_pred, latent_mean, latent_std)
        B = int(z0_pred.size(0))
        max_geom_batch = int(geom_cfg.get("max_batch", B))
        B_geom = min(B, max_geom_batch)
        if B_geom > 0:
            z_sel = z0_pred[:B_geom]               # [B_g, M, D]
            len_sel = target_len[:B_geom]          # [B_g]
            L_max = int(len_sel.max().item())
            if L_max > 0:
                decode_mask = torch.zeros(
                    B_geom, L_max, dtype=torch.bool, device=z_sel.device
                )
                for i in range(B_geom):
                    L_i = int(len_sel[i].item())
                    if L_i > 0:
                        decode_mask[i, :L_i] = True

                # VQ-VAE decoder returns [B_g, L_max, C]; use xyz channels only
                coords_pred = vq_core.decode(z_sel, mask=decode_mask)[..., :3]

                loss_coll, loss_bond, loss_angle = _geometry_losses_from_coords(
                    coords_pred,
                    len_sel,
                    geom_cfg,
                )

                loss = (
                    loss
                    + lambda_collision * loss_coll
                    + lambda_bond * loss_bond
                    + lambda_angle * loss_angle
                )

    return loss


# -----------------------------------------------------------------------------#
# Validation loop
# -----------------------------------------------------------------------------#


@torch.no_grad()
def evaluate(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    vq_codebook: torch.Tensor,
    num_q: int,
    pad_id: int,
    loader: DataLoader,
    device: torch.device,
    use_geo: bool,
    use_length_cond: bool,
    max_target_len: int,
    length_cond_dim: int,
    latent_mean: Optional[torch.Tensor],
    latent_std: Optional[torch.Tensor],
    use_ze_latent: bool,
) -> float:
    """Evaluate diffusion loss on a validation set."""
    model.eval()
    losses = []
    for batch in loader:
        if use_ze_latent:
            # latent mode: directly use z_e from dataset
            z0 = batch["latent"].to(device)          # [B, M, D]
            token_mask = batch["mask"].to(device)    # [B, M]
        else:
            # indices mode: convert indices to z_q via RVQ
            indices = batch["indices"].to(device)
            mask_flat = batch["mask"].to(device)
            z0, token_mask = indices_to_latent_sum(
                vq_codebook, indices, num_quantizers=num_q, pad_id=pad_id, return_token_mask=True
            )
            z0 = z0.to(device)
            if token_mask is not None:
                token_mask = token_mask.to(device)

        # Normalize latent before feeding into diffusion
        z0_norm = _normalize_latent(z0, latent_mean, latent_std)

        geo_tok = None
        if use_geo and ("geo" in batch):
            geo = batch["geo"].to(device)
            geo_tok = _geo_to_token_cond(geo, num_q=num_q)

        length_cond = None
        if use_length_cond:
            target_len = batch["target_len"].to(device)
            length_cond = _build_length_condition(
                target_len,
                num_tokens=z0.size(1),
                max_target_len=max_target_len,
                cond_dim=length_cond_dim,
            )

        cond_parts = []
        if geo_tok is not None:
            cond_parts.append(geo_tok)
        if length_cond is not None:
            cond_parts.append(length_cond)
        cond = torch.cat(cond_parts, dim=-1) if cond_parts else None

        t = torch.randint(0, diffusion.num_steps, (z0.size(0),), device=device, dtype=torch.long)
        loss = diffusion.training_loss(model, z0_norm, t=t, cond=cond, mask=token_mask)
        losses.append(float(loss.item()))

    model.train()
    if not losses:
        return 0.0
    return sum(losses) / float(len(losses))


# -----------------------------------------------------------------------------#
# Main training entry
# -----------------------------------------------------------------------------#


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "diffusion_prior.yaml"),
    )
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--vq_ckpt", type=str, default="")
    ap.add_argument("--vq_yaml", type=str, default="")
    args = ap.parse_args()

    init_dist("nccl")
    rank = get_rank()

    cfg = load_yaml(args.config)
    cfg_data = cfg.get("data", {})
    cfg_vq = cfg.get("vq", {})
    cfg_diff = cfg.get("diffusion", {})
    cfg_model = cfg.get("model", {})
    cfg_optim = cfg.get("optim", {})
    cfg_ema = cfg.get("ema", {})
    cfg_runtime = cfg.get("runtime", {})
    cfg_geom = cfg.get("geom", {})

    if args.vq_ckpt:
        cfg_vq["ckpt"] = args.vq_ckpt
    if args.vq_yaml:
        cfg_vq["yaml"] = args.vq_yaml

    latent_type = str(cfg_data.get("latent_type", "indices")).lower()
    use_ze_latent = latent_type in ("ze", "latent", "continuous")

    seed = int(cfg_runtime.get("seed", 42))
    set_seed(seed + rank)

    device = get_device(cfg_runtime)

    if rank == 0:
        print(f"[prior][diffusion] device={device} world={get_world_size()} latent_type={latent_type}")

    # ----- Load VQ-VAE (frozen) -----
    vq_ckpt = str(cfg_vq.get("ckpt", ""))
    vq_yaml = str(cfg_vq.get("yaml", ""))
    if not vq_ckpt or not vq_yaml:
        raise RuntimeError("config.vq must include ckpt and yaml")

    vq_exp = load_vq_experiment(vq_ckpt, vq_yaml, device)
    vq_core = core_model(vq_exp)
    vq_info = get_vq_info(vq_core)
    vq_codebook = vq_info.codebook.to(device)
    num_q = int(cfg_data.get("num_quantizers", vq_info.num_quantizers))
    if num_q != vq_info.num_quantizers and rank == 0:
        print(
            f"[warn] config num_quantizers={num_q} != vq num_quantizers={vq_info.num_quantizers}, using vq"
        )
    num_q = int(vq_info.num_quantizers)
    pad_id = int(cfg_data.get("pad_token_id", vq_info.K_total))

    # Freeze VQ-VAE core; used only for decoding in geometry regularization
    vq_core.eval()
    for p in vq_core.parameters():
        p.requires_grad_(False)

    max_len = cfg_data.get("max_len", None)
    batch_size = int(cfg_data.get("batch_size", 64))
    num_workers = int(cfg_data.get("num_workers", 4))
    use_geo = bool(cfg_data.get("use_geo", False))
    geo_dim = int(cfg_data.get("geo_dim", 0))
    use_length_cond = bool(cfg_data.get("use_length_cond", True))
    max_target_len = int(cfg_data.get("max_target_len", 350))
    length_cond_dim = int(cfg_model.get("length_cond_dim", 1)) if use_length_cond else 0

    use_geom_loss = bool(cfg_geom.get("enable", False))

    # Load latent statistics if provided (z_e or z_q)
    latent_stats_path = str(
        cfg_runtime.get("latent_stats_path", cfg_runtime.get("zq_stats_path", ""))
    )
    latent_mean: Optional[torch.Tensor] = None
    latent_std: Optional[torch.Tensor] = None
    if latent_stats_path:
        stats = np.load(latent_stats_path)
        if "mean" not in stats or "std" not in stats:
            raise RuntimeError(
                f"latent_stats_path={latent_stats_path} must contain 'mean' and 'std' arrays"
            )
        latent_mean = torch.from_numpy(stats["mean"].astype(np.float32))
        latent_std = torch.from_numpy(stats["std"].astype(np.float32))
        if latent_mean.numel() != int(vq_info.code_dim):
            raise RuntimeError(
                f"latent_stats dim {latent_mean.numel()} != code_dim {vq_info.code_dim}"
            )
        if rank == 0:
            print(f"[latent_stats] loaded mean/std from {latent_stats_path}")

    # ----- Datasets -----
    train_ds = DiffusionIndexDataset(
        cfg_data["train_manifest"],
        pad_token_id=pad_id,
        max_len=max_len,
        load_geo=use_geo,
        use_latent=use_ze_latent,
    )
    val_ds = None
    if cfg_data.get("val_manifest", ""):
        val_ds = DiffusionIndexDataset(
            cfg_data["val_manifest"],
            pad_token_id=pad_id,
            max_len=max_len,
            load_geo=use_geo,
            use_latent=use_ze_latent,
        )

    sampler = None
    if is_dist():
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            train_ds,
            num_replicas=get_world_size(),
            rank=rank,
            shuffle=True,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(cfg_data.get("pin_memory", True)),
        drop_last=False,
        collate_fn=lambda b: collate_pad(
            b,
            pad_id=pad_id,
            geo_dim=geo_dim,
            multiple_of=num_q,
            use_latent=use_ze_latent,
        ),
        persistent_workers=bool(num_workers > 0),
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=max(1, batch_size // 2),
            shuffle=False,
            num_workers=max(0, num_workers // 2),
            pin_memory=bool(cfg_data.get("pin_memory", True)),
            drop_last=False,
            collate_fn=lambda b: collate_pad(
                b,
                pad_id=pad_id,
                geo_dim=geo_dim,
                multiple_of=num_q,
                use_latent=use_ze_latent,
            ),
            persistent_workers=bool(num_workers > 0),
        )

    # ----- Diffusion + denoiser -----
    schedule = DiffusionSchedule.build(
        num_steps=int(cfg_diff.get("num_steps", 1000)),
        schedule=str(cfg_diff.get("schedule", "cosine")),
        beta_start=float(cfg_diff.get("beta_start", 1e-4)),
        beta_end=float(cfg_diff.get("beta_end", 2e-2)),
        cosine_s=float(cfg_diff.get("cosine_s", 0.008)),
    )
    diffusion = GaussianDiffusion(betas=schedule.betas).to(device)

    cond_dim = (geo_dim if use_geo else 0) + length_cond_dim
    den_cfg = DenoiserConfig(
        code_dim=int(vq_info.code_dim),
        hidden_channels=int(cfg_model.get("hidden_channels", 256)),
        time_embed_dim=int(cfg_model.get("time_embed_dim", 256)),
        num_blocks=int(cfg_model.get("num_blocks", 12)),
        dropout=float(cfg_model.get("dropout", 0.0)),
        cond_dim=int(cond_dim),
    )
    model = DiffusionDenoiserResNet1D(den_cfg).to(device)

    if is_dist():
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    optimizer = build_optimizer(model, cfg_optim)

    use_amp = bool(cfg_runtime.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ema = None
    if bool(cfg_ema.get("enable", True)):
        ema = EMA(decay=float(cfg_ema.get("decay", 0.999)))
        ema.register(model.module if isinstance(model, DDP) else model)

    # ----- Resume -----
    start_step = 0
    if args.resume:
        ckpt = load_ckpt(args.resume)
        (model.module if isinstance(model, DDP) else model).load_state_dict(
            ckpt["model"], strict=True
        )
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        if ema is not None and ("ema" in ckpt):
            ema.load_state_dict(ckpt["ema"], device=device)
        if rank == 0:
            print(f"[resume] step={start_step} from {args.resume}")

    max_updates = int(cfg_optim.get("max_updates", 80000))
    warmup = int(cfg_optim.get("warmup_updates", 2000))
    grad_clip = float(cfg_optim.get("grad_clip_norm", 1.0))
    base_lr = float(cfg_optim.get("lr", 2e-4))

    log_interval = int(cfg_runtime.get("log_interval", 100))
    eval_interval = int(cfg_runtime.get("eval_interval_updates", 2000))
    save_interval = int(cfg_runtime.get("save_interval_updates", 10000))
    ckpt_dir = Path(str(cfg_runtime.get("ckpt_dir", "prior/prior_ckpts/diffusion_prior")))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        meta = {
            "vq_ckpt": vq_ckpt,
            "vq_yaml": vq_yaml,
            "K_total": int(vq_info.K_total),
            "code_dim": int(vq_info.code_dim),
            "num_quantizers": int(num_q),
            "pad_token_id": int(pad_id),
            "latent_type": latent_type,
        }
        with (ckpt_dir / "train_meta.json").open("w") as f:
            json.dump(meta, f, indent=2)

    # ----- Training loop -----
    model.train()
    step = start_step
    it = iter(train_loader)

    while step < max_updates:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(step // max(1, len(train_loader)))

        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        # Build latent z0 and token mask according to latent_type
        if use_ze_latent:
            z0 = batch["latent"].to(device, non_blocking=True)   # [B, M, D]
            token_mask = batch["mask"].to(device, non_blocking=True)  # [B, M]
        else:
            indices = batch["indices"].to(device, non_blocking=True)
            mask_flat = batch["mask"].to(device, non_blocking=True)
            z0, token_mask = indices_to_latent_sum(
                vq_codebook,
                indices,
                num_quantizers=num_q,
                pad_id=pad_id,
                return_token_mask=True,
            )
            z0 = z0.to(device)
            if token_mask is not None:
                token_mask = token_mask.to(device)

        # Normalize latent before feeding into diffusion
        z0_norm = _normalize_latent(z0, latent_mean, latent_std)

        geo_tok = None
        if use_geo and ("geo" in batch):
            geo = batch["geo"].to(device, non_blocking=True)
            geo_tok = _geo_to_token_cond(geo, num_q=num_q)

        # Always load target_len (needed for geometry); length_cond is optional
        target_len = batch["target_len"].to(device, non_blocking=True)

        length_cond = None
        if use_length_cond:
            length_cond = _build_length_condition(
                target_len,
                num_tokens=z0.size(1),
                max_target_len=max_target_len,
                cond_dim=length_cond_dim,
            )

        cond_parts = []
        if geo_tok is not None:
            cond_parts.append(geo_tok)
        if length_cond is not None:
            cond_parts.append(length_cond)
        cond = torch.cat(cond_parts, dim=-1) if cond_parts else None

        t = torch.randint(0, diffusion.num_steps, (z0.size(0),), device=device, dtype=torch.long)

        lr = lr_warmup_cosine(step, warmup=warmup, total=max_updates, base_lr=base_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if use_geom_loss:
                loss = _training_loss_with_geom(
                    model,
                    diffusion,
                    vq_core,
                    z0_norm,
                    t,
                    cond,
                    token_mask,
                    target_len,
                    latent_mean,
                    latent_std,
                    cfg_geom,
                    step=step,
                )
            else:
                loss = diffusion.training_loss(
                    model, z0_norm, t=t, cond=cond, mask=token_mask
                )

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (model.module if isinstance(model, DDP) else model).parameters(),
                grad_clip,
            )

        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model.module if isinstance(model, DDP) else model)

        if rank == 0 and (step % log_interval == 0):
            print(f"step {step} | loss {loss.item():.6f} | lr {lr:.6e}")

        if rank == 0 and (save_interval > 0) and (step > 0) and (step % save_interval == 0):
            ckpt_path = ckpt_dir / f"diffusion_prior_step{step}.pt"
            to_save = model.module if isinstance(model, DDP) else model
            save_ckpt(ckpt_path, to_save, optimizer, step, cfg, ema=ema)

        if rank == 0 and val_loader is not None and (eval_interval > 0) and (
            step > 0
        ) and (step % eval_interval == 0):
            to_eval = model.module if isinstance(model, DDP) else model
            if ema is not None:
                backup = {k: v.detach().clone() for k, v in to_eval.state_dict().items()}
                ema.copy_to(to_eval)
                val_loss = evaluate(
                    to_eval,
                    diffusion,
                    vq_codebook,
                    num_q,
                    pad_id,
                    val_loader,
                    device,
                    use_geo=use_geo,
                    use_length_cond=use_length_cond,
                    max_target_len=max_target_len,
                    length_cond_dim=length_cond_dim,
                    latent_mean=latent_mean,
                    latent_std=latent_std,
                    use_ze_latent=use_ze_latent,
                )
                to_eval.load_state_dict(backup, strict=True)
            else:
                val_loss = evaluate(
                    to_eval,
                    diffusion,
                    vq_codebook,
                    num_q,
                    pad_id,
                    val_loader,
                    device,
                    use_geo=use_geo,
                    use_length_cond=use_length_cond,
                    max_target_len=max_target_len,
                    length_cond_dim=length_cond_dim,
                    latent_mean=latent_mean,
                    latent_std=latent_std,
                    use_ze_latent=use_ze_latent,
                )
            print(f"[val] step {step} | loss {val_loss:.6f}")

        step += 1

    if rank == 0:
        final_path = ckpt_dir / f"diffusion_prior_final_step{step}.pt"
        to_save = model.module if isinstance(model, DDP) else model
        save_ckpt(final_path, to_save, optimizer, step, cfg, ema=ema)

    barrier()


if __name__ == "__main__":
    main()
