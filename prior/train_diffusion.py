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


def _normalize_zq(
    zq: torch.Tensor,
    zq_mean: Optional[torch.Tensor],
    zq_std: Optional[torch.Tensor],
) -> torch.Tensor:
    """Normalize z_q using precomputed mean/std if available.

    Args:
        zq: [B, M, D] latent vectors.
        zq_mean: [D] or None.
        zq_std: [D] or None.

    Returns:
        Normalized z_q with same shape as input.
    """
    if zq_mean is None or zq_std is None:
        return zq
    mean = zq_mean.view(1, 1, -1).to(zq.device)
    std = torch.clamp(zq_std.view(1, 1, -1).to(zq.device), min=1e-6)
    return (zq - mean) / std


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
    zq_mean: Optional[torch.Tensor],
    zq_std: Optional[torch.Tensor],
) -> float:
    """Evaluate diffusion loss on a validation set."""
    model.eval()
    losses = []
    for batch in loader:
        indices = batch["indices"].to(device)
        mask_flat = batch["mask"].to(device)

        z0, token_mask = indices_to_latent_sum(
            vq_codebook, indices, num_quantizers=num_q, pad_id=pad_id, return_token_mask=True
        )
        z0 = z0.to(device)
        if token_mask is not None:
            token_mask = token_mask.to(device)

        # Normalize z_q before feeding into diffusion
        z0_norm = _normalize_zq(z0, zq_mean, zq_std)

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

    if args.vq_ckpt:
        cfg_vq["ckpt"] = args.vq_ckpt
    if args.vq_yaml:
        cfg_vq["yaml"] = args.vq_yaml

    seed = int(cfg_runtime.get("seed", 42))
    set_seed(seed + rank)

    device = get_device(cfg_runtime)

    if rank == 0:
        print(f"[prior][diffusion] device={device} world={get_world_size()}")

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

    max_len = cfg_data.get("max_len", None)
    batch_size = int(cfg_data.get("batch_size", 64))
    num_workers = int(cfg_data.get("num_workers", 4))
    use_geo = bool(cfg_data.get("use_geo", False))
    geo_dim = int(cfg_data.get("geo_dim", 0))
    use_length_cond = bool(cfg_data.get("use_length_cond", True))
    max_target_len = int(cfg_data.get("max_target_len", 350))
    length_cond_dim = int(cfg_model.get("length_cond_dim", 1)) if use_length_cond else 0

    # Load z_q statistics if provided
    zq_stats_path = str(cfg_runtime.get("zq_stats_path", ""))
    zq_mean: Optional[torch.Tensor] = None
    zq_std: Optional[torch.Tensor] = None
    if zq_stats_path:
        stats = np.load(zq_stats_path)
        if "mean" not in stats or "std" not in stats:
            raise RuntimeError(
                f"zq_stats_path={zq_stats_path} must contain 'mean' and 'std' arrays"
            )
        zq_mean = torch.from_numpy(stats["mean"].astype(np.float32))
        zq_std = torch.from_numpy(stats["std"].astype(np.float32))
        if zq_mean.numel() != int(vq_info.code_dim):
            raise RuntimeError(
                f"zq_stats dim {zq_mean.numel()} != code_dim {vq_info.code_dim}"
            )
        if rank == 0:
            print(f"[zq_stats] loaded mean/std from {zq_stats_path}")

    train_ds = DiffusionIndexDataset(
        cfg_data["train_manifest"],
        pad_token_id=pad_id,
        max_len=max_len,
        load_geo=use_geo,
    )
    val_ds = None
    if cfg_data.get("val_manifest", ""):
        val_ds = DiffusionIndexDataset(
            cfg_data["val_manifest"],
            pad_token_id=pad_id,
            max_len=max_len,
            load_geo=use_geo,
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
            ),
            persistent_workers=bool(num_workers > 0),
        )

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
        }
        with (ckpt_dir / "train_meta.json").open("w") as f:
            json.dump(meta, f, indent=2)

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

        # Normalize z_q before feeding into diffusion
        z0_norm = _normalize_zq(z0, zq_mean, zq_std)

        geo_tok = None
        if use_geo and ("geo" in batch):
            geo = batch["geo"].to(device, non_blocking=True)
            geo_tok = _geo_to_token_cond(geo, num_q=num_q)

        length_cond = None
        if use_length_cond:
            target_len = batch["target_len"].to(device, non_blocking=True)
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
            loss = diffusion.training_loss(model, z0_norm, t=t, cond=cond, mask=token_mask)

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
                    zq_mean=zq_mean,
                    zq_std=zq_std,
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
                    zq_mean=zq_mean,
                    zq_std=zq_std,
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
