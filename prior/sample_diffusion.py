#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

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
    latent_to_indices_flat,
    indices_to_latent_sum,
)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(device_str: str) -> torch.device:
    device_str = str(device_str)
    if device_str.startswith("cuda") and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


class LengthSampler:
    """Sample target lengths from an existing manifest."""

    def __init__(self, manifest_path: str, key: str = "target_len"):
        self.lengths: List[int] = []
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                v = rec.get(key, rec.get("length", 0))
                try:
                    v = int(v)
                except Exception:
                    v = 0
                if v > 0:
                    self.lengths.append(v)
        if not self.lengths:
            # Fallback length if manifest is empty or invalid
            self.lengths = [128]

    def sample(self, n: int) -> List[int]:
        idx = np.random.randint(0, len(self.lengths), size=int(n))
        return [self.lengths[i] for i in idx]


def build_length_condition_from_list(
    target_lens: List[int],
    num_tokens: int,
    max_target_len: int,
    cond_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Build per-token length condition from a Python list of lengths.

    Args:
        target_lens: list of ints, length B.
        num_tokens: M, latent token count.
        max_target_len: global max length for normalization.
        cond_dim: feature dimension of length condition.
        device: torch device.

    Returns:
        Tensor of shape [B, M, cond_dim].
    """
    cond_dim = int(cond_dim)
    if cond_dim <= 0:
        raise ValueError("cond_dim must be > 0 to build length condition")

    B = len(target_lens)
    max_len = float(max(1, int(max_target_len)))
    lengths = torch.tensor(target_lens, device=device, dtype=torch.long)
    len_clamped = lengths.clamp(min=1)
    len_norm = (len_clamped.float() / max_len).unsqueeze(-1)  # [B, 1]

    if cond_dim == 1:
        len_feat = len_norm  # [B, 1]
    else:
        len_feat = len_norm.repeat(1, cond_dim)  # [B, cond_dim]

    cond = len_feat.unsqueeze(1).expand(B, num_tokens, cond_dim)  # [B, M, cond_dim]
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="diffusion prior ckpt")
    ap.add_argument("--config", type=str, default="", help="diffusion prior yaml (optional)")
    ap.add_argument("--vq_ckpt", type=str, default="")
    ap.add_argument("--vq_yaml", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    # Increased default sample steps for better quality
    ap.add_argument("--sample_steps", type=int, default=250)
    ap.add_argument("--latent_tokens", type=int, default=0, help="override M tokens (optional)")
    ap.add_argument(
        "--target_len",
        type=int,
        default=0,
        help="fixed target length (optional, 0 = sample from train manifest)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    # Optional clipping in normalized latent space
    ap.add_argument(
        "--clip_norm",
        type=float,
        default=0.0,
        help="if > 0, clamp normalized latent to [-clip_norm, clip_norm]",
    )
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latent_dir = out_dir / "latent_npy"
    latent_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = out_dir / "curves_npy"
    curves_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load diffusion checkpoint and config
    # -------------------------------------------------------------------------
    ckpt_obj = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt_obj.get("config", {})
    if args.config:
        # Allow overriding config from an external yaml if desired
        cfg = load_yaml(args.config)

    cfg_data = cfg.get("data", {})
    cfg_vq = cfg.get("vq", {})
    cfg_diff = cfg.get("diffusion", {})
    cfg_model = cfg.get("model", {})
    cfg_runtime = cfg.get("runtime", {})

    if args.vq_ckpt:
        cfg_vq["ckpt"] = args.vq_ckpt
    if args.vq_yaml:
        cfg_vq["yaml"] = args.vq_yaml

    vq_ckpt = str(cfg_vq.get("ckpt", ""))
    vq_yaml = str(cfg_vq.get("yaml", ""))
    if not vq_ckpt or not vq_yaml:
        raise RuntimeError("vq ckpt/yaml not provided")

    # -------------------------------------------------------------------------
    # Load VQ-VAE experiment and RVQ info
    # -------------------------------------------------------------------------
    vq_exp = load_vq_experiment(vq_ckpt, vq_yaml, device)
    vq_core = core_model(vq_exp).to(device)
    vq_core.eval()
    vq_info = get_vq_info(vq_core)
    num_q = int(vq_info.num_quantizers)

    # Pad token id used by RVQ utilities (for indices_to_latent_sum)
    pad_id = int(cfg_data.get("pad_token_id", int(vq_info.K_total)))

    # -------------------------------------------------------------------------
    # Data-level hyperparameters
    # -------------------------------------------------------------------------
    use_geo = bool(cfg_data.get("use_geo", False))
    geo_dim = int(cfg_data.get("geo_dim", 0))
    use_length_cond = bool(cfg_data.get("use_length_cond", True))
    max_target_len = int(cfg_data.get("max_target_len", 350))
    length_cond_dim = int(cfg_model.get("length_cond_dim", 1)) if use_length_cond else 0

    # -------------------------------------------------------------------------
    # Load latent statistics for normalization (z_e or z_q)
    # -------------------------------------------------------------------------
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
        latent_mean = torch.from_numpy(stats["mean"].astype(np.float32)).to(device)
        latent_std = torch.from_numpy(stats["std"].astype(np.float32)).to(device)
        if latent_mean.numel() != int(vq_info.code_dim):
            raise RuntimeError(
                f"latent_stats dim {latent_mean.numel()} != code_dim {vq_info.code_dim}"
            )
        print(f"[latent_stats] loaded mean/std from {latent_stats_path}")

    # -------------------------------------------------------------------------
    # Build diffusion schedule and GaussianDiffusion wrapper
    # -------------------------------------------------------------------------
    schedule = DiffusionSchedule.build(
        num_steps=int(cfg_diff.get("num_steps", 1000)),
        schedule=str(cfg_diff.get("schedule", "cosine")),
        beta_start=float(cfg_diff.get("beta_start", 1e-4)),
        beta_end=float(cfg_diff.get("beta_end", 2e-2)),
        cosine_s=float(cfg_diff.get("cosine_s", 0.008)),
    )
    diffusion = GaussianDiffusion(betas=schedule.betas).to(device)

    # -------------------------------------------------------------------------
    # Build denoiser model with the same cond_dim as in training
    # -------------------------------------------------------------------------
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
    model.load_state_dict(ckpt_obj["model"], strict=True)
    model.eval()

    # Optional EMA
    if "ema" in ckpt_obj and bool(cfg.get("ema", {}).get("enable", True)):
        ema = EMA(decay=0.999)
        ema.load_state_dict(ckpt_obj["ema"], device=device)
        ema.copy_to(model)
        print("[ema] EMA weights loaded and copied to model")

    # -------------------------------------------------------------------------
    # Determine latent token count M
    # -------------------------------------------------------------------------
    M = int(args.latent_tokens) if int(args.latent_tokens) > 0 else 0
    if M <= 0:
        max_len_cfg = int(cfg_data.get("max_len", 0))
        if max_len_cfg > 0 and max_len_cfg % num_q == 0:
            M = max_len_cfg // num_q
        else:
            raise RuntimeError(
                "latent_tokens not provided and cannot infer from config.data.max_len"
            )

    # -------------------------------------------------------------------------
    # Optional length sampler from training manifest
    # -------------------------------------------------------------------------
    length_sampler = None
    train_manifest = str(cfg_data.get("train_manifest", ""))
    if train_manifest:
        length_sampler = LengthSampler(train_manifest)

    manifest_path = out_dir / "samples_manifest.jsonl"
    f_manifest = manifest_path.open("w")

    remaining = int(args.num_samples)
    sample_id = 0

    # Pre-cache codebook on the correct device
    codebook = vq_info.codebook.to(device)

    while remaining > 0:
        bs = min(int(args.batch_size), remaining)

        # ---------------------------------------------------------------------
        # Decide target length for each sample in this batch
        # ---------------------------------------------------------------------
        if int(args.target_len) > 0:
            # Fixed length for all samples in this batch
            target_lens = [int(args.target_len)] * bs
        elif length_sampler is not None:
            # Sample lengths from train manifest distribution
            target_lens = length_sampler.sample(bs)
        else:
            # Fallback default
            target_lens = [128] * bs

        # ---------------------------------------------------------------------
        # Build length condition if enabled
        # ---------------------------------------------------------------------
        length_cond = None
        if use_length_cond and length_cond_dim > 0:
            length_cond = build_length_condition_from_list(
                target_lens=target_lens,
                num_tokens=M,
                max_target_len=max_target_len,
                cond_dim=length_cond_dim,
                device=device,
            )

        # Build geo condition for sampling.
        # For now, we set geo condition to zeros if geo is enabled.
        geo_cond = None
        if use_geo and geo_dim > 0:
            geo_cond = torch.zeros(bs, M, geo_dim, device=device, dtype=torch.float32)

        # Concatenate geo and length conditions to match training cond_dim
        if geo_cond is None and length_cond is None:
            cond = None
        elif geo_cond is None:
            cond = length_cond
        elif length_cond is None:
            cond = geo_cond
        else:
            cond = torch.cat([geo_cond, length_cond], dim=-1)

        # ---------------------------------------------------------------------
        # Sample in normalized latent space using DDIM
        # ---------------------------------------------------------------------
        with torch.no_grad():
            z0_norm = diffusion.ddim_sample_loop(
                model,
                shape=(bs, M, int(vq_info.code_dim)),
                steps=int(args.sample_steps),
                device=device,
                cond=cond,
                mask=None,
            )  # [B, M, D]

        # Optional clipping in normalized space to avoid extreme values
        if float(args.clip_norm) > 0.0:
            clip_val = float(args.clip_norm)
            z0_norm = torch.clamp(z0_norm, -clip_val, clip_val)

        # ---------------------------------------------------------------------
        # Denormalize back to original latent space using stats (z_e or z_q)
        # ---------------------------------------------------------------------
        if latent_mean is not None and latent_std is not None:
            mean = latent_mean.view(1, 1, -1)
            std = torch.clamp(latent_std.view(1, 1, -1), min=1e-6)
            z0_continuous = z0_norm * std + mean
        else:
            z0_continuous = z0_norm

        # ---------------------------------------------------------------------
        # RVQ Projection: Quantize -> Dequantize
        # This forces the generated continuous latent onto the codebook manifold,
        # which greatly reduces geometric artifacts (collisions, overlaps) after
        # decoding, especially for long sequences.
        # ---------------------------------------------------------------------
        with torch.no_grad():
            # 1) Continuous latent -> discrete flat indices
            indices_flat = latent_to_indices_flat(
                vq_core,
                z0_continuous,
                token_mask=None,
                pad_id=None,
            ).long()  # [B, M * num_q]

            # 2) Discrete indices -> summed latent (projected latent)
            # returns [B, M, D]
            z0_projected, _ = indices_to_latent_sum(
                codebook,
                indices_flat,
                num_quantizers=num_q,
                pad_id=pad_id,
                return_token_mask=False,
            )

            final_latent = z0_projected

        # ---------------------------------------------------------------------
        # Decode to curves using VQ-VAE decoder
        # ---------------------------------------------------------------------
        L_max = int(max(target_lens))
        mask = torch.zeros(bs, L_max, dtype=torch.bool, device=device)
        for i, L_i in enumerate(target_lens):
            L_i = int(L_i)
            if L_i <= 0:
                continue
            if L_i > L_max:
                L_i = L_max
            mask[i, :L_i] = True

        with torch.no_grad():
            # VQ-VAE decoder expects [B, M, D] latents and a mask over output length
            recon = vq_core.decode(final_latent, mask=mask)  # [B, L_max, C]

        # ---------------------------------------------------------------------
        # Save latents and curves along with metadata
        # ---------------------------------------------------------------------
        for i in range(bs):
            sid = f"sample_{sample_id:07d}"

            # Save projected latent (after RVQ projection)
            z_np = final_latent[i].detach().cpu().numpy().astype(np.float32, copy=False)
            latent_path = latent_dir / f"{sid}.npy"
            np.save(latent_path, z_np, allow_pickle=False)

            # Save reconstructed curve up to its own target length
            L_i = int(target_lens[i])
            L_i = max(1, min(L_i, L_max))
            curve_np = (
                recon[i, :L_i]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            curve_path = curves_dir / f"{sid}.npy"
            np.save(curve_path, curve_np, allow_pickle=False)

            rec = {
                "id": sid,
                "latent_path": str(latent_path),
                "curve_path": str(curve_path),
                "latent_tokens": int(M),
                "target_len": int(L_i),
                "code_dim": int(vq_info.code_dim),
                "dtype": "float32",
            }
            f_manifest.write(json.dumps(rec) + "\n")
            sample_id += 1

        remaining -= bs

    f_manifest.close()
    print(f"[done] saved {sample_id} samples (latent + curves) to {out_dir}")


if __name__ == "__main__":
    main()
