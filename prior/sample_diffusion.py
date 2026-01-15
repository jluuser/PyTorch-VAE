#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

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
    """Sample target curve lengths from a manifest."""

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
            self.lengths = [128]

    def sample(self, n: int) -> List[int]:
        idx = np.random.randint(0, len(self.lengths), size=int(n))
        return [self.lengths[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="diffusion prior ckpt")
    ap.add_argument("--config", type=str, default="", help="diffusion prior yaml (optional)")
    ap.add_argument("--vq_ckpt", type=str, default="")
    ap.add_argument("--vq_yaml", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--sample_steps", type=int, default=100)
    ap.add_argument(
        "--latent_tokens",
        type=int,
        default=0,
        help="override latent token count M (optional)",
    )
    ap.add_argument(
        "--target_len",
        type=int,
        default=0,
        help="fixed target curve length (optional)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
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

    # Load diffusion prior checkpoint and config
    ckpt_obj = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt_obj.get("config", {})
    if args.config:
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

    # Load VQ-VAE
    vq_exp = load_vq_experiment(vq_ckpt, vq_yaml, device)
    vq_core = core_model(vq_exp)
    vq_info = get_vq_info(vq_core)
    num_q = int(vq_info.num_quantizers)

    # Load z_q normalization stats
    stats_path = str(cfg_runtime.get("zq_stats_path", "")).strip()
    if not stats_path:
        raise RuntimeError("runtime.zq_stats_path is required in config")
    stats_path = str(Path(stats_path).expanduser())
    if not Path(stats_path).is_file():
        raise FileNotFoundError(f"z_q stats file not found: {stats_path}")
    stats = np.load(stats_path)
    mean_np = stats["mean"].astype("float32")
    std_np = stats["std"].astype("float32")
    zq_mean = torch.from_numpy(mean_np).to(device)
    zq_std = torch.from_numpy(std_np).to(device)
    zq_std = torch.clamp(zq_std, min=1e-6)

    # Diffusion schedule and model
    schedule = DiffusionSchedule.build(
        num_steps=int(cfg_diff.get("num_steps", 1000)),
        schedule=str(cfg_diff.get("schedule", "cosine")),
        beta_start=float(cfg_diff.get("beta_start", 1e-4)),
        beta_end=float(cfg_diff.get("beta_end", 2e-2)),
        cosine_s=float(cfg_diff.get("cosine_s", 0.008)),
    )
    diffusion = GaussianDiffusion(betas=schedule.betas).to(device)

    den_cfg = DenoiserConfig(
        code_dim=int(vq_info.code_dim),
        hidden_channels=int(cfg_model.get("hidden_channels", 256)),
        time_embed_dim=int(cfg_model.get("time_embed_dim", 256)),
        num_blocks=int(cfg_model.get("num_blocks", 12)),
        dropout=float(cfg_model.get("dropout", 0.0)),
        cond_dim=0,
    )
    model = DiffusionDenoiserResNet1D(den_cfg).to(device)
    model.load_state_dict(ckpt_obj["model"], strict=True)
    model.eval()

    # Apply EMA if available
    if "ema" in ckpt_obj and bool(cfg.get("ema", {}).get("enable", True)):
        ema = EMA(decay=0.999)
        ema.load_state_dict(ckpt_obj["ema"], device=device)
        ema.copy_to(model)

    # Determine latent token length M
    M = int(args.latent_tokens) if int(args.latent_tokens) > 0 else 0
    if M <= 0:
        max_len = int(cfg_data.get("max_len", 0))
        if max_len > 0 and max_len % num_q == 0:
            M = max_len // num_q
        else:
            raise RuntimeError(
                "latent_tokens not provided and cannot infer from config.data.max_len"
            )

    # Length sampler from training manifest (for target curves)
    length_sampler = None
    train_manifest = str(cfg_data.get("train_manifest", ""))
    if train_manifest:
        length_sampler = LengthSampler(train_manifest)

    manifest_path = out_dir / "samples_manifest.jsonl"
    f_manifest = manifest_path.open("w")

    remaining = int(args.num_samples)
    sample_id = 0
    while remaining > 0:
        bs = min(int(args.batch_size), remaining)

        # Sample target curve lengths
        if int(args.target_len) > 0:
            target_lens = [int(args.target_len)] * bs
        elif length_sampler is not None:
            target_lens = length_sampler.sample(bs)
        else:
            target_lens = [128] * bs

        # Latent diffusion mask (no token-level masking here, full length M)
        token_mask = None

        # Sample normalized latent x0 ~ p(x0) via DDIM
        z_norm = diffusion.ddim_sample_loop(
            model,
            shape=(bs, M, int(vq_info.code_dim)),
            steps=int(args.sample_steps),
            device=device,
            cond=None,
            mask=token_mask,
        )  # [B, M, D]

        # Inverse normalization to recover z_q
        z_q = z_norm * zq_std.view(1, 1, -1) + zq_mean.view(1, 1, -1)

        # Build curve-length mask for the decoder
        L_max = int(max(target_lens))
        mask_L = torch.zeros(bs, L_max, dtype=torch.bool, device=device)
        for i, L_i in enumerate(target_lens):
            L_i = int(L_i)
            if L_i <= 0:
                continue
            if L_i > L_max:
                L_i = L_max
            mask_L[i, :L_i] = True

        # Decode z_q to curves
        with torch.no_grad():
            recon = vq_core.decode(z_q, mask=mask_L)  # [B, L_max, C]

        # Save per-sample latent and curves
        for i in range(bs):
            sid = f"sample_{sample_id:07d}"

            # Save latent z_q
            z_np = z_q[i].detach().cpu().numpy().astype(np.float32, copy=False)
            latent_path = latent_dir / f"{sid}.npy"
            np.save(latent_path, z_np, allow_pickle=False)

            # Save curve up to target length
            L_i = int(target_lens[i])
            L_i = max(1, min(L_i, L_max))
            curve_np = (
                recon[i, :L_i].detach().cpu().numpy().astype(np.float32, copy=False)
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
