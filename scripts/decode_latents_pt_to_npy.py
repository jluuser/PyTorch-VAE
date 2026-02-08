#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Decode latent vectors stored in a .pt file back to curve .npy files using a trained AE decoder.

This script writes each output .npy as a pure ndarray float32 of shape [L,6]:
  - [:, :3] = xyz
  - [:, 3:6] = ss_one_hot (hard one-hot converted from decoder logits)

This output format is compatible with:
  1) prior/filter_curves.py (np.load(..., allow_pickle=False) expects ndarray)
  2) scripts/visualize_inference_curves.py (supports ndarray [L,6])

Example:
python scripts/decode_latents_pt_to_npy.py \
  --ae_config configs/stage1_ae.yaml \
  --ae_ckpt /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/checkpoints/aeot_sigmoid/epochepoch=epoch=089.ckpt \
  --latents_pt /public/home/zhangyangroup/chengshiz/keyuan.zhou/prp-dataset/gen_features.pt \
  --out_dir results/decoded_npy_122_sigmoid \
  --latent_key latents \
  --batch_size 64 \
  --save_manifest
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add repo root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from experiment import build_experiment_from_yaml


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--ae_config", type=str, required=True, help="Path to AE config yaml (stage1_ae.yaml)")
    p.add_argument("--ae_ckpt", type=str, required=True, help="Path to trained AE checkpoint")
    p.add_argument("--latents_pt", type=str, required=True, help="Path to input .pt file containing latents")
    p.add_argument("--out_dir", type=str, required=True, help="Directory to write decoded .npy files")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")

    # Which latent tensor to decode: "latents" or "latents_norm"
    p.add_argument("--latent_key", type=str, default="latents",
                   choices=["latents", "latents_norm"],
                   help="Which key in pt to use as latent input.")

    # If using latents_norm, denormalize by per-dim minmax: z = zn*(max-min)+min
    p.add_argument("--denorm_minmax", action="store_true",
                   help="Apply per-dimension min-max denormalization using norm_min/norm_max in pt.")

    # Length control
    p.add_argument("--gen_len", type=int, default=128,
                   help="Fallback length if no 'lengths' exists in .pt (ignored when lengths are provided).")
    p.add_argument("--min_len", type=int, default=1, help="Clamp minimal length when using 'lengths'")
    p.add_argument("--max_len", type=int, default=0, help="Clamp maximal length when using 'lengths' (0 disables)")

    # Subset decode
    p.add_argument("--start", type=int, default=0, help="Start index in latents")
    p.add_argument("--num", type=int, default=0, help="Number of samples to decode (0 means decode all from start)")

    # Output naming
    p.add_argument("--name_pattern", type=str, default="gen_{idx:06d}.npy",
                   help='Filename pattern, e.g. "sample_{idx:06d}_recon.npy"')

    # Manifest
    p.add_argument("--save_manifest", action="store_true", help="Also save a samples_manifest.jsonl (json per line)")

    return p.parse_args()


def _safe_load_ae(ae_config: str, ae_ckpt: str, device: torch.device):
    """
    Load AE model using build_experiment_from_yaml and Lightning checkpoint format.
    """
    exp, cfg = build_experiment_from_yaml(ae_config)

    ckpt = torch.load(ae_ckpt, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Strip possible Lightning prefix "model."
    new_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            new_state[k[6:]] = v
        else:
            new_state[k] = v

    exp.model.load_state_dict(new_state, strict=False)
    exp.model.eval().to(device)

    # Read latent_tokens / code_dim from yaml config (critical when pt doesn't contain them)
    mp = cfg.get("model_params", {}) if isinstance(cfg, dict) else {}
    latent_tokens = int(mp.get("latent_tokens", getattr(exp.model, "latent_n_tokens", 0)) or 0)
    code_dim = int(mp.get("code_dim", getattr(exp.model, "code_dim", 0)) or 0)

    if latent_tokens <= 0 or code_dim <= 0:
        raise RuntimeError("Failed to obtain latent_tokens/code_dim from YAML or model.")

    return exp.model, latent_tokens, code_dim


def _clamp_lengths(lengths: torch.Tensor, min_len: int, max_len: int) -> torch.Tensor:
    lengths = lengths.to(torch.int64)
    lengths = torch.clamp(lengths, min=min_len)
    if max_len and max_len > 0:
        lengths = torch.clamp(lengths, max=max_len)
    return lengths


def _build_mask_from_lengths(lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Build boolean mask [B, Lmax] from per-sample lengths [B]. True indicates valid positions.
    """
    lengths = lengths.to(torch.int64)
    Lmax = int(lengths.max().item())
    ar = torch.arange(Lmax, device=device).view(1, -1)
    mask = ar < lengths.view(-1, 1)
    return mask


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    # 1) Load AE + get latent shape from YAML
    print("[Info] Loading AE...")
    ae, latent_tokens, code_dim = _safe_load_ae(args.ae_config, args.ae_ckpt, device)
    flat_dim_expected = latent_tokens * code_dim
    print(f"[Info] latent_tokens={latent_tokens} code_dim={code_dim} flat_dim={flat_dim_expected}")

    # 2) Load pt
    print(f"[Info] Loading latents from: {args.latents_pt}")
    data = torch.load(args.latents_pt, map_location="cpu")
    if not isinstance(data, dict):
        raise RuntimeError("Input .pt must be a dict.")

    if args.latent_key not in data:
        raise KeyError(f"Missing key '{args.latent_key}' in {args.latents_pt}")

    z_in = data[args.latent_key]
    if not isinstance(z_in, torch.Tensor):
        raise RuntimeError(f"{args.latent_key} must be a torch.Tensor")

    z_in = z_in.float().contiguous()
    if z_in.ndim != 2:
        raise RuntimeError(f"{args.latent_key} must be [N, D_flat], got shape={tuple(z_in.shape)}")

    N, D = z_in.shape
    if D != flat_dim_expected:
        raise RuntimeError(f"Latent dim mismatch: got D={D}, expected {flat_dim_expected} (=latent_tokens*code_dim)")

    lengths = data.get("lengths", None)
    has_lengths = isinstance(lengths, torch.Tensor)

    # Slice range
    start = max(0, int(args.start))
    end = min(N, start + int(args.num)) if int(args.num) > 0 else N
    if start >= end:
        raise ValueError(f"Invalid slice: start={start}, end={end}, N={N}")

    z_in = z_in[start:end].contiguous()
    Nsel = z_in.size(0)

    if has_lengths:
        lengths = lengths.view(-1).contiguous()
        if lengths.numel() != N:
            print(f"[Warn] lengths numel {lengths.numel()} != N {N}, will still slice by index range.")
        lengths = lengths[start:end].contiguous()
        lengths = _clamp_lengths(lengths, int(args.min_len), int(args.max_len))
        print(f"[Info] Using variable lengths from pt: N={Nsel}")
        print(f"[Info] Length stats: min={int(lengths.min().item())}, mean={float(lengths.float().mean().item()):.2f}, max={int(lengths.max().item())}")
    else:
        print(f"[Info] No lengths in pt. Using fixed gen_len={int(args.gen_len)} for all samples. N={Nsel}")

    # Optional min-max denormalization (per-dimension)
    do_minmax = bool(args.denorm_minmax)
    if do_minmax:
        norm_min = data.get("norm_min", None)
        norm_max = data.get("norm_max", None)
        if not isinstance(norm_min, torch.Tensor) or not isinstance(norm_max, torch.Tensor):
            raise RuntimeError("--denorm_minmax is set but norm_min/norm_max are missing or not tensors.")
        norm_min = norm_min.float()
        norm_max = norm_max.float()
        if norm_min.numel() != D or norm_max.numel() != D:
            raise RuntimeError(f"norm_min/norm_max must be [D_flat], got {tuple(norm_min.shape)} / {tuple(norm_max.shape)}")
        # Keep on CPU; we will move to GPU once per batch
        print("[Info] Will apply per-dimension min-max denormalization: z = zn*(max-min)+min")

    # 3) Decode loop
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_f = None
    if args.save_manifest:
        manifest_path = out_dir / "samples_manifest.jsonl"
        manifest_f = open(manifest_path, "w", encoding="utf-8")
        print(f"[Info] Writing manifest: {manifest_path}")

    bs = int(args.batch_size)
    pbar = tqdm(total=Nsel, desc="Decoding")

    for i0 in range(0, Nsel, bs):
        i1 = min(Nsel, i0 + bs)
        z_flat = z_in[i0:i1].to(device, non_blocking=True)  # [B, D]

        if do_minmax:
            # Move min/max to GPU for this batch once
            nm = norm_min.to(device, non_blocking=True)
            nx = norm_max.to(device, non_blocking=True)
            z_flat = z_flat * (nx - nm) + nm

        # Reshape to token form: [B, latent_tokens, code_dim]
        z_tokens = z_flat.view(i1 - i0, latent_tokens, code_dim).contiguous()

        # Build mask
        if has_lengths:
            batch_lengths = lengths[i0:i1].to(device, non_blocking=True)
            mask = _build_mask_from_lengths(batch_lengths, device=device)  # [B, Lmax]
        else:
            mask = torch.ones((i1 - i0, int(args.gen_len)), dtype=torch.bool, device=device)

        # Decode: [B, L, 6] (xyz + ss_logits)
        recons = ae.decode(z_tokens, mask=mask)

        # Convert to [B, L, 6] where last 3 are hard one-hot
        coords = recons[..., :3].float()              # [B, L, 3]
        ss_logits = recons[..., 3:].float()           # [B, L, 3]
        ss_idx = torch.argmax(ss_logits, dim=-1)      # [B, L]
        ss_one_hot = F.one_hot(ss_idx, num_classes=3).float()  # [B, L, 3]

        arr6 = torch.cat([coords, ss_one_hot], dim=-1).cpu().numpy().astype(np.float32)  # [B, L, 6]

        # Save per-sample, trimming padding if variable length
        for bi in range(i1 - i0):
            global_idx = start + i0 + bi
            fname = args.name_pattern.format(idx=global_idx)
            out_path = out_dir / fname

            L = int(lengths[i0 + bi].item()) if has_lengths else int(args.gen_len)
            np.save(str(out_path), arr6[bi, :L], allow_pickle=False)

            if manifest_f is not None:
                rec = {
                    "i": int(global_idx),
                    "recon_path": str(out_path),
                    "length_recon": int(L),
                    "latent_key": str(args.latent_key),
                    "denorm_minmax": bool(do_minmax),
                }
                manifest_f.write(json.dumps(rec) + "\n")

        pbar.update(i1 - i0)

    pbar.close()
    if manifest_f is not None:
        manifest_f.close()

    print(f"[Info] Done. Wrote {Nsel} files to: {out_dir}")


if __name__ == "__main__":
    main()
