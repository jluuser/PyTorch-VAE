# scripts/sample_with_aeot_omt.py
'''
python scripts/sample_with_aeot_omt.py \
  --ae_config configs/stage1_ae.yaml \
  --ae_ckpt /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/checkpoints/aeot/epochepoch=epoch=089.ckpt \
  --omt_ckpt checkpoints/omt_map_high_quality.pt \
  --out_dir results/aeot_generated \
  --num_samples 200 \
  --batch_size 32
'''
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Add repo root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiment import build_experiment_from_yaml
from aeot.omt_brenier import SemiDiscreteOMT, OMTCheckpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--ae_config", type=str, required=True, help="Path to configs/stage1_ae.yaml")
    p.add_argument("--ae_ckpt", type=str, required=True, help="Path to AE checkpoint")
    p.add_argument("--omt_ckpt", type=str, required=True, help="Path to OMT checkpoint produced by aeot/train_omt.py")

    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)

    # Length control:
    # - If length distribution exists in OMT checkpoint, we sample lengths from it.
    # - Otherwise, we fall back to --gen_len.
    p.add_argument("--gen_len", type=int, default=128, help="Fallback fixed length when no length distribution is available")
    p.add_argument("--min_len", type=int, default=1, help="Clamp sampled lengths to be at least this value")
    p.add_argument("--max_len", type=int, default=0, help="Optional clamp: if >0, clamp sampled lengths to be <= max_len")

    # AE-OT PL extension parameters
    p.add_argument("--k_neighbors", type=int, default=8)
    p.add_argument("--theta_deg", type=float, default=30.0)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--fp16", action="store_true", help="Use float16 inside OMT sampling (decoder still uses float32)")
    return p.parse_args()


@torch.no_grad()
def _load_ae(ae_config: str, ae_ckpt: str, device: torch.device):
    """
    Load AE model using your project's build_experiment_from_yaml and Lightning checkpoint format.
    """
    exp, _ = build_experiment_from_yaml(ae_config)

    ckpt = torch.load(ae_ckpt, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    new_state = {k.replace("model.", ""): v for k, v in state.items()}
    exp.model.load_state_dict(new_state, strict=False)

    exp.model.eval().to(device)
    return exp


@torch.no_grad()
def _load_omt_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load OMT checkpoint in dict format:
      {
        "omt": OMTCheckpoint,
        "normalize_targets": bool,
        "mean": Tensor,
        "std": Tensor,
        "length_values": Tensor (optional),
        "length_probs": Tensor (optional),
        ...
      }
    Also supports legacy format where the file is directly an OMTCheckpoint.
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "omt" in obj:
        return obj
    return {
        "omt": obj,
        "normalize_targets": False,
        "mean": None,
        "std": None,
    }


def _sample_lengths_from_dist(
    length_values: torch.Tensor,
    length_probs: torch.Tensor,
    n: int,
    min_len: int = 1,
    max_len: int = 0,
) -> torch.Tensor:
    """
    Sample integer lengths from a discrete distribution.

    Args:
        length_values: [M] int (possible lengths)
        length_probs:  [M] float (probabilities)
        n: number of samples
    Returns:
        lengths: [n] int64
    """
    lv = length_values.detach().cpu().numpy().astype(np.int64)
    lp = length_probs.detach().cpu().numpy().astype(np.float64)
    lp = lp / (lp.sum() + 1e-12)

    sampled = np.random.choice(lv, size=int(n), replace=True, p=lp)
    sampled = np.maximum(sampled, int(min_len))
    if int(max_len) > 0:
        sampled = np.minimum(sampled, int(max_len))
    return torch.from_numpy(sampled.astype(np.int64))


def _build_mask_from_lengths(lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Build a boolean mask [B, Lmax] from per-sample lengths [B].
    True indicates valid positions.
    """
    lengths = lengths.to(torch.int64)
    Lmax = int(lengths.max().item())
    ar = torch.arange(Lmax, device=device).view(1, -1)  # [1, Lmax]
    mask = ar < lengths.view(-1, 1)                     # [B, Lmax]
    return mask


@torch.no_grad()
def main() -> None:
    args = parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    omt_dtype = torch.float16 if args.fp16 else torch.float32

    # 1) Load AE
    exp = _load_ae(args.ae_config, args.ae_ckpt, device)
    ae = exp.model

    latent_tokens = int(ae.latent_n_tokens)
    code_dim = int(ae.code_dim)
    flat_dim = latent_tokens * code_dim

    # 2) Load OMT checkpoint + normalization metadata + (optional) length distribution
    obj = _load_omt_checkpoint(args.omt_ckpt)
    ckpt: OMTCheckpoint = obj["omt"]

    normalize_targets = bool(obj.get("normalize_targets", False))
    mean = obj.get("mean", None)
    std = obj.get("std", None)

    length_values = obj.get("length_values", None)
    length_probs = obj.get("length_probs", None)

    # Basic compatibility checks
    if ckpt.latent_tokens != latent_tokens or ckpt.code_dim != code_dim:
        raise RuntimeError(
            f"AE latent shape mismatch: AE has tokens={latent_tokens}, code_dim={code_dim}, "
            f"but OMT ckpt has tokens={ckpt.latent_tokens}, code_dim={ckpt.code_dim}."
        )
    if ckpt.y.shape[1] != flat_dim:
        raise RuntimeError(f"OMT latent dim mismatch: ckpt.y has D={ckpt.y.shape[1]}, expected {flat_dim}.")

    omt = SemiDiscreteOMT.from_checkpoint(ckpt, device=device, dtype=omt_dtype)

    if normalize_targets:
        if mean is None or std is None:
            raise RuntimeError("normalize_targets=True but checkpoint missing mean/std.")
        mean_dev = mean.to(device=device, dtype=torch.float32)
        std_dev = std.to(device=device, dtype=torch.float32)
    else:
        mean_dev = None
        std_dev = None

    # 3) Sampling loop
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = int(args.num_samples)
    bs_max = int(args.batch_size)
    saved = 0

    use_length_dist = (length_values is not None) and (length_probs is not None)

    if use_length_dist:
        # Print a quick summary
        lv = length_values
        lp = length_probs
        lmin = int(lv.min().item())
        lmax = int(lv.max().item())
        lmean = float((lv.to(torch.float32) * lp.to(torch.float32)).sum().item())
        print(f"[SampleOMT] Using length distribution from checkpoint: support={lv.numel()} min={lmin} mean={lmean:.2f} max={lmax}")
    else:
        print(f"[SampleOMT] No length distribution found. Using fixed gen_len={int(args.gen_len)}")

    while saved < total:
        bs = min(bs_max, total - saved)

        # Sample per-sample lengths
        if use_length_dist:
            lengths = _sample_lengths_from_dist(
                length_values=length_values,
                length_probs=length_probs,
                n=bs,
                min_len=int(args.min_len),
                max_len=int(args.max_len),
            )
        else:
            lengths = torch.full((bs,), int(args.gen_len), dtype=torch.int64)

        # Build variable-length mask
        mask = _build_mask_from_lengths(lengths, device=device)  # [B, Lmax]

        # Generate flattened latent codes in the OMT training space
        z_flat = omt.sample_extended(
            num_samples=bs,
            k_neighbors=int(args.k_neighbors),
            theta_deg=float(args.theta_deg),
            batch_size=max(64, bs),
        )  # [B, D] in omt space (normalized if normalize_targets)

        # Denormalize to AE latent space before decoding
        z_flat = z_flat.to(device=device, dtype=torch.float32)
        if normalize_targets:
            z_flat = z_flat * std_dev + mean_dev

        # Reshape back to token form for decoder
        z_tokens = z_flat.view(bs, latent_tokens, code_dim).to(device=device, dtype=torch.float32)

        # Decode
        recons = ae.decode(z_tokens, mask=mask)  # [B, Lmax, 6]

        xyz = recons[..., :3].float().cpu().numpy()         # [B, Lmax, 3]
        ss_logits = recons[..., 3:].float()                 # [B, Lmax, 3]
        ss_idx = torch.argmax(ss_logits, dim=-1)            # [B, Lmax]
        ss_one_hot = F.one_hot(ss_idx, num_classes=3).float().cpu().numpy()

        lengths_np = lengths.cpu().numpy().astype(np.int64)

        # Save per-sample with its own true length (no padding saved)
        for i in range(bs):
            Li = int(lengths_np[i])
            path = out_dir / f"gen_{saved + i:06d}.npy"
            np.save(
                str(path),
                {
                    "curve_coords": xyz[i, :Li].astype(np.float32),
                    "ss_one_hot": ss_one_hot[i, :Li].astype(np.float32),
                },
            )

        saved += bs
        print(f"[SampleOMT] Generated {saved}/{total}")

    print("[SampleOMT] Done.")


if __name__ == "__main__":
    main()
