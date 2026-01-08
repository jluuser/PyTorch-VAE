#!/usr/bin/env python3
# coding: utf-8
"""
Decode fixed-N code indices to curves with a trained VQVAE.

Supports both:
  - single-codebook VQ (num_quantizers = 1)
  - residual VQ (num_quantizers > 1) with flattened indices of length
    latent_tokens * num_quantizers, where codes per token are summed.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

# repo root
sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from experiment import build_experiment_from_yaml
except Exception:
    build_experiment_from_yaml = None

try:
    from experiment import VQVAEExperiment
except Exception:
    VQVAEExperiment = None


def load_experiment(ckpt: str, yaml_path: str, device: torch.device):
    if build_experiment_from_yaml is not None:
        exp, _ = build_experiment_from_yaml(yaml_path)
        state = torch.load(ckpt, map_location="cpu")
        sd = state.get("state_dict", state)
        exp.load_state_dict(sd, strict=False)
        exp.to(device)
        exp.eval()
        return exp
    if VQVAEExperiment is not None:
        exp = VQVAEExperiment.load_from_checkpoint(
            checkpoint_path=ckpt, map_location="cpu", strict=False
        ).to(device)
        exp.eval()
        return exp
    raise RuntimeError("No experiment loader found")


@torch.no_grad()
def core_model(exp_or_model):
    return exp_or_model.model if hasattr(exp_or_model, "model") else exp_or_model


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
    Map discrete indices to latent z_q.

    For single-codebook VQ (num_quantizers=1):
        indices_np: [N]
        z_q: [1, N, D]

    For residual VQ (num_quantizers>1) with flattened indices:
        indices_np: [N_flat] where N_flat = N_tokens * num_quantizers
        We interpret the sequence as:
            [t0_level0, t0_level1, ..., t0_levelQ-1,
             t1_level0, t1_level1, ..., t1_levelQ-1, ...]
        Then:
            embed -> [1, N_flat, D]
            reshape -> [1, N_tokens, num_quantizers, D]
            sum over num_quantizers -> [1, N_tokens, D]
    """
    if indices_np.ndim != 1:
        raise ValueError(f"indices must be 1D, got {indices_np.shape}")
    dev = next(core.parameters()).device
    inds = torch.from_numpy(indices_np).long().unsqueeze(0).to(dev)  # [1, N_flat]
    E = get_codebook(core).to(dev)  # [K_total, D]

    num_q = get_num_quantizers(core)
    # Simple case: single codebook
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

    z_all = F.embedding(inds, E)  # [1, N_flat, D]
    z_all = z_all.view(1, N_tokens, num_q, -1)  # [1, N, Q, D]
    z_q = z_all.sum(dim=2)  # [1, N, D]
    return z_q


@torch.no_grad()
def decode_one(core, indices_np: np.ndarray, target_len: int) -> torch.Tensor:
    dev = next(core.parameters()).device

    # If model provides a helper, use it (it may already know about RVQ layout)
    if hasattr(core, "decode_from_indices") and callable(core.decode_from_indices):
        inds = torch.from_numpy(indices_np).long().unsqueeze(0).to(dev)  # [1, N_flat]
        try:
            out = core.decode_from_indices(inds, target_len=target_len)
        except TypeError:
            out = core.decode_from_indices(inds)
        return out

    # Fallback: reconstruct z_q from indices and call decode
    z_q = indices_to_latent(core, indices_np)  # [1, N_latent, D]
    B, L_latent, _ = z_q.shape

    # build latent mask (all valid)
    mask = torch.ones(B, L_latent, dtype=torch.bool, device=z_q.device)

    # Try various decode signatures
    try:
        out = core.decode(z_q, mask=mask, target_len=int(target_len))
    except TypeError:
        try:
            out = core.decode(z_q, mask=mask)
        except TypeError:
            try:
                out = core.decode(z_q, int(target_len))
            except TypeError:
                out = core.decode(z_q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vq_ckpt", type=str, required=True)
    ap.add_argument("--vq_yaml", type=str, required=True)
    ap.add_argument("--samples_manifest", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check_latent_len", type=int, default=0)
    args = ap.parse_args()

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")

    exp = load_experiment(args.vq_ckpt, args.vq_yaml, device)
    core = core_model(exp)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.samples_manifest, "r") as f:
        records = [json.loads(l) for l in f if l.strip()]

    if args.limit > 0:
        records = records[: args.limit]

    n_ok = 0
    for r in records:
        idx_path = Path(r["indices_path"])
        if not idx_path.exists():
            print(f"[warn] missing indices: {idx_path}")
            continue

        tlen = int(r["target_len"]) if "target_len" in r else int(r.get("length", 0))
        if tlen <= 0:
            print(f"[warn] invalid target_len for {idx_path}")
            continue

        idxs = np.load(str(idx_path), allow_pickle=False)
        if args.check_latent_len > 0 and int(idxs.shape[0]) != int(args.check_latent_len):
            print(
                f"[warn] latent_len mismatch {idxs.shape[0]} != {args.check_latent_len} at {idx_path}"
            )

        recon = decode_one(core, idxs, tlen)  # [1, L, C]
        save_path = out_dir / (idx_path.stem + "_recon.npy")
        np.save(str(save_path), recon.squeeze(0).cpu().numpy(), allow_pickle=False)
        n_ok += 1

    print(f"Decoded {n_ok} sequences -> {out_dir}")


if __name__ == "__main__":
    main()
