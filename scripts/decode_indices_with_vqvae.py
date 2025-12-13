#!/usr/bin/env python3
# coding: utf-8
"""
Decode fixed-N code indices to curves with a trained VQVAE.
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
def indices_to_latent(core, indices_np: np.ndarray) -> torch.Tensor:
    if indices_np.ndim != 1:
        raise ValueError(f"indices must be 1D, got {indices_np.shape}")
    dev = next(core.parameters()).device
    inds = torch.from_numpy(indices_np).long().unsqueeze(0).to(dev)  # [1,N]
    E = get_codebook(core).to(dev)  # [K,D]
    z_q = F.embedding(inds, E)  # [1,N,D]
    return z_q


@torch.no_grad()
def decode_one(core, indices_np: np.ndarray, target_len: int) -> torch.Tensor:
    dev = next(core.parameters()).device

    # If model provides a helper, use it
    if hasattr(core, "decode_from_indices") and callable(core.decode_from_indices):
        inds = torch.from_numpy(indices_np).long().unsqueeze(0).to(dev)  # [1,N]
        out = core.decode_from_indices(inds, target_len=target_len)
        return out

    # Fallback: embed via codebook and call decode
    z_q = indices_to_latent(core, indices_np)  # [1,N,D]

    max_len = int(getattr(core, "max_seq_len", target_len))
    L = int(min(target_len, max_len))
    mask = torch.ones(1, L, dtype=torch.bool, device=z_q.device)

    # Default: decode(z_q, mask=mask)
    try:
        out = core.decode(z_q, mask=mask)
    except TypeError:
        # Try positional mask only
        try:
            out = core.decode(z_q, mask)
        except TypeError:
            # Last resort: maybe has target_len kwarg
            out = core.decode(z_q, target_len=L)
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
            print(f"[warn] latent_len mismatch {idxs.shape[0]} != {args.check_latent_len} at {idx_path}")

        recon = decode_one(core, idxs, tlen)  # [1,L,C]
        save_path = out_dir / (idx_path.stem + "_recon.npy")
        np.save(str(save_path), recon.squeeze(0).cpu().numpy(), allow_pickle=False)
        n_ok += 1

    print(f"Decoded {n_ok} sequences -> {out_dir}")


if __name__ == "__main__":
    main()
