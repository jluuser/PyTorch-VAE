#!/usr/bin/env python3
# coding: utf-8
"""
Extract fixed-N code indices (N tokens) from a trained VQVAE for prior training.
Also exports continuous encoder latents z_e and per-latent geometry descriptors.

Outputs (per sample):
  - indices_npy/sid.npy         : flattened RVQ indices [N_flat]
  - ze_npy/sid_ze.npy           : encoder latents z_e [M, D]
  - geo_npy/sid_geo.npy         : per-flat-position geometry [N_flat, G]
  - manifest.jsonl              : JSON lines with fields:
        {
          "id": ...,
          "indices_path": ...,
          "latent_path": ...,
          "latent_len": N_flat,
          "latent_tokens": M,
          "target_len": L,
          "dtype": "int16"/"int32",
          "rank": ...,
          "geo_path": ...,
          "geo_dim": ...
        }
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# repo root
sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from experiment import build_experiment_from_yaml
except Exception:
    build_experiment_from_yaml = None


def is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    return torch.distributed.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return torch.distributed.get_world_size() if is_dist() else 1


def barrier():
    if is_dist():
        torch.distributed.barrier()


def init_dist(backend: str = "nccl"):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)


def sha256_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser(description="Extract VQ indices and encoder latents (z_e) for prior.")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--yaml", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--max_batches", type=int, default=-1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--indices_dtype", type=str, default="int32", choices=["int16", "int32"])
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--expect_latent_len", type=int, default=0)
    return ap.parse_args()


def load_experiment(ckpt_path: str, yaml_path: str, device: str):
    if build_experiment_from_yaml is None:
        raise RuntimeError("Cannot import build_experiment_from_yaml from experiment.py")
    exp, _ = build_experiment_from_yaml(yaml_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    exp.load_state_dict(state, strict=False)
    exp.to(device)
    exp.eval()
    return exp


def build_dataloader(exp, split: str, num_workers: int, pin_memory: bool):
    if hasattr(exp, "setup"):
        exp.setup(stage="fit" if split in ["train", "val"] else "test")

    if split == "train":
        dl = exp.train_dataloader()
    elif split == "val":
        dl = exp.val_dataloader()
    else:
        if hasattr(exp, "test_dataloader"):
            dl = exp.test_dataloader()
        else:
            raise RuntimeError("No test_dataloader() defined")

    if not isinstance(dl, DataLoader):
        raise RuntimeError("Expected torch.utils.data.DataLoader")

    if is_dist():
        dataset = dl.dataset
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
        )
        dl = DataLoader(
            dataset,
            batch_size=dl.batch_size,
            sampler=sampler,
            num_workers=num_workers if num_workers is not None else dl.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=dl.collate_fn,
            persistent_workers=bool(num_workers and num_workers > 0),
            worker_init_fn=getattr(dl, "worker_init_fn", None),
        )
    else:
        sampler = None

    return dl, sampler


@torch.no_grad()
def _fallback_tokenize(core, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Fallback path: old-style encode -> tokenizer -> to_code.
    features: [B, L, H]
    Returns:
        z_e: [B, M, D]
    """
    if not hasattr(core, "tokenizer") or not hasattr(core, "to_code"):
        raise RuntimeError("encode() returned [B,L,H] but model has no tokenizer/to_code.")
    kpm = (~mask) if mask is not None else None
    h_mem = core.tokenizer(features, key_padding_mask=kpm)  # [B, M, H]
    z_e = core.to_code(h_mem)  # [B, M, D]
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

    For residual VQ (num_quantizers>1) when indices is a 1D flattened tensor,
    we interpret the original layout from VectorQuantizerEMA as:
        [level, batch, token] flattened -> [Q * B * M]
    and we convert it to:
        [B, M, Q] -> [B, M * Q]
    where per-sample sequence is:
        [t0_l0, t0_l1, ..., t0_l(Q-1), t1_l0, t1_l1, ..., t(M-1)_l(Q-1)].
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

        idx = indices.view(num_quantizers, B, M)  # [Q, B, M]
        idx = idx.permute(1, 2, 0).contiguous()   # [B, M, Q]
        return idx.view(B, M * num_quantizers)    # [B, M*Q]

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
def tokenize_and_quantize(
    core,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    """
    Run encoder to obtain encoder latents z_e and quantized indices.

    Returns:
        indices: [B, N_flat]
        lengths: [B] (original curve lengths in residue space)
        z_e:     [B, M, D] encoder latents (pre-quantization)
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
    if num_quantizers <= 0:
        num_quantizers = 1

    # Find 3D float features from encode output
    if isinstance(enc_out, (tuple, list)):
        feats = None
        for item in enc_out:
            if torch.is_tensor(item) and item.dim() == 3 and item.dtype.is_floating_point:
                feats = item
                break
        if feats is None:
            raise RuntimeError("encode() did not return 3D float features for z_e extraction.")
    else:
        feats = enc_out

    if not torch.is_tensor(feats) or feats.dim() != 3:
        raise RuntimeError("encode() tensor output must be [B, T, D].")
    B, T, _ = feats.shape

    # Obtain encoder latents z_e
    if latent_tokens is not None and T == int(latent_tokens):
        z_e = feats
    else:
        z_e = _fallback_tokenize(core, feats, mask)  # [B, M, D]

    # Quantize z_e to obtain indices
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

    lengths = mask.sum(dim=1).long().cpu().numpy()
    return indices, lengths, z_e


def compute_latent_geometry_for_sample(
    coords: np.ndarray,
    ss: np.ndarray,
    valid_len: int,
    num_codes: int,
    num_quantizers: int,
) -> np.ndarray:
    """
    Compute a simple per-latent geometry descriptor and broadcast it across RVQ levels.

    Args:
        coords: [L, 3] array of backbone coordinates.
        ss: [L, C] array of secondary-structure one-hot or similar.
        valid_len: number of valid residues (L).
        num_codes: total number of latent codes (flattened RVQ, N_flat).
        num_quantizers: number of RVQ levels (Q).

    Returns:
        geo_flat: [N_flat, D_geo] float32 array.
    """
    L = int(valid_len)
    if L <= 0 or num_codes <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    coords = coords[:L]
    ss = ss[:L]

    if num_quantizers <= 0:
        num_quantizers = 1

    N = int(num_codes)
    if N % num_quantizers != 0:
        num_quantizers = 1
    M = N // num_quantizers

    if M <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    bounds = np.linspace(0, L, M + 1, dtype=np.int64)
    ss_dim = ss.shape[1] if ss.ndim == 2 else 0
    geo_dim = 3 + 3 + ss_dim + 1  # center(3) + direction(3) + ss_mean + radius
    geo_per_pos = np.zeros((M, geo_dim), dtype=np.float32)

    for t in range(M):
        start = int(bounds[t])
        end = int(bounds[t + 1])
        if end <= start:
            end = min(L, start + 1)
        seg_coords = coords[start:end]
        seg_ss = ss[start:end]

        if seg_coords.shape[0] == 0:
            center = np.zeros(3, dtype=np.float32)
            direction = np.zeros(3, dtype=np.float32)
            radius = 0.0
        else:
            center = seg_coords.mean(axis=0).astype(np.float32)
            if seg_coords.shape[0] >= 2:
                vec = seg_coords[-1] - seg_coords[0]
                norm = float(np.linalg.norm(vec) + 1e-8)
                direction = (vec / norm).astype(np.float32)
            else:
                direction = np.zeros(3, dtype=np.float32)
            diffs = seg_coords - center
            radius = float(np.sqrt((diffs ** 2).sum(axis=1).mean()))

        if seg_ss.shape[0] > 0 and ss_dim > 0:
            ss_mean = seg_ss.mean(axis=0).astype(np.float32)
        else:
            ss_mean = np.zeros(ss_dim, dtype=np.float32)

        geo_vec = np.concatenate(
            [center, direction, ss_mean, np.array([radius], dtype=np.float32)],
            axis=0,
        )
        geo_per_pos[t] = geo_vec

    if num_quantizers == 1:
        geo_flat = geo_per_pos
    else:
        geo_flat = np.repeat(geo_per_pos, num_quantizers, axis=0)

    return geo_flat.astype(np.float32, copy=False)


def main():
    args = parse_args()
    init_dist("nccl")
    rank = get_rank()
    world = get_world_size()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    ckpt_path = Path(args.ckpt).resolve()
    yaml_path = Path(args.yaml).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    if rank == 0:
        meta: Dict[str, Any] = {
            "ckpt_path": str(ckpt_path),
            "yaml_path": str(yaml_path),
            "ckpt_sha256": sha256_of_file(str(ckpt_path)) if ckpt_path.exists() else "",
            "dtype": args.indices_dtype,
            "split": args.split,
            "world_size": world,
        }
        with open(out_dir / "extract_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    exp = load_experiment(str(ckpt_path), str(yaml_path), str(device))
    model = exp.model if hasattr(exp, "model") else exp
    model.eval()
    core = model

    q = getattr(core, "quantizer", None)
    if q is None:
        num_quantizers = 1
    else:
        num_quantizers = int(getattr(q, "num_quantizers", 1))
    if num_quantizers <= 0:
        num_quantizers = 1

    dl, _ = build_dataloader(exp, split=args.split, num_workers=args.num_workers, pin_memory=args.pin_memory)

    rank_dir = out_dir / f"rank{rank}"
    indices_dir = rank_dir / "indices_npy"
    geo_dir = rank_dir / "geo_npy"
    ze_dir = rank_dir / "ze_npy"
    ensure_dir(indices_dir)
    ensure_dir(geo_dir)
    ensure_dir(ze_dir)
    manifest_rank_path = out_dir / f"manifest_rank{rank}.jsonl"

    buffer_lines: List[str] = []
    batches_done = 0
    total_saved = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            if isinstance(batch, dict):
                x = batch.get("x", None)
                mask = batch.get("mask", None)
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, mask = batch[0], batch[1]
            else:
                raise ValueError("Batch must be dict or tuple (x, mask)")

            if x is None or mask is None:
                raise ValueError("Batch missing x or mask")

            indices_bt, lengths_bt, z_e_bt = tokenize_and_quantize(core, x, mask)  # [B,N_flat], [B], [B,M,D]
            B, N_flat = indices_bt.shape

            if args.expect_latent_len > 0 and N_flat != int(args.expect_latent_len):
                print(f"[warn][rank{rank}] latent_len mismatch: got {N_flat}, expect {args.expect_latent_len}")

            x_np = x.cpu().numpy()
            mask_np = mask.cpu().numpy()
            ze_np = z_e_bt.cpu().numpy()  # [B, M, D]

            for b in range(B):
                seq = indices_bt[b].cpu().numpy()
                latent_flat_len = int(seq.shape[0])
                target_len = int(lengths_bt[b])

                # encoder latents z_e for this sample
                ze_b = ze_np[b]  # [M, D]
                latent_tokens = int(ze_b.shape[0])

                # Save indices
                if args.indices_dtype == "int16" and int(seq.max(initial=0)) < np.iinfo(np.int16).max:
                    seq_to_save = seq.astype(np.int16, copy=False)
                    save_dtype = "int16"
                else:
                    seq_to_save = seq.astype(np.int32, copy=False)
                    save_dtype = "int32"

                sid = f"rank{rank}_sample_{batches_done:06d}_{b:03d}"
                out_path = indices_dir / f"{sid}.npy"
                np.save(str(out_path), seq_to_save, allow_pickle=False)

                # Save encoder latent z_e
                ze_path = ze_dir / f"{sid}_ze.npy"
                np.save(str(ze_path), ze_b.astype(np.float32, copy=False), allow_pickle=False)

                # Compute geometry descriptors based on original coordinates + ss
                x_b = x_np[b]
                m_b = mask_np[b]
                L = int(m_b.sum())
                coords = x_b[:L, :3]
                ss = x_b[:L, 3:]
                geo_flat = compute_latent_geometry_for_sample(
                    coords=coords,
                    ss=ss,
                    valid_len=L,
                    num_codes=latent_flat_len,
                    num_quantizers=num_quantizers,
                )
                geo_path = geo_dir / f"{sid}_geo.npy"
                np.save(str(geo_path), geo_flat.astype(np.float32, copy=False), allow_pickle=False)
                geo_dim = int(geo_flat.shape[1]) if geo_flat.size > 0 else 0

                rec = {
                    "id": sid,
                    "indices_path": str(out_path),
                    "latent_path": str(ze_path),
                    "latent_len": latent_flat_len,        # flattened indices length
                    "latent_tokens": latent_tokens,       # encoder tokens M
                    "target_len": target_len,
                    "dtype": save_dtype,
                    "rank": rank,
                    "geo_path": str(geo_path),
                    "geo_dim": geo_dim,
                }
                buffer_lines.append(json.dumps(rec))
                total_saved += 1

            batches_done += 1
            if (batches_done % args.save_every) == 0 and buffer_lines:
                with open(manifest_rank_path, "a") as fw:
                    fw.write("\n".join(buffer_lines) + "\n")
                buffer_lines = []

    if buffer_lines:
        with open(manifest_rank_path, "a") as fw:
            fw.write("\n".join(buffer_lines) + "\n")

    barrier()

    if rank == 0:
        merged = out_dir / "manifest.jsonl"
        with open(merged, "w") as fout:
            for r in range(world):
                part = out_dir / f"manifest_rank{r}.jsonl"
                if part.exists():
                    with open(part, "r") as fin:
                        for line in fin:
                            if line.strip():
                                fout.write(line.rstrip("\n") + "\n")
        print(f"[rank0] merged manifest -> {merged}")

    print(f"[rank{rank}] Done. Batches: {batches_done}, samples saved: {total_saved}, manifest: {manifest_rank_path}")
    print(f"[rank{rank}] Indices dir: {indices_dir}")
    print(f"[rank{rank}] z_e dir: {ze_dir}")


if __name__ == "__main__":
    main()
