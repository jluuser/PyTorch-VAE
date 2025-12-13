#!/usr/bin/env python3
# coding: utf-8
"""
Extract fixed-N code indices (N tokens) from a trained VQVAE for prior training.
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import Tuple, List

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
    ap = argparse.ArgumentParser(description="Extract VQ indices for prior.")
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
            shuffle=True
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
            worker_init_fn=getattr(dl, "worker_init_fn", None)
        )
    else:
        sampler = None

    return dl, sampler


@torch.no_grad()
def _fallback_tokenize(core, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Fallback path: old-style encode -> tokenizer -> to_code.
    features: [B, L, H]
    """
    if not hasattr(core, "tokenizer") or not hasattr(core, "to_code"):
        raise RuntimeError("encode() returned [B,L,H] but model has no tokenizer/to_code.")
    kpm = (~mask) if mask is not None else None
    h_mem = core.tokenizer(features, key_padding_mask=kpm)  # [B, N, H]
    z_e = core.to_code(h_mem)  # [B, N, D]
    return z_e


@torch.no_grad()
def tokenize_and_quantize(core, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
    dev = next(core.parameters()).device
    x = x.to(dev, non_blocking=True)
    mask = mask.to(dev, non_blocking=True)

    if not hasattr(core, "encode") or not callable(core.encode):
        raise RuntimeError("VQVAE model must implement encode(x, mask=mask).")

    enc_out = core.encode(x, mask=mask)
    q = getattr(core, "quantizer", None)
    if q is None:
        raise RuntimeError("model.quantizer not found")

    latent_tokens = getattr(core, "latent_tokens", None)

    indices: torch.Tensor

    # Case 1: encode returns tuple/list
    if isinstance(enc_out, (tuple, list)):
        # Try to find a 2D integer tensor as indices
        cand_idx = None
        for item in enc_out:
            if torch.is_tensor(item) and item.dim() == 2 and item.dtype in (torch.int64, torch.int32):
                cand_idx = item
                break
        if cand_idx is not None:
            indices = cand_idx.long()
        else:
            # assume first tensor is latent features
            if not torch.is_tensor(enc_out[0]):
                raise RuntimeError("encode() output[0] must be a tensor.")
            feats = enc_out[0]
            if feats.dim() != 3:
                raise RuntimeError("encode()[0] tensor must be [B, T, D].")
            B, T, _ = feats.shape

            # If T == latent_tokens -> already [B,N,D]
            if latent_tokens is not None and T == int(latent_tokens):
                z_e = feats
            else:
                # fallback: encoder features [B,L,H]
                z_e = _fallback_tokenize(core, feats, mask)

            q_out = q(z_e, do_ema_update=False, allow_reinit=False, mask=None)
            if isinstance(q_out, (tuple, list)) and len(q_out) >= 3:
                indices = q_out[2].long()
            elif torch.is_tensor(q_out):
                indices = q_out.long()
            else:
                raise RuntimeError("Unsupported quantizer output type.")
    else:
        # Case 2: encode returns a single tensor [B,T,D]
        feats = enc_out
        if not torch.is_tensor(feats) or feats.dim() != 3:
            raise RuntimeError("encode() tensor output must be [B, T, D].")
        B, T, _ = feats.shape
        if latent_tokens is not None and T == int(latent_tokens):
            z_e = feats
        else:
            z_e = _fallback_tokenize(core, feats, mask)

        q_out = q(z_e, do_ema_update=False, allow_reinit=False, mask=None)
        if isinstance(q_out, (tuple, list)) and len(q_out) >= 3:
            indices = q_out[2].long()
        elif torch.is_tensor(q_out):
            indices = q_out.long()
        else:
            raise RuntimeError("Unsupported quantizer output type.")

    lengths = mask.sum(dim=1).long().cpu().numpy()
    return indices, lengths


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
        meta = {
            "ckpt_path": str(ckpt_path),
            "yaml_path": str(yaml_path),
            "ckpt_sha256": sha256_of_file(str(ckpt_path)) if ckpt_path.exists() else "",
            "dtype": args.indices_dtype,
            "split": args.split,
            "world_size": world
        }
        with open(out_dir / "extract_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    exp = load_experiment(str(ckpt_path), str(yaml_path), str(device))
    model = exp.model if hasattr(exp, "model") else exp
    model.eval()
    core = model

    dl, _ = build_dataloader(exp, split=args.split, num_workers=args.num_workers, pin_memory=args.pin_memory)

    rank_dir = out_dir / f"rank{rank}"
    indices_dir = rank_dir / "indices_npy"
    ensure_dir(indices_dir)
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

            indices_bt, lengths_bt = tokenize_and_quantize(core, x, mask)  # [B,N], [B]
            B, N = indices_bt.shape

            if args.expect_latent_len > 0 and N != int(args.expect_latent_len):
                print(f"[warn] latent_len mismatch: got {N}, expect {args.expect_latent_len}")

            for b in range(B):
                seq = indices_bt[b].cpu().numpy()
                latent_len = int(seq.shape[0])
                target_len = int(lengths_bt[b])

                if args.indices_dtype == "int16" and int(seq.max(initial=0)) < np.iinfo(np.int16).max:
                    seq_to_save = seq.astype(np.int16, copy=False)
                    save_dtype = "int16"
                else:
                    seq_to_save = seq.astype(np.int32, copy=False)
                    save_dtype = "int32"

                sid = f"rank{rank}_sample_{batches_done:06d}_{b:03d}"
                out_path = indices_dir / f"{sid}.npy"
                np.save(str(out_path), seq_to_save, allow_pickle=False)

                rec = {
                    "id": sid,
                    "indices_path": str(out_path),
                    "latent_len": latent_len,
                    "target_len": target_len,
                    "dtype": save_dtype,
                    "rank": rank
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


if __name__ == "__main__":
    main()
