# coding: utf-8
"""
Dataset and batching utilities for Transformer Prior over VQ-VAE code indices.

- PriorIndexDataset: loads lines from manifest.jsonl; each line points to one indices .npy
- collate_pad: pads input/target to batch max length using PAD id; builds boolean attn_mask
- BucketBatchSampler (optional): length-aware bucketed batching with distributed sharding

This file is ASCII-only.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class PriorIndexDataset(Dataset):
    """
    Each sample:
      - load a 1D int array [L] of code indices from .npy
      - build:
          inp = [BOS, z1, z2, ..., zL]          (length L+1, clipped by max_len)
          tgt = [z1,  z2, ..., zL, EOS]         (length L+1, clipped by max_len)
      - attn_mask is built in collate_pad, not here

    Notes:
      - We trust 'length' in manifest if present to avoid pre-loading for length queries.
      - max_len applies to the raw sequence length L BEFORE adding BOS/EOS; so effective len is <= max_len+1.
    """

    def __init__(
        self,
        manifest_path: str,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        max_len: int = 350,
        strict_dtype: bool = False,
    ):
        self.pad_id = int(pad_token_id)
        self.bos_id = int(bos_token_id)
        self.eos_id = int(eos_token_id)
        self.max_len = int(max_len)
        self.strict_dtype = bool(strict_dtype)

        self.records: List[Dict[str, Any]] = []
        mp = Path(manifest_path)
        if not mp.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        with mp.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # normalize fields
                rec["indices_path"] = str(rec["indices_path"])
                # prefer explicit length if present; otherwise, allow latent_len as a hint
                if "length" in rec:
                    rec["length"] = int(rec["length"])
                elif "latent_len" in rec:
                    rec["length"] = int(rec["latent_len"])
                self.records.append(rec)

        if len(self.records) == 0:
            raise ValueError(f"manifest is empty: {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def _load_indices(self, npy_path: str) -> np.ndarray:
        arr = np.load(npy_path, allow_pickle=False)
        if arr.ndim != 1:
            raise ValueError(f"indices must be 1D array, got shape {arr.shape} at {npy_path}")
        if self.strict_dtype and arr.dtype not in (np.int16, np.int32, np.int64):
            raise TypeError(f"indices dtype must be int, got {arr.dtype} at {npy_path}")
        # use int64 for PyTorch
        return arr.astype(np.int64, copy=False)

    def raw_length(self, idx: int) -> int:
        """
        Raw sequence length L from manifest if present, otherwise load .npy header (fallback).
        """
        rec = self.records[idx]
        if "length" in rec:
            return int(rec["length"])
        # fallback: load file to query length (slower; should rarely happen)
        arr = self._load_indices(rec["indices_path"])
        return int(arr.shape[0])

    def effective_length(self, idx: int) -> int:
        """
        Effective training length after clipping and adding BOS/EOS.
        We add exactly one token (BOS or EOS), so length becomes min(L, max_len) + 1.
        """
        L = self.raw_length(idx)
        return min(L, self.max_len) + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        seq = self._load_indices(rec["indices_path"])  # int64 [L]

        # truncate raw seq to max_len
        if seq.shape[0] > self.max_len:
            seq = seq[: self.max_len]

        # inp = [BOS, z1..zL]
        # tgt = [z1..zL, EOS]
        inp = np.concatenate(([self.bos_id], seq), axis=0)
        tgt = np.concatenate((seq, [self.eos_id]), axis=0)

        item = {
            "inp": torch.from_numpy(inp.astype(np.int64, copy=False)),  # [T]
            "tgt": torch.from_numpy(tgt.astype(np.int64, copy=False)),  # [T]
        }
        # optional returns
        if "id" in rec:
            item["id"] = rec["id"]
        if "rank" in rec:
            item["rank"] = int(rec["rank"])
        return item


def collate_pad(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    """
    Pad variable-length sequences to the max length in batch.
    Inputs:
      - batch: list of dicts with keys 'inp' [T], 'tgt' [T]
    Returns:
      - inp: LongTensor [B, T_max]
      - tgt: LongTensor [B, T_max]
      - attn_mask: BoolTensor [B, T_max] (True=valid)
    """
    if len(batch) == 0:
        raise ValueError("Empty batch.")

    max_len = max(x["inp"].numel() for x in batch)
    B = len(batch)

    inp_pad = torch.full((B, max_len), int(pad_id), dtype=torch.long)
    tgt_pad = torch.full((B, max_len), int(pad_id), dtype=torch.long)
    attn_mask = torch.zeros((B, max_len), dtype=torch.bool)

    for i, ex in enumerate(batch):
        T = ex["inp"].numel()
        inp_pad[i, :T] = ex["inp"]
        tgt_pad[i, :T] = ex["tgt"]
        attn_mask[i, :T] = True

    return {"inp": inp_pad, "tgt": tgt_pad, "attn_mask": attn_mask}


# ------------------------ Optional: length-aware bucketed batching ------------------------ #

class BucketBatchSampler(Sampler[List[int]]):
    """
    Length-aware bucketed batch sampler with optional distributed sharding.

    - Buckets are defined by 'boundaries', e.g. [64,128,192,256,320,384,448]
      where lengths are dataset.effective_length(i).
    - Within each bucket we shuffle indices and then pack into batches.
    - Supports:
        * world_size / rank sharding (for torch.distributed)
        * .set_epoch(epoch) to reshuffle deterministically per epoch

    Usage:
        ds = PriorIndexDataset(...)
        sampler = BucketBatchSampler(
            dataset=ds,
            batch_size=64,
            boundaries=[64,128,192,256,320,384,448],
            seed=42,
            shuffle=True,
            drop_last=False,
            world_size=dist.get_world_size() if dist.is_initialized() else 1,
            rank=dist.get_rank() if dist.is_initialized() else 0,
        )
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=lambda b: collate_pad(b, pad_id))

    Notes:
      - This sampler yields lists of indices (batches). Use as DataLoader(..., batch_sampler=...).
      - If you use this, DO NOT also set DataLoader(..., sampler=...) at the same time.
    """

    def __init__(
        self,
        dataset: PriorIndexDataset,
        batch_size: int,
        boundaries: List[int],
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.boundaries = list(sorted(int(b) for b in boundaries))
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.world_size = max(1, int(world_size))
        self.rank = max(0, int(rank))

        # Pre-assign indices into buckets by effective length
        self._buckets: List[List[int]] = [[] for _ in range(len(self.boundaries) + 1)]
        self._assign_buckets()

        # epoch affects RNG
        self._epoch = 0

    def _assign_buckets(self):
        self._buckets = [[] for _ in range(len(self.boundaries) + 1)]
        for idx in range(len(self.dataset)):
            L = self.dataset.effective_length(idx)
            b = 0
            while b < len(self.boundaries) and L > self.boundaries[b]:
                b += 1
            self._buckets[b].append(idx)

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __len__(self) -> int:
        # Number of batches that THIS rank will yield
        # Compute total batches across all buckets, then divide by world_size (ceil/division depends on drop_last)
        total = 0
        for bucket in self._buckets:
            n = len(bucket)
            if self.shuffle:
                # approximate number of batches per bucket
                nb = n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size
            else:
                nb = n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size
            total += nb
        # shard across ranks as evenly as possible
        return (total + self.world_size - 1 - self.rank) // self.world_size

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        # Different global seed per epoch
        g.manual_seed(self.seed + self._epoch)

        # Prepare batches bucket by bucket
        all_batches: List[List[int]] = []
        for bucket in self._buckets:
            if len(bucket) == 0:
                continue
            if self.shuffle:
                idx_tensor = torch.tensor(bucket, dtype=torch.long)
                perm = torch.randperm(len(idx_tensor), generator=g)
                bucket = idx_tensor[perm].tolist()
            # chunk into batches
            for i in range(0, len(bucket), self.batch_size):
                chunk = bucket[i : i + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(chunk)

        # Optionally shuffle the list of batches (to mix buckets)
        if self.shuffle and len(all_batches) > 1:
            idx = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in idx]

        # Distributed sharding over batches (NOT over samples)
        if self.world_size > 1:
            all_batches = all_batches[self.rank :: self.world_size]

        # Yield
        for b in all_batches:
            yield b
