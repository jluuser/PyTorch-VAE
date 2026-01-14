# coding: utf-8

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class DiffusionIndexDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        pad_token_id: int,
        max_len: Optional[int] = None,
        load_geo: bool = False,
    ):
        self.manifest_path = str(manifest_path)
        self.pad_token_id = int(pad_token_id)
        self.max_len = int(max_len) if max_len is not None else None
        self.load_geo = bool(load_geo)

        path = Path(self.manifest_path)
        if not path.is_file():
            raise FileNotFoundError(self.manifest_path)

        self.records: List[Dict[str, Any]] = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

        if not self.records:
            raise RuntimeError(f"empty manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def raw_length(self, idx: int) -> int:
        rec = self.records[idx]
        L = rec.get("latent_len", None)
        if L is None:
            p = rec.get("indices_path")
            if p and Path(p).is_file():
                arr = np.load(p, allow_pickle=False)
                L = int(arr.reshape(-1).shape[0])
            else:
                L = 0
        return int(L)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        p = rec["indices_path"]
        arr = np.load(p, allow_pickle=False).reshape(-1)
        if self.max_len is not None:
            arr = arr[: self.max_len]
        indices = torch.from_numpy(arr.astype(np.int64, copy=False))
        mask = torch.ones(indices.shape[0], dtype=torch.bool)

        out: Dict[str, Any] = {
            "indices": indices,
            "mask": mask,
            "target_len": int(rec.get("target_len", rec.get("length", 0)) or 0),
        }

        if self.load_geo:
            geo_path = rec.get("geo_path", "")
            if geo_path and Path(geo_path).is_file():
                geo = np.load(geo_path, allow_pickle=False)
                if self.max_len is not None:
                    geo = geo[: self.max_len]
                out["geo"] = torch.from_numpy(geo.astype(np.float32, copy=False))
            else:
                out["geo"] = None

        return out


def collate_pad(
    batch: List[Dict[str, Any]],
    pad_id: int,
    geo_dim: int = 0,
    multiple_of: int = 1,
) -> Dict[str, torch.Tensor]:
    pad_id = int(pad_id)
    B = len(batch)
    max_len = 0
    for item in batch:
        max_len = max(max_len, int(item["indices"].numel()))

    multiple_of = max(1, int(multiple_of))
    if max_len % multiple_of != 0:
        max_len = ((max_len + multiple_of - 1) // multiple_of) * multiple_of

    indices_pad = torch.full((B, max_len), pad_id, dtype=torch.long)
    mask_pad = torch.zeros((B, max_len), dtype=torch.bool)
    target_len = torch.zeros((B,), dtype=torch.long)

    has_geo = any(("geo" in item) and (item["geo"] is not None) for item in batch)
    geo_pad = None
    if has_geo and int(geo_dim) > 0:
        geo_pad = torch.zeros((B, max_len, int(geo_dim)), dtype=torch.float32)

    for i, item in enumerate(batch):
        idx = item["indices"]
        L = int(idx.numel())
        indices_pad[i, :L] = idx
        mask_pad[i, :L] = True
        target_len[i] = int(item.get("target_len", 0))

        if geo_pad is not None:
            g = item.get("geo", None)
            if g is not None:
                g = g[:L]
                gd = min(int(g.size(-1)), int(geo_dim))
                geo_pad[i, :L, :gd] = g[:, :gd]

    out = {"indices": indices_pad, "mask": mask_pad, "target_len": target_len}
    if geo_pad is not None:
        out["geo"] = geo_pad
    return out
