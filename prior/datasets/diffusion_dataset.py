# coding: utf-8

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class DiffusionIndexDataset(Dataset):
    """
    Dataset for diffusion prior training.

    It supports two modes, controlled by `use_latent`:
      - indices mode (use_latent=False): load flattened RVQ indices from "indices_path".
      - latent  mode (use_latent=True) : load continuous latents from "latent_path" (z_e).

    Manifest JSONL records may contain:
      - indices_path: path to [.npy] with flattened indices [N_flat]
      - latent_path:  path to [.npy] with latents [M, D]
      - latent_len:   flattened length (optional, for indices mode)
      - latent_tokens: number of latent tokens M (optional, for latent mode)
      - target_len:   original curve length L
      - geo_path:     optional per-position geometry [.npy]
      - geo_dim:      dimension of geometry features
    """

    def __init__(
        self,
        manifest_path: str,
        pad_token_id: int,
        max_len: Optional[int] = None,
        load_geo: bool = False,
        use_latent: bool = False,
    ):
        self.manifest_path = str(manifest_path)
        self.pad_token_id = int(pad_token_id)
        self.max_len = int(max_len) if max_len is not None else None
        self.load_geo = bool(load_geo)
        self.use_latent = bool(use_latent)

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
        """
        Return the raw latent sequence length for item idx.

        For latent mode, prefers "latent_tokens" or infers from latent_path.
        For indices mode, prefers "latent_len" or infers from indices_path.
        """
        rec = self.records[idx]

        if self.use_latent or ("latent_path" in rec):
            L = rec.get("latent_tokens", None)
            if L is not None:
                return int(L)
            p_lat = rec.get("latent_path", "")
            if p_lat and Path(p_lat).is_file():
                arr = np.load(p_lat, allow_pickle=False)
                if arr.ndim >= 2:
                    return int(arr.shape[0])
                return int(arr.reshape(-1).shape[0])
            return 0

        # indices mode
        L = rec.get("latent_len", None)
        if L is not None:
            return int(L)
        p_idx = rec.get("indices_path", "")
        if p_idx and Path(p_idx).is_file():
            arr = np.load(p_idx, allow_pickle=False)
            return int(arr.reshape(-1).shape[0])
        return 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        # latent mode (z_e)
        if self.use_latent or ("latent_path" in rec):
            p_lat = rec.get("latent_path", None)
            if p_lat is None:
                raise KeyError("Record missing 'latent_path' for latent mode.")
            arr = np.load(p_lat, allow_pickle=False)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if self.max_len is not None:
                arr = arr[: self.max_len]
            latent = torch.from_numpy(arr.astype(np.float32, copy=False))
            mask = torch.ones(latent.shape[0], dtype=torch.bool)

            out: Dict[str, Any] = {
                "latent": latent,  # [M, D]
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

        # indices mode (original behavior)
        p = rec["indices_path"]
        arr = np.load(p, allow_pickle=False).reshape(-1)
        if self.max_len is not None:
            arr = arr[: self.max_len]
        indices = torch.from_numpy(arr.astype(np.int64, copy=False))
        mask = torch.ones(indices.shape[0], dtype=torch.bool)

        out = {
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
    use_latent: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for diffusion prior batches.

    If use_latent is True:
      - pads "latent" tensors [M_i, D] to [B, M_max, D].
    Else:
      - pads "indices" tensors [N_i] to [B, N_max] with pad_id.

    Geometry ("geo") is padded along the sequence dimension and truncated in feature dim.
    """
    pad_id = int(pad_id)
    B = len(batch)

    # Decide mode from flag; if not set, infer from first item
    if use_latent is None:
        use_latent = "latent" in batch[0]

    if use_latent or ("latent" in batch[0]):
        # Latent mode
        max_len = 0
        code_dim = None
        for item in batch:
            lat = item["latent"]
            max_len = max(max_len, int(lat.size(0)))
            if code_dim is None:
                code_dim = int(lat.size(-1))
        if code_dim is None:
            raise RuntimeError("Unable to infer latent code_dim from batch.")

        indices_pad = torch.zeros((B, max_len, code_dim), dtype=torch.float32)
        mask_pad = torch.zeros((B, max_len), dtype=torch.bool)
        target_len = torch.zeros((B,), dtype=torch.long)

        has_geo = any(("geo" in item) and (item["geo"] is not None) for item in batch)
        geo_pad = None
        if has_geo and int(geo_dim) > 0:
            geo_pad = torch.zeros((B, max_len, int(geo_dim)), dtype=torch.float32)

        for i, item in enumerate(batch):
            lat = item["latent"]
            L = int(lat.size(0))
            indices_pad[i, :L] = lat
            mask_pad[i, :L] = True
            target_len[i] = int(item.get("target_len", 0))

            if geo_pad is not None:
                g = item.get("geo", None)
                if g is not None:
                    g = g[:L]
                    gd = min(int(g.size(-1)), int(geo_dim))
                    geo_pad[i, :L, :gd] = g[:, :gd]

        out: Dict[str, torch.Tensor] = {
            "latent": indices_pad,
            "mask": mask_pad,
            "target_len": target_len,
        }
        if geo_pad is not None:
            out["geo"] = geo_pad
        return out

    # Indices mode (original)
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
