# -*- coding: utf-8 -*-
"""
Dataset and DataModule for protein-like curve sequences (centered only, no global normalization).
Compatible with VQVAE pipeline (expects .npy dicts with 'curve_coords' & 'ss_one_hot').

Features:
- Per-curve centering (translation-invariant, retains Å scale)
- Variable-length padding + boolean mask
- Compatible with LightningDataModule & new-style arguments (list_path / train_list)
"""

import os
import torch
import numpy as np
from pathlib import Path
import numpy.core as _np_core
import sys as _sys
_sys.modules.setdefault("numpy._core", _np_core)
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningDataModule
from typing import List, Optional, Tuple

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Collate function
# --------------------------------------------------------------------------- #
def pad_collate(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads variable-length tensors in a batch.
    Args:
        batch: list of [L_i, 6] float tensors
    Returns:
        padded_batch: [B, L_max, 6] float32 (zero-padded)
        mask:         [B, L_max] bool (True=valid, False=pad)
    """
    if len(batch) == 0:
        raise RuntimeError("Empty batch given to pad_collate.")
    lengths = [int(x.size(0)) for x in batch]
    padded_batch = pad_sequence(batch, batch_first=True)
    max_len = int(padded_batch.size(1))

    mask = torch.zeros(len(lengths), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        if l > 0:
            mask[i, :l] = True
    return padded_batch, mask


# --------------------------------------------------------------------------- #
# Dataset class
# --------------------------------------------------------------------------- #
class CurveDataset(Dataset):
    """
    Loads per-curve .npy dicts with keys:
      - "curve_coords": [L, 3] float32, raw coordinates (Å)
      - "ss_one_hot":   [L, 3] float32, one-hot secondary structure

    Normalization:
      Per-curve centering (remove translation). No global scaling.
    """

    def __init__(
        self,
        npy_dir: str,
        list_path: Optional[str] = None,
        list_file: Optional[str] = None,  # backward compatibility
        train: bool = True,
    ):
        """
        Args:
            npy_dir: directory of .npy files
            list_path or list_file: txt file listing filenames to load
            train: bool, used for logging only
        """
        super().__init__()
        self.npy_dir = Path(npy_dir)
        self.train = train

        # Accept both 'list_path' and 'list_file'
        list_txt = list_path or list_file
        if list_txt is None:
            raise ValueError("CurveDataset requires a valid list_path or list_file.")

        # load file list
        with open(list_txt, "r") as f:
            rels = [line.strip() for line in f if line.strip()]
        self.file_paths = [os.path.join(self.npy_dir, p) for p in rels]
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No files found using list file: {list_txt}")

        self._debug_printed = False
        if self.train:
            print(f"[Dataset] Train set: {len(self.file_paths)} curves from {self.npy_dir}")
        else:
            print(f"[Dataset] Val set: {len(self.file_paths)} curves from {self.npy_dir}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.file_paths[idx]
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            data = {k: data[k] for k in data.files}
        else:
            data = data.item()

        coords = np.asarray(data["curve_coords"], dtype=np.float32)  # [L,3]
        ss_one_hot = np.asarray(data["ss_one_hot"], dtype=np.float32)  # [L,3]

        # sanity checks
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Bad coords shape at {path}: {coords.shape}")
        if ss_one_hot.ndim != 2 or ss_one_hot.shape != coords.shape:
            raise ValueError(f"Bad ss_one_hot shape at {path}: {ss_one_hot.shape}")

        # Per-curve centering
        mean = coords.mean(axis=0, keepdims=True)
        coords_centered = coords - mean

        # Debug print (first curve only)
        if not self._debug_printed and idx == 0:
            cm = coords_centered.mean(axis=0)
            cs = coords_centered.std(axis=0)
            l1_err = np.abs(ss_one_hot.sum(axis=1) - 1.0).mean()
            print(f"[DS-CHK] mean (pre-center): {mean.reshape(-1).tolist()}")
            print(f"[DS-CHK] coords_centered mean: {cm.tolist()} std: {cs.tolist()}")
            print(f"[DS-CHK] ss_one_hot L1 error: {float(l1_err):.6f}")
            self._debug_printed = True

        # Combine and sanitize
        full_input = np.concatenate([coords_centered, ss_one_hot], axis=-1).astype(np.float32)
        if not np.isfinite(full_input).all():
            full_input = np.nan_to_num(full_input, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(full_input)


# --------------------------------------------------------------------------- #
# Lightning DataModule
# --------------------------------------------------------------------------- #
class CurveDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for protein curve sequences.
    """

    def __init__(
        self,
        npy_dir: str,
        train_list: str,
        val_list: str,
        train_batch_size: int = 256,
        val_batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.npy_dir = npy_dir
        self.train_list = os.path.join(npy_dir, train_list)
        self.val_list = os.path.join(npy_dir, val_list)
        self.train_batch_size = int(train_batch_size)
        self.val_batch_size = int(val_batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

    def setup(self, stage: Optional[str] = None):
        """Initialize train and val datasets."""
        self.train_dataset = CurveDataset(
            npy_dir=self.npy_dir,
            list_path=self.train_list,
            train=True,
        )
        self.val_dataset = CurveDataset(
            npy_dir=self.npy_dir,
            list_path=self.val_list,
            train=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=pad_collate,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=pad_collate,
            drop_last=False,
        )
