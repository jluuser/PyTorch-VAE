#!/usr/bin/env python3
# coding: utf-8
"""
Convert curve file to standard dataset npy dict:
  - curve_coords: [L,3] float32
  - ss_one_hot  : [L,3] float32 (H/E/C)

Accepted inputs:
  1) Text (CSV/whitespace) with rows: x,y,z,label   (-1=H, 0=C, 1=E)
  2) .npy (numpy) file:
     - dict saved via np.save (possibly as 0-d object ndarray)
     - ndarray [L,4] (x,y,z,label) or [L,6] (x,y,z,ss1,ss2,ss3)

Output is a real dataset dict (saved via np.save with allow_pickle=True).
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Union, Any

# -------------------- helpers --------------------

def _unwrap_object0d(obj: Any) -> Any:
    """If obj is a 0-d object ndarray, unwrap with .item(); otherwise return as-is."""
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        try:
            return obj.item()
        except Exception:
            return obj
    return obj

def _try_load_text(path: str) -> np.ndarray:
    """Load a text file with either comma-separated or whitespace-separated values."""
    p = Path(path)
    # prefer CSV with comma
    try:
        arr = np.loadtxt(str(p), delimiter=",", dtype=np.float32)
    except Exception:
        # fallback: whitespace
        arr = np.loadtxt(str(p), dtype=np.float32)

    if arr.ndim == 1:
        if arr.size not in (4, 6):
            raise ValueError(f"Expected 4 or 6 columns, got {arr.size}")
        arr = arr.reshape(1, -1)

    if arr.ndim != 2 or arr.shape[1] not in (4, 6):
        raise ValueError(f"Expected shape [L,4] or [L,6] from text, got {arr.shape}")
    return arr.astype(np.float32)

def _try_load_npy(path: str) -> Any:
    """Load a numpy npy file (dict or ndarray)."""
    obj = np.load(path, allow_pickle=True)
    obj = _unwrap_object0d(obj)
    return obj

def _labels_to_one_hot(labels: np.ndarray) -> np.ndarray:
    """Map -1->H, 1->E, 0->C to one-hot order H/E/C."""
    labels = np.rint(labels).astype(np.int32)
    L = labels.shape[0]
    oh = np.zeros((L, 3), dtype=np.float32)  # H,E,C
    oh[labels == -1, 0] = 1.0  # H
    oh[labels ==  1, 1] = 1.0  # E
    oh[labels ==  0, 2] = 1.0  # C
    unknown = ~((labels == -1) | (labels == 0) | (labels == 1))
    if np.any(unknown):
        raise ValueError(f"Unknown label values: {np.unique(labels[unknown])} (expect -1/0/1)")
    return oh

def _to_one_hot_from_any(ss: np.ndarray) -> np.ndarray:
    """If already one-hot, keep; else argmax -> one-hot. Expect shape [L,3]."""
    ss = ss.astype(np.float32)
    if ss.ndim != 2 or ss.shape[1] != 3:
        raise ValueError(f"SS must be [L,3], got {ss.shape}")
    sums = ss.sum(axis=-1)
    is_binary = np.all((ss == 0.0) | (ss == 1.0))
    if is_binary and np.allclose(sums, 1.0):
        return ss
    idx = np.argmax(ss, axis=-1)
    L = ss.shape[0]
    oh = np.zeros((L, 3), dtype=np.float32)
    oh[np.arange(L), idx] = 1.0
    return oh

def _from_ndarray(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Parse ndarray -> (coords [L,3], ss_one_hot [L,3])."""
    if arr.ndim == 1 and arr.size in (4, 6):
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] not in (4, 6):
        raise ValueError(f"Unsupported array shape {arr.shape}. Need [L,4] or [L,6].")

    if arr.shape[1] == 4:
        coords = arr[:, :3].astype(np.float32)
        labels = arr[:, 3].astype(np.float32)
        ss_one_hot = _labels_to_one_hot(labels)
    else:
        coords = arr[:, :3].astype(np.float32)
        ss = arr[:, 3:].astype(np.float32)
        ss_one_hot = _to_one_hot_from_any(ss)
    return coords, ss_one_hot

# -------------------- main convert --------------------

def convert(src: str, dst: str):
    p = Path(src)

    # 1) Try as text first (你的当前数据就是文本，即使扩展名是 .npy)
    arr = None
    try:
        arr = _try_load_text(str(p))
    except Exception:
        arr = None

    if arr is not None:
        coords, ss_one_hot = _from_ndarray(arr)
    else:
        # 2) Fall back to real npy load (dict or ndarray, possibly 0-d object wrapper)
        obj = _try_load_npy(str(p))
        if isinstance(obj, dict):
            if "curve_coords" not in obj or "ss_one_hot" not in obj:
                raise ValueError("Dict missing 'curve_coords' and 'ss_one_hot'.")
            coords = np.asarray(obj["curve_coords"], dtype=np.float32)
            ss_one_hot = np.asarray(obj["ss_one_hot"], dtype=np.float32)
        elif isinstance(obj, np.ndarray):
            coords, ss_one_hot = _from_ndarray(obj)
        else:
            raise ValueError(f"Unsupported input type from npy: {type(obj)}")

    # Final sanity
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be [L,3], got {coords.shape}")
    if ss_one_hot.ndim != 2 or ss_one_hot.shape[1] != 3:
        raise ValueError(f"ss_one_hot must be [L,3], got {ss_one_hot.shape}")
    if coords.shape[0] != ss_one_hot.shape[0]:
        raise ValueError("coords and ss_one_hot length mismatch.")

    out = {
        "curve_coords": coords.astype(np.float32),
        "ss_one_hot":   ss_one_hot.astype(np.float32),
    }
    np.save(dst, out, allow_pickle=True)

    idx = np.argmax(out["ss_one_hot"], axis=-1)
    nH = int((idx == 0).sum()); nE = int((idx == 1).sum()); nC = int((idx == 2).sum())
    print(f"Saved dataset dict: {dst}  (L={coords.shape[0]})")
    print(f"Counts -> H:{nH}, E:{nE}, C:{nC}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=str, help="Path to curve file (text CSV or .npy).")
    ap.add_argument("--dst", required=True, type=str, help="Output path for standard dict .npy")
    args = ap.parse_args()
    convert(args.src, args.dst)

if __name__ == "__main__":
    main()
