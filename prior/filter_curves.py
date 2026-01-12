#!/usr/bin/env python3
# coding: utf-8

"""
Filter decoded protein curves by simple geometric and secondary-structure heuristics.

Typical usage:

python prior/filter_curves.py \
  --recon_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/prior/prior_samples_1_8_new/recon \
  --out_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/prior/prior_samples_1_8_new/filtered \
  --samples_manifest /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/prior/prior_samples_1_8_new/samples_manifest.jsonl \
  --filtered_manifest_out /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/prior/prior_samples_1_8_new/filtered/manifest.jsonl \
  --min_pairwise_dist 3.0 \
  --neighbor_exclude 2 \
  --min_beta_run 0 \
  --min_beta_total 0 \
  --min_length 50
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_manifest(path: Optional[str]) -> Dict[int, dict]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        print(f"[warn] samples_manifest not found: {p}")
        return {}
    mapping: Dict[int, dict] = {}
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            idx = rec.get("i", None)
            if idx is None:
                try:
                    ipath = rec.get("indices_path", "")
                    stem = Path(ipath).stem
                    parts = stem.split("_")
                    if parts:
                        idx = int(parts[-1])
                except Exception:
                    idx = None
            if idx is None:
                continue
            mapping[int(idx)] = rec
    print(f"[info] loaded {len(mapping)} records from {p}")
    return mapping


def extract_index_from_name(name: str) -> Optional[int]:
    """
    Try to extract integer index from a file name like 'sample_prior_0003_recon.npy'.
    """
    stem = Path(name).stem
    if stem.endswith("_recon"):
        stem = stem[:-6]
    parts = stem.split("_")
    for part in reversed(parts):
        try:
            return int(part)
        except Exception:
            continue
    return None


def has_self_collision(
    coords: np.ndarray,
    min_pairwise_dist: float,
    neighbor_exclude: int,
) -> bool:
    """
    Detect self-collision by checking pairwise distances between non-neighbor points.

    coords: [L, 3]
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        return False
    L = coords.shape[0]
    if L <= neighbor_exclude + 1:
        return False

    idx = np.arange(L, dtype=np.int32)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    mask = np.abs(ii - jj) > int(neighbor_exclude)

    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)

    dist2_masked = dist2[mask]
    if dist2_masked.size == 0:
        return False

    thresh2 = float(min_pairwise_dist) * float(min_pairwise_dist)
    return bool(np.any(dist2_masked < thresh2))


def beta_stats(
    ss_one_hot: np.ndarray,
    beta_channel: int = 1,
    threshold: float = 0.5,
) -> Tuple[int, int]:
    """
    Compute total beta positions and max contiguous beta run length.

    ss_one_hot: [L, C] one-hot secondary structure (h=0, s=1, l=2 by default).
    """
    if ss_one_hot.ndim != 2 or ss_one_hot.shape[1] <= beta_channel:
        return 0, 0
    beta = ss_one_hot[:, beta_channel] > threshold
    total = int(beta.sum())
    if total == 0:
        return 0, 0
    max_run = 0
    cur = 0
    for v in beta:
        if v:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return total, max_run


def filter_curve(
    curve: np.ndarray,
    min_length: int,
    max_length: int,
    min_pairwise_dist: float,
    neighbor_exclude: int,
    min_beta_run: int,
    min_beta_total: int,
    beta_channel: int,
) -> bool:
    """
    Return True if the curve passes all filters.
    """
    if curve.ndim != 2 or curve.shape[1] < 3:
        return False

    L = curve.shape[0]
    if L < int(min_length):
        return False
    if int(max_length) > 0 and L > int(max_length):
        return False

    coords = curve[:, :3]
    if has_self_collision(coords, min_pairwise_dist=min_pairwise_dist, neighbor_exclude=neighbor_exclude):
        return False

    if curve.shape[1] >= 6 and (min_beta_run > 0 or min_beta_total > 0):
        ss_one_hot = curve[:, 3:6]
        total_beta, max_run = beta_stats(ss_one_hot, beta_channel=beta_channel)
        if min_beta_total > 0 and 0 < total_beta < min_beta_total:
            return False
        if min_beta_run > 0 and total_beta > 0 and max_run < min_beta_run:
            return False

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_dir", type=str, required=True, help="Directory with *_recon.npy curves")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory for filtered curves")
    ap.add_argument(
        "--samples_manifest",
        type=str,
        default="",
        help="Optional: original samples_manifest.jsonl produced by prior/sample_prior.py",
    )
    ap.add_argument(
        "--filtered_manifest_out",
        type=str,
        default="",
        help="Optional: path to write filtered manifest jsonl",
    )
    ap.add_argument("--min_length", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=0)
    ap.add_argument(
        "--min_pairwise_dist",
        type=float,
        default=1.0,
        help="Minimum distance between non-neighbor points; curves with any closer pairs are rejected",
    )
    ap.add_argument(
        "--neighbor_exclude",
        type=int,
        default=2,
        help="Ignore pairs with |i-j| <= neighbor_exclude in collision check",
    )
    ap.add_argument(
        "--min_beta_run",
        type=int,
        default=0,
        help="If >0, require max contiguous beta run length >= this value whenever beta exists",
    )
    ap.add_argument(
        "--min_beta_total",
        type=int,
        default=0,
        help="If >0, reject curves with 0 < total_beta < min_beta_total (requires ss channels present)",
    )
    ap.add_argument(
        "--beta_channel",
        type=int,
        default=1,
        help="Index of beta channel in ss_one_hot (default assumes order h=0, s=1, l=2)",
    )
    ap.add_argument(
        "--max_curves",
        type=int,
        default=0,
        help="Optional cap on number of accepted curves (0 = no cap)",
    )
    args = ap.parse_args()

    recon_dir = Path(args.recon_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_map = load_manifest(args.samples_manifest) if args.samples_manifest else {}

    recon_files: List[Path] = sorted(recon_dir.glob("*.npy"))
    print(f"[info] found {len(recon_files)} recon npy files in {recon_dir}")

    filtered_records: List[dict] = []
    n_total = 0
    n_kept = 0
    n_collide = 0
    n_too_short = 0
    n_ss_reject = 0

    for path in recon_files:
        curve = np.load(str(path), allow_pickle=False)
        n_total += 1

        if curve.ndim != 2 or curve.shape[1] < 3:
            continue

        L = curve.shape[0]
        if L < int(args.min_length):
            n_too_short += 1
            continue
        if int(args.max_length) > 0 and L > int(args.max_length):
            continue

        coords = curve[:, :3]
        if has_self_collision(
            coords,
            min_pairwise_dist=float(args.min_pairwise_dist),
            neighbor_exclude=int(args.neighbor_exclude),
        ):
            n_collide += 1
            continue

        ss_reject = False
        if curve.shape[1] >= 6 and (args.min_beta_run > 0 or args.min_beta_total > 0):
            ss_one_hot = curve[:, 3:6]
            total_beta, max_run = beta_stats(ss_one_hot, beta_channel=int(args.beta_channel))
            if args.min_beta_total > 0 and 0 < total_beta < args.min_beta_total:
                ss_reject = True
            if args.min_beta_run > 0 and total_beta > 0 and max_run < args.min_beta_run:
                ss_reject = True

        if ss_reject:
            n_ss_reject += 1
            continue

        idx = extract_index_from_name(path.name)
        if idx is not None and idx in manifest_map:
            rec = dict(manifest_map[idx])
        else:
            rec = {"i": int(idx) if idx is not None else n_total - 1}
        rec["recon_path"] = str(path)
        rec["length_recon"] = int(L)
        filtered_records.append(rec)

        out_path = out_dir / path.name
        if out_path != path:
            np.save(str(out_path), curve, allow_pickle=False)

        n_kept += 1
        if args.max_curves > 0 and n_kept >= int(args.max_curves):
            break

    print(f"[summary] total curves: {n_total}")
    print(f"[summary] kept: {n_kept}")
    print(f"[summary] rejected (too short): {n_too_short}")
    print(f"[summary] rejected (self-collision): {n_collide}")
    print(f"[summary] rejected (ss heuristics): {n_ss_reject}")

    if args.filtered_manifest_out:
        mpath = Path(args.filtered_manifest_out)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        with mpath.open("w") as f:
            for rec in filtered_records:
                f.write(json.dumps(rec) + "\n")
        print(f"[info] wrote filtered manifest with {len(filtered_records)} records to {mpath}")


if __name__ == "__main__":
    main()
