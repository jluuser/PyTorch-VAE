#!/usr/bin/env python3
# coding: utf-8

"""
Filter decoded protein curves by geometric and secondary-structure heuristics.

Typical usage:

python prior/filter_curves.py \
  --recon_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/prior/diffusion_prior_samples_step70001/curves_npy \
  --out_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/prior/diffusion_prior_samples_step70001/curves_npy_filtered \
  --samples_manifest /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/prior/diffusion_prior_samples_step70001/samples_manifest.jsonl \
  --filtered_manifest_out /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/prior/diffusion_prior_samples_step70001/filtered_manifest.jsonl \
  --min_pairwise_dist 2.0 \
  --neighbor_exclude 2 \
  --min_beta_run 0 \
  --min_beta_total 0 \
  --min_length 2
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ------------------------ Manifest utilities ------------------------ #


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


# ------------------------ Local backbone geometry ------------------------ #


def bond_length_stats(
    coords: np.ndarray,
    good_min: float = 3.5,
    good_max: float = 4.2,
) -> Dict[str, float]:
    """
    Compute statistics of Cα–Cα bond lengths.

    Returns a dict with keys:
      - num: number of bonds (L-1)
      - mean, std, min, max
      - frac_out: fraction of bonds outside [good_min, good_max]
    """
    L = coords.shape[0]
    if L < 2:
        return {
            "num": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "frac_out": 0.0,
        }

    diffs = coords[1:] - coords[:-1]  # [L-1, 3]
    dists = np.linalg.norm(diffs, axis=-1)  # [L-1]

    mean = float(dists.mean())
    std = float(dists.std())
    dmin = float(dists.min())
    dmax = float(dists.max())
    frac_out = float(np.mean((dists < good_min) | (dists > good_max)))

    return {
        "num": int(dists.shape[0]),
        "mean": mean,
        "std": std,
        "min": dmin,
        "max": dmax,
        "frac_out": frac_out,
    }


def bond_angle_stats(
    coords: np.ndarray,
    good_min_deg: float = 80.0,
    good_max_deg: float = 150.0,
) -> Dict[str, float]:
    """
    Compute statistics of Cα–Cα–Cα bond angles in degrees.

    Returns a dict with keys:
      - num: number of angles (L-2)
      - mean, std, min, max
      - frac_out: fraction of angles outside [good_min_deg, good_max_deg]
    """
    L = coords.shape[0]
    if L < 3:
        return {
            "num": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "frac_out": 0.0,
        }

    p0 = coords[:-2]      # [L-2, 3]
    p1 = coords[1:-1]     # [L-2, 3]
    p2 = coords[2:]       # [L-2, 3]

    v1 = p0 - p1          # [L-2, 3]
    v2 = p2 - p1          # [L-2, 3]

    v1_norm = np.linalg.norm(v1, axis=-1)
    v2_norm = np.linalg.norm(v2, axis=-1)
    denom = v1_norm * v2_norm

    mask = denom > 1e-6
    if not np.any(mask):
        return {
            "num": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "frac_out": 0.0,
        }

    cos_theta = np.zeros_like(denom, dtype=np.float64)
    cos_theta[mask] = np.einsum("ij,ij->i", v1[mask], v2[mask]) / denom[mask]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_theta[mask]))  # [N_valid]

    mean = float(angles.mean())
    std = float(angles.std())
    amin = float(angles.min())
    amax = float(angles.max())
    frac_out = float(np.mean((angles < good_min_deg) | (angles > good_max_deg)))

    return {
        "num": int(angles.shape[0]),
        "mean": mean,
        "std": std,
        "min": amin,
        "max": amax,
        "frac_out": frac_out,
    }


def radius_of_gyration(coords: np.ndarray) -> float:
    """
    Radius of gyration of Cα coordinates.
    """
    if coords.ndim != 2 or coords.shape[0] == 0:
        return 0.0
    center = coords.mean(axis=0)
    diff = coords - center
    rg2 = np.mean(np.sum(diff * diff, axis=-1))
    return float(np.sqrt(max(rg2, 0.0)))


# ------------------------ Collision / self-intersection ------------------------ #


def self_collision_stats(
    coords: np.ndarray,
    min_pairwise_dist: float,
    neighbor_exclude: int,
) -> int:
    """
    Count self-collision pairs based on point-to-point distances between non-neighbor Cα atoms.
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        return 0
    L = coords.shape[0]
    if L <= neighbor_exclude + 1:
        return 0

    idx = np.arange(L, dtype=np.int32)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    mask = np.abs(ii - jj) > int(neighbor_exclude)

    diff = coords[:, None, :] - coords[None, :, :]  # [L, L, 3]
    dist2 = np.sum(diff * diff, axis=-1)            # [L, L]

    thresh2 = float(min_pairwise_dist) * float(min_pairwise_dist)
    hit_mask = mask & (dist2 < thresh2)
    return int(hit_mask.sum())


def has_self_collision(
    coords: np.ndarray,
    min_pairwise_dist: float,
    neighbor_exclude: int,
) -> bool:
    """
    Boolean wrapper around self_collision_stats.
    """
    return self_collision_stats(coords, min_pairwise_dist, neighbor_exclude) > 0


def segment_self_clash_count(
    coords: np.ndarray,
    min_seg_dist: float = 2.5,
    neighbor_exclude_segments: int = 1,
    num_samples: int = 5,
) -> int:
    """
    Count approximate segment-segment self-intersections based on sampled distances.
    Ignore segments that share endpoints or are immediate neighbors (controlled by neighbor_exclude_segments).
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        return 0
    L = coords.shape[0]
    if L < 3:
        return 0

    n_seg = L - 1
    thresh2 = float(min_seg_dist) * float(min_seg_dist)

    t_vals = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)

    count = 0
    for i in range(n_seg):
        p0 = coords[i]
        p1 = coords[i + 1]
        pts1 = p0[None, :] + (p1 - p0)[None, :] * t_vals[:, None]  # [S,3]

        for j in range(i + 1 + neighbor_exclude_segments, n_seg):
            q0 = coords[j]
            q1 = coords[j + 1]
            pts2 = q0[None, :] + (q1 - q0)[None, :] * t_vals[:, None]  # [S,3]

            diff = pts1[:, None, :] - pts2[None, :, :]  # [S,S,3]
            dist2 = np.sum(diff * diff, axis=-1)
            if np.any(dist2 < thresh2):
                count += 1

    return count


# ------------------------ Secondary structure utilities ------------------------ #


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


def beta_strand_and_sheet_stats(
    coords: np.ndarray,
    ss_one_hot: np.ndarray,
    beta_channel: int = 1,
    threshold: float = 0.5,
    neighbor_exclude: int = 2,
    min_strand_len: int = 3,
    sheet_min_dist: float = 4.0,
    sheet_max_dist: float = 6.0,
) -> Dict[str, float]:
    """
    Compute statistics about beta strands and whether they participate in sheet-like contacts.

    Returns a dict with keys:
      - beta_total
      - beta_in_sheet
      - beta_sheet_fraction
      - n_strands_total
      - n_sheet_strands
      - n_isolated_strands
    """
    L = ss_one_hot.shape[0]
    if ss_one_hot.ndim != 2 or ss_one_hot.shape[1] <= beta_channel or L == 0:
        return {
            "beta_total": 0,
            "beta_in_sheet": 0,
            "beta_sheet_fraction": 0.0,
            "n_strands_total": 0,
            "n_sheet_strands": 0,
            "n_isolated_strands": 0,
        }

    beta_mask = ss_one_hot[:, beta_channel] > threshold
    beta_total = int(beta_mask.sum())
    if beta_total == 0:
        return {
            "beta_total": 0,
            "beta_in_sheet": 0,
            "beta_sheet_fraction": 0.0,
            "n_strands_total": 0,
            "n_sheet_strands": 0,
            "n_isolated_strands": 0,
        }

    # Find beta runs (strands) with length >= min_strand_len
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < L:
        if beta_mask[i]:
            j = i
            while j + 1 < L and beta_mask[j + 1]:
                j += 1
            run_len = j - i + 1
            if run_len >= min_strand_len:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1

    n_strands_total = len(runs)
    if n_strands_total == 0:
        return {
            "beta_total": beta_total,
            "beta_in_sheet": 0,
            "beta_sheet_fraction": 0.0,
            "n_strands_total": 0,
            "n_sheet_strands": 0,
            "n_isolated_strands": 0,
        }

    # Compute sheet-like partners using distance between beta residues
    beta_idx = np.nonzero(beta_mask)[0]
    beta_coords = coords[beta_idx]  # [B, 3]
    B = beta_coords.shape[0]
    if B == 0:
        return {
            "beta_total": beta_total,
            "beta_in_sheet": 0,
            "beta_sheet_fraction": 0.0,
            "n_strands_total": n_strands_total,
            "n_sheet_strands": 0,
            "n_isolated_strands": n_strands_total,
        }

    # Pairwise distances among beta residues
    diff = beta_coords[:, None, :] - beta_coords[None, :, :]  # [B,B,3]
    dist2 = np.sum(diff * diff, axis=-1)
    dist = np.sqrt(dist2)

    # Sequence distance constraints
    beta_idx_mat_i = beta_idx[:, None]
    beta_idx_mat_j = beta_idx[None, :]
    seq_diff = np.abs(beta_idx_mat_i - beta_idx_mat_j)

    # Sheet-like adjacency: within [sheet_min_dist, sheet_max_dist] and not close in sequence
    sheet_mask = (
        (dist >= float(sheet_min_dist))
        & (dist <= float(sheet_max_dist))
        & (seq_diff > int(neighbor_exclude))
    )
    np.fill_diagonal(sheet_mask, False)

    # For each beta residue, does it have at least one sheet-like partner?
    beta_has_partner = sheet_mask.any(axis=1)
    beta_sheet_mask = np.zeros(L, dtype=bool)
    beta_sheet_mask[beta_idx] = beta_has_partner

    beta_in_sheet = int(beta_sheet_mask.sum())
    beta_sheet_fraction = float(beta_in_sheet) / float(beta_total) if beta_total > 0 else 0.0

    # Classify strands as sheet or isolated
    n_sheet_strands = 0
    n_isolated_strands = 0
    for (start, end) in runs:
        strand_mask = beta_sheet_mask[start : end + 1]
        if strand_mask.any():
            n_sheet_strands += 1
        else:
            n_isolated_strands += 1

    return {
        "beta_total": beta_total,
        "beta_in_sheet": beta_in_sheet,
        "beta_sheet_fraction": beta_sheet_fraction,
        "n_strands_total": n_strands_total,
        "n_sheet_strands": n_sheet_strands,
        "n_isolated_strands": n_isolated_strands,
    }


# ------------------------ Main filtering logic ------------------------ #


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
    ap.add_argument(
        "--min_beta_sheet_fraction",
        type=float,
        default=0.0,
        help="If >0, require at least this fraction of beta residues to have sheet-like partners",
    )
    ap.add_argument(
        "--max_isolated_beta_strands",
        type=int,
        default=-1,
        help="If >=0, reject curves with more isolated beta strands than this (strands with no sheet-like partners)",
    )
    ap.add_argument(
        "--min_strand_len",
        type=int,
        default=3,
        help="Minimum length of a beta run to be treated as a strand in sheet/isolated statistics",
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
    n_too_short = 0
    n_too_long = 0
    n_geom_bond = 0
    n_geom_angle = 0
    n_collide = 0
    n_seg_collide = 0
    n_ss_reject = 0

    # Hard thresholds (can be adjusted here if needed)
    BOND_MIN_ALLOWED = 1.0
    BOND_MAX_ALLOWED = 9.5
    BOND_GOOD_MIN = 2.0
    BOND_GOOD_MAX = 7.2
    BOND_FRAC_OUT_MAX = 0.90

    ANGLE_MIN_ALLOWED = 10.0
    ANGLE_MAX_ALLOWED = 180.0
    ANGLE_GOOD_MIN = 30.0
    ANGLE_GOOD_MAX = 180.0
    ANGLE_FRAC_OUT_MAX = 0.90

    SEG_MIN_DIST = 1.3
    SEG_NEIGHBOR_EXCLUDE = 1

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
            n_too_long += 1
            continue

        coords = curve[:, :3]

        # Local bond length sanity
        bl_stats = bond_length_stats(coords, good_min=BOND_GOOD_MIN, good_max=BOND_GOOD_MAX)
        if bl_stats["num"] > 0:
            if (
                bl_stats["min"] < BOND_MIN_ALLOWED
                or bl_stats["max"] > BOND_MAX_ALLOWED
                or bl_stats["frac_out"] > BOND_FRAC_OUT_MAX
            ):
                n_geom_bond += 1
                continue

        # Local bond angle sanity
        ba_stats = bond_angle_stats(coords, good_min_deg=ANGLE_GOOD_MIN, good_max_deg=ANGLE_GOOD_MAX)
        if ba_stats["num"] > 0:
            if (
                ba_stats["min"] < ANGLE_MIN_ALLOWED
                or ba_stats["max"] > ANGLE_MAX_ALLOWED
                or ba_stats["frac_out"] > ANGLE_FRAC_OUT_MAX
            ):
                n_geom_angle += 1
                continue

        # Point-based self-collision
        if has_self_collision(
            coords,
            min_pairwise_dist=float(args.min_pairwise_dist),
            neighbor_exclude=int(args.neighbor_exclude),
        ):
            n_collide += 1
            continue

        # Segment-segment approximate self-intersection
        seg_clashes = segment_self_clash_count(
            coords,
            min_seg_dist=SEG_MIN_DIST,
            neighbor_exclude_segments=SEG_NEIGHBOR_EXCLUDE,
            num_samples=5,
        )
        if seg_clashes > 0:
            n_seg_collide += 1
            continue

        ss_reject = False
        beta_total = 0
        beta_max_run = 0
        beta_sheet_fraction = 0.0
        beta_in_sheet = 0
        n_strands_total = 0
        n_sheet_strands = 0
        n_isolated_strands = 0

        if curve.shape[1] >= 6:
            ss_one_hot = curve[:, 3:6]

            # Basic beta stats (total and max run length)
            beta_total, beta_max_run = beta_stats(
                ss_one_hot, beta_channel=int(args.beta_channel)
            )

            if args.min_beta_total > 0 and 0 < beta_total < args.min_beta_total:
                ss_reject = True

            if args.min_beta_run > 0 and beta_total > 0 and beta_max_run < args.min_beta_run:
                ss_reject = True

            # Strand-level and sheet-level beta stats
            strand_stats = beta_strand_and_sheet_stats(
                coords=coords,
                ss_one_hot=ss_one_hot,
                beta_channel=int(args.beta_channel),
                threshold=0.5,
                neighbor_exclude=int(args.neighbor_exclude),
                min_strand_len=int(args.min_strand_len),
                sheet_min_dist=4.0,
                sheet_max_dist=6.0,
            )
            beta_sheet_fraction = float(strand_stats["beta_sheet_fraction"])
            beta_in_sheet = int(strand_stats["beta_in_sheet"])
            n_strands_total = int(strand_stats["n_strands_total"])
            n_sheet_strands = int(strand_stats["n_sheet_strands"])
            n_isolated_strands = int(strand_stats["n_isolated_strands"])

            if args.min_beta_sheet_fraction > 0.0 and beta_total > 0:
                if beta_sheet_fraction < float(args.min_beta_sheet_fraction):
                    ss_reject = True

            if int(args.max_isolated_beta_strands) >= 0:
                if n_isolated_strands > int(args.max_isolated_beta_strands):
                    ss_reject = True

        if ss_reject:
            n_ss_reject += 1
            continue

        # Radius of gyration (stored for analysis, not used as a hard filter)
        rg = radius_of_gyration(coords)

        # Build record, merging original manifest info if available
        idx = extract_index_from_name(path.name)
        if idx is not None and idx in manifest_map:
            rec = dict(manifest_map[idx])
        else:
            rec = {"i": int(idx) if idx is not None else n_total - 1}

        rec["recon_path"] = str(path)
        rec["length_recon"] = int(L)
        rec["rg"] = float(rg)

        # Attach geometry stats for downstream analysis
        rec["bond_mean"] = float(bl_stats["mean"])
        rec["bond_std"] = float(bl_stats["std"])
        rec["bond_min"] = float(bl_stats["min"])
        rec["bond_max"] = float(bl_stats["max"])
        rec["bond_frac_out"] = float(bl_stats["frac_out"])

        rec["angle_mean"] = float(ba_stats["mean"])
        rec["angle_std"] = float(ba_stats["std"])
        rec["angle_min"] = float(ba_stats["min"])
        rec["angle_max"] = float(ba_stats["max"])
        rec["angle_frac_out"] = float(ba_stats["frac_out"])

        rec["beta_total"] = int(beta_total)
        rec["beta_max_run"] = int(beta_max_run)
        rec["beta_in_sheet"] = int(beta_in_sheet)
        rec["beta_sheet_fraction"] = float(beta_sheet_fraction)
        rec["beta_strands_total"] = int(n_strands_total)
        rec["beta_strands_sheet"] = int(n_sheet_strands)
        rec["beta_strands_isolated"] = int(n_isolated_strands)

        rec["n_self_clash_pairs"] = int(
            self_collision_stats(
                coords,
                min_pairwise_dist=float(args.min_pairwise_dist),
                neighbor_exclude=int(args.neighbor_exclude),
            )
        )
        rec["n_seg_clash_pairs"] = int(seg_clashes)

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
    print(f"[summary] rejected (too long): {n_too_long}")
    print(f"[summary] rejected (bond length out-of-range): {n_geom_bond}")
    print(f"[summary] rejected (bond angle out-of-range): {n_geom_angle}")
    print(f"[summary] rejected (point self-collision): {n_collide}")
    print(f"[summary] rejected (segment self-intersection): {n_seg_collide}")
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
