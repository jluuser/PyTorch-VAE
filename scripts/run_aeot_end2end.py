#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end AEOT inference pipeline (single-file, ONE-SHOT):

OT sample latents  ->  AE decode to curves  ->  filter (one pass)  ->  save outputs

Key behavior (per your requirement):
  - Generate a fixed number of candidate curves (n_generate) ONCE.
  - Run filtering ONCE (no iterative re-generation to reach some kept target).
  - Return however many pass the filter.

Outputs:
  run_dir/
    filtered_npy/                 # final accepted curves as npy [L,6]
    filtered_manifest.jsonl        # json per line with geometry/ss stats
    summary.json                   # timings and rejection stats
    args.json

Example:
python scripts/run_aeot_end2end.py \
  --ae_config configs/stage1_ae.yaml \
  --ae_ckpt /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/checkpoints/aeot_sigmoid/epochepoch=epoch=089.ckpt \
  --features_pt /public/home/zhangyangroup/chengshiz/keyuan.zhou/AE-OT/results_curves/features_5w.pt \
  --ot_h /public/home/zhangyangroup/chengshiz/keyuan.zhou/AE-OT/results_curves/h.pt \
  --out_root results/aeot_runs \
  --run_name test_run_random_02 \
  --n_generate 5000 \
  --num_gen_x 100000 \
  --ot_bat_size_n 10000 \
  --ot_thresh 0.3 \
  --decode_batch_size 256 \
  --min_length 2 \
  --min_pairwise_dist 2.0 \
  --neighbor_exclude 2 \
  --ot_root /public/home/zhangyangroup/chengshiz/keyuan.zhou/AE-OT \
  --select_random \
  --seed 42
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ------------------------ Repo imports ------------------------ #

def _add_to_syspath(p: str):
    p = str(Path(p).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _import_ot(ot_root: Optional[str]):
    if ot_root:
        _add_to_syspath(ot_root)
    try:
        from pyOMT_raw import pyOMT_raw  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import OT module 'pyOMT_raw'. "
            "Please set --ot_root to the directory containing pyOMT_raw.py"
        ) from e
    return pyOMT_raw


def _import_build_experiment(repo_root: str):
    _add_to_syspath(repo_root)
    try:
        from experiment import build_experiment_from_yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import build_experiment_from_yaml from experiment.py. "
            "Make sure you run this script inside your PyTorch-VAE repo, "
            "or pass correct --repo_root."
        ) from e
    return build_experiment_from_yaml


# ------------------------ AE loading / decoding ------------------------ #

def _safe_load_ae(build_experiment_from_yaml, ae_config: str, ae_ckpt: str, device: torch.device):
    exp, cfg = build_experiment_from_yaml(ae_config)

    ckpt = torch.load(ae_ckpt, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    new_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            new_state[k[6:]] = v
        else:
            new_state[k] = v

    exp.model.load_state_dict(new_state, strict=False)
    exp.model.eval().to(device)

    mp = cfg.get("model_params", {}) if isinstance(cfg, dict) else {}
    latent_tokens = int(mp.get("latent_tokens", getattr(exp.model, "latent_n_tokens", 0)) or 0)
    code_dim = int(mp.get("code_dim", getattr(exp.model, "code_dim", 0)) or 0)
    if latent_tokens <= 0 or code_dim <= 0:
        raise RuntimeError("Failed to obtain latent_tokens/code_dim from YAML or model.")
    return exp.model, latent_tokens, code_dim


def _clamp_lengths(lengths: torch.Tensor, min_len: int, max_len: int) -> torch.Tensor:
    lengths = lengths.to(torch.int64)
    lengths = torch.clamp(lengths, min=min_len)
    if max_len and max_len > 0:
        lengths = torch.clamp(lengths, max=max_len)
    return lengths


def _build_mask_from_lengths(lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    lengths = lengths.to(torch.int64)
    Lmax = int(lengths.max().item())
    ar = torch.arange(Lmax, device=device).view(1, -1)
    return ar < lengths.view(-1, 1)


# ------------------------ OT sampling (from demo_curves.py logic) ------------------------ #

@torch.no_grad()
def ot_generate_latents(
    pyOMT_raw_cls,
    features_pt: str,
    ot_h_path: str,
    num_gen_x: int,
    bat_size_n: int,
    thresh: float,
    latent_key: str = "latents",
    lengths_key: str = "lengths",
    ot_device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Load training latents P from features_pt, load OT parameter h.pt, then generate new latents
    via top-2 neighbor selection + angular threshold + random convex interpolation.

    Returns:
      dict: { "latents": [M, D], "lengths": [M] }   (M after angle-filter + unique)
    """

    data = torch.load(features_pt, map_location="cpu")
    if not isinstance(data, dict):
        raise RuntimeError("features_pt must be a dict with keys including 'latents' and 'lengths'.")

    if latent_key not in data:
        raise KeyError(f"features_pt missing key '{latent_key}'")
    if lengths_key not in data:
        raise KeyError(f"features_pt missing key '{lengths_key}'")

    h_P = data[latent_key]
    lengths = data[lengths_key]
    if not isinstance(h_P, torch.Tensor) or not isinstance(lengths, torch.Tensor):
        raise RuntimeError("features_pt['latents'] and ['lengths'] must be torch.Tensor")

    h_P = h_P.float().contiguous()
    lengths = lengths.view(-1).contiguous()

    num_P, dim_y = h_P.shape
    if lengths.numel() != num_P:
        raise RuntimeError(f"lengths numel {lengths.numel()} != num_P {num_P}")

    # Ensure num_gen_x is multiple of bat_size_n (demo logic processes full blocks)
    if num_gen_x < bat_size_n:
        raise ValueError(f"num_gen_x must be >= bat_size_n. Got {num_gen_x} < {bat_size_n}")
    num_bat_x = num_gen_x // bat_size_n
    num_gen_x_eff = num_bat_x * bat_size_n
    if num_gen_x_eff != num_gen_x:
        print(f"[warn] num_gen_x={num_gen_x} not multiple of bat_size_n={bat_size_n}, using {num_gen_x_eff} instead.")
        num_gen_x = num_gen_x_eff

    device = torch.device(ot_device if (ot_device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    h_P_dev = h_P.to(device)
    lengths_dev = lengths.to(device)

    # OT object (maxIter=0 means no training)
    maxIter = 0
    lr = 1e-2  # unused when maxIter=0
    bat_size_P = num_P
    p_s = pyOMT_raw_cls(h_P_dev, num_P, dim_y, maxIter, lr, bat_size_P, bat_size_n)

    # Load OT parameter
    d_h = torch.load(ot_h_path, map_location="cpu")
    # best-effort move to OT device if tensor-like
    if isinstance(d_h, torch.Tensor):
        d_h = d_h.to(device)
    p_s.set_h(d_h)

    # Collect top-2 indices for each x_j
    I_all = torch.empty([2, num_gen_x], dtype=torch.long, device=device)
    for ii in range(num_bat_x):
        p_s.pre_cal(ii)
        p_s.cal_measure()
        _, I = torch.sort(p_s.d_U, dim=0, descending=True)
        s0 = ii * bat_size_n
        s1 = (ii + 1) * bat_size_n
        I_all[0, s0:s1].copy_(I[0, :])
        I_all[1, s0:s1].copy_(I[1, :])

    # Angular filtering using cosine threshold (avoid acos)
    # theta < thresh  <=> cos(theta) > cos(thresh)
    P_norm = p_s.h_P.double()
    nm = torch.cat([P_norm, -torch.ones(p_s.num_P, 1, dtype=torch.float64, device=device)], dim=1)
    nm = nm / torch.norm(nm, dim=1, keepdim=True).clamp_min(1e-12)

    a = I_all[0, :]
    b = I_all[1, :]
    cs = torch.sum(nm[a, :] * nm[b, :], dim=1)
    cs = torch.clamp(cs, -1.0, 1.0)

    cos_thresh = float(np.cos(float(thresh)))
    keep = cs > cos_thresh
    I_gen = I_all[:, keep]

    # Canonicalize pair order to reduce (a,b) vs (b,a) duplicates
    I_gen, _ = torch.sort(I_gen, dim=0)

    # Unique columns
    I_gen_np = I_gen.detach().cpu().numpy()
    _, uni_gen_id = np.unique(I_gen_np, return_index=True, axis=1)
    uni_gen_id = np.asarray(uni_gen_id, dtype=np.int64)
    I_gen = I_gen[:, torch.from_numpy(uni_gen_id).to(device)]
    numGen = int(I_gen.shape[1])
    print(f"[info] OT produced {numGen} unique candidates after angle filter (thresh={thresh}).")

    if numGen == 0:
        return {
            "latents": torch.empty((0, dim_y), dtype=torch.float32),
            "lengths": torch.empty((0,), dtype=torch.long),
        }

    # IMPORTANT: do NOT randomize / subset here. Do it in main with user-specified n_generate.

    # Random convex interpolation weights (generate on CPU, then move)
    rand_w_cpu = torch.rand([numGen, 1], dtype=torch.float64)
    rand_w = rand_w_cpu.to(device)

    P_gen = (P_norm[I_gen[0, :], :] * rand_w) + (P_norm[I_gen[1, :], :] * (1.0 - rand_w))
    P_gen = P_gen.float()

    w1 = rand_w.squeeze(1)
    len_gen = (lengths_dev[I_gen[0, :]].double() * w1) + (lengths_dev[I_gen[1, :]].double() * (1.0 - w1))
    len_gen = torch.round(len_gen).long()

    return {"latents": P_gen.detach().cpu(), "lengths": len_gen.detach().cpu()}


# ------------------------ Filtering (aligned with prior/filter_curves.py) ------------------------ #

def bond_length_stats(coords: np.ndarray, good_min: float = 2.0, good_max: float = 7.2) -> Dict[str, float]:
    L = coords.shape[0]
    if L < 2:
        return {"num": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "frac_out": 0.0}
    diffs = coords[1:] - coords[:-1]
    dists = np.linalg.norm(diffs, axis=-1)
    return {
        "num": int(dists.shape[0]),
        "mean": float(dists.mean()),
        "std": float(dists.std()),
        "min": float(dists.min()),
        "max": float(dists.max()),
        "frac_out": float(np.mean((dists < good_min) | (dists > good_max))),
    }


def bond_angle_stats(coords: np.ndarray, good_min_deg: float = 30.0, good_max_deg: float = 180.0) -> Dict[str, float]:
    L = coords.shape[0]
    if L < 3:
        return {"num": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "frac_out": 0.0}
    p0 = coords[:-2]
    p1 = coords[1:-1]
    p2 = coords[2:]
    v1 = p0 - p1
    v2 = p2 - p1
    v1n = np.linalg.norm(v1, axis=-1)
    v2n = np.linalg.norm(v2, axis=-1)
    denom = v1n * v2n
    mask = denom > 1e-6
    if not np.any(mask):
        return {"num": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "frac_out": 0.0}
    cos_theta = np.zeros_like(denom, dtype=np.float64)
    cos_theta[mask] = np.einsum("ij,ij->i", v1[mask], v2[mask]) / denom[mask]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_theta[mask]))
    return {
        "num": int(angles.shape[0]),
        "mean": float(angles.mean()),
        "std": float(angles.std()),
        "min": float(angles.min()),
        "max": float(angles.max()),
        "frac_out": float(np.mean((angles < good_min_deg) | (angles > good_max_deg))),
    }


def radius_of_gyration(coords: np.ndarray) -> float:
    if coords.ndim != 2 or coords.shape[0] == 0:
        return 0.0
    center = coords.mean(axis=0)
    diff = coords - center
    rg2 = np.mean(np.sum(diff * diff, axis=-1))
    return float(np.sqrt(max(rg2, 0.0)))


def self_collision_stats(coords: np.ndarray, min_pairwise_dist: float, neighbor_exclude: int) -> int:
    if coords.ndim != 2 or coords.shape[1] != 3:
        return 0
    L = coords.shape[0]
    if L <= neighbor_exclude + 1:
        return 0
    idx = np.arange(L, dtype=np.int32)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    mask = np.abs(ii - jj) > int(neighbor_exclude)
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    thresh2 = float(min_pairwise_dist) * float(min_pairwise_dist)
    hit_mask = mask & (dist2 < thresh2)
    return int(hit_mask.sum())


def has_self_collision(coords: np.ndarray, min_pairwise_dist: float, neighbor_exclude: int) -> bool:
    return self_collision_stats(coords, min_pairwise_dist, neighbor_exclude) > 0


def segment_self_clash_count(
    coords: np.ndarray,
    min_seg_dist: float = 1.3,
    neighbor_exclude_segments: int = 1,
    num_samples: int = 5,
) -> int:
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
        pts1 = p0[None, :] + (p1 - p0)[None, :] * t_vals[:, None]
        for j in range(i + 1 + neighbor_exclude_segments, n_seg):
            q0 = coords[j]
            q1 = coords[j + 1]
            pts2 = q0[None, :] + (q1 - q0)[None, :] * t_vals[:, None]
            diff = pts1[:, None, :] - pts2[None, :, :]
            dist2 = np.sum(diff * diff, axis=-1)
            if np.any(dist2 < thresh2):
                count += 1
    return count


def beta_stats(ss_one_hot: np.ndarray, beta_channel: int = 1, threshold: float = 0.5) -> Tuple[int, int]:
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
            max_run = max(max_run, cur)
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
    L = ss_one_hot.shape[0]
    if ss_one_hot.ndim != 2 or ss_one_hot.shape[1] <= beta_channel or L == 0:
        return {
            "beta_total": 0, "beta_in_sheet": 0, "beta_sheet_fraction": 0.0,
            "n_strands_total": 0, "n_sheet_strands": 0, "n_isolated_strands": 0
        }
    beta_mask = ss_one_hot[:, beta_channel] > threshold
    beta_total = int(beta_mask.sum())
    if beta_total == 0:
        return {
            "beta_total": 0, "beta_in_sheet": 0, "beta_sheet_fraction": 0.0,
            "n_strands_total": 0, "n_sheet_strands": 0, "n_isolated_strands": 0
        }

    runs: List[Tuple[int, int]] = []
    i = 0
    while i < L:
        if beta_mask[i]:
            j = i
            while j + 1 < L and beta_mask[j + 1]:
                j += 1
            if (j - i + 1) >= min_strand_len:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1

    n_strands_total = len(runs)
    if n_strands_total == 0:
        return {
            "beta_total": beta_total, "beta_in_sheet": 0, "beta_sheet_fraction": 0.0,
            "n_strands_total": 0, "n_sheet_strands": 0, "n_isolated_strands": 0
        }

    beta_idx = np.nonzero(beta_mask)[0]
    beta_coords = coords[beta_idx]
    B = beta_coords.shape[0]
    if B == 0:
        return {
            "beta_total": beta_total, "beta_in_sheet": 0, "beta_sheet_fraction": 0.0,
            "n_strands_total": n_strands_total, "n_sheet_strands": 0, "n_isolated_strands": n_strands_total
        }

    diff = beta_coords[:, None, :] - beta_coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))

    bi = beta_idx[:, None]
    bj = beta_idx[None, :]
    seq_diff = np.abs(bi - bj)

    sheet_mask = (
        (dist >= float(sheet_min_dist)) &
        (dist <= float(sheet_max_dist)) &
        (seq_diff > int(neighbor_exclude))
    )
    np.fill_diagonal(sheet_mask, False)

    beta_has_partner = sheet_mask.any(axis=1)
    beta_sheet_mask = np.zeros(L, dtype=bool)
    beta_sheet_mask[beta_idx] = beta_has_partner

    beta_in_sheet = int(beta_sheet_mask.sum())
    beta_sheet_fraction = float(beta_in_sheet) / float(beta_total) if beta_total > 0 else 0.0

    n_sheet_strands = 0
    n_isolated_strands = 0
    for (s, e) in runs:
        if beta_sheet_mask[s:e+1].any():
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


def curve_pass_filter(
    curve6: np.ndarray,
    args,
) -> Tuple[bool, Dict[str, object], str]:
    """
    Returns (passed, stats_dict, reject_reason)
    reject_reason is "" if passed.
    """
    if curve6.ndim != 2 or curve6.shape[1] < 3:
        return False, {}, "bad_shape"
    if not np.isfinite(curve6[:, :3]).all():
        return False, {}, "nan_inf"

    L = int(curve6.shape[0])
    if L < int(args.min_length):
        return False, {}, "too_short"
    if int(args.max_length) > 0 and L > int(args.max_length):
        return False, {}, "too_long"

    coords = curve6[:, :3]

    # thresholds aligned with your filter script
    BOND_MIN_ALLOWED = 2.2
    BOND_MAX_ALLOWED = 7.5
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

    bl_stats = bond_length_stats(coords, good_min=BOND_GOOD_MIN, good_max=BOND_GOOD_MAX)
    if bl_stats["num"] > 0:
        if (bl_stats["min"] < BOND_MIN_ALLOWED or bl_stats["max"] > BOND_MAX_ALLOWED or bl_stats["frac_out"] > BOND_FRAC_OUT_MAX):
            return False, {}, "bond_out"

    ba_stats = bond_angle_stats(coords, good_min_deg=ANGLE_GOOD_MIN, good_max_deg=ANGLE_GOOD_MAX)
    if ba_stats["num"] > 0:
        if (ba_stats["min"] < ANGLE_MIN_ALLOWED or ba_stats["max"] > ANGLE_MAX_ALLOWED or ba_stats["frac_out"] > ANGLE_FRAC_OUT_MAX):
            return False, {}, "angle_out"

    if has_self_collision(coords, float(args.min_pairwise_dist), int(args.neighbor_exclude)):
        return False, {}, "point_collision"

    seg_clashes = segment_self_clash_count(
        coords,
        min_seg_dist=SEG_MIN_DIST,
        neighbor_exclude_segments=SEG_NEIGHBOR_EXCLUDE,
        num_samples=5,
    )
    if seg_clashes > 0:
        return False, {}, "segment_collision"

    ss_reject = False
    beta_total = 0
    beta_max_run = 0
    beta_sheet_fraction = 0.0
    beta_in_sheet = 0
    n_strands_total = 0
    n_sheet_strands = 0
    n_isolated_strands = 0

    if curve6.shape[1] >= 6:
        ss_one_hot = curve6[:, 3:6]

        beta_total, beta_max_run = beta_stats(ss_one_hot, beta_channel=int(args.beta_channel))
        if args.min_beta_total > 0 and 0 < beta_total < int(args.min_beta_total):
            ss_reject = True
        if args.min_beta_run > 0 and beta_total > 0 and beta_max_run < int(args.min_beta_run):
            ss_reject = True

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

        if float(args.min_beta_sheet_fraction) > 0.0 and beta_total > 0:
            if beta_sheet_fraction < float(args.min_beta_sheet_fraction):
                ss_reject = True
        if int(args.max_isolated_beta_strands) >= 0:
            if n_isolated_strands > int(args.max_isolated_beta_strands):
                ss_reject = True

    if ss_reject:
        return False, {}, "ss_reject"

    rg = radius_of_gyration(coords)

    stats = {
        "length_recon": L,
        "rg": float(rg),
        "bond_mean": float(bl_stats["mean"]),
        "bond_std": float(bl_stats["std"]),
        "bond_min": float(bl_stats["min"]),
        "bond_max": float(bl_stats["max"]),
        "bond_frac_out": float(bl_stats["frac_out"]),
        "angle_mean": float(ba_stats["mean"]),
        "angle_std": float(ba_stats["std"]),
        "angle_min": float(ba_stats["min"]),
        "angle_max": float(ba_stats["max"]),
        "angle_frac_out": float(ba_stats["frac_out"]),
        "beta_total": int(beta_total),
        "beta_max_run": int(beta_max_run),
        "beta_in_sheet": int(beta_in_sheet),
        "beta_sheet_fraction": float(beta_sheet_fraction),
        "beta_strands_total": int(n_strands_total),
        "beta_strands_sheet": int(n_sheet_strands),
        "beta_strands_isolated": int(n_isolated_strands),
        "n_self_clash_pairs": int(self_collision_stats(coords, float(args.min_pairwise_dist), int(args.neighbor_exclude))),
        "n_seg_clash_pairs": int(seg_clashes),
    }
    return True, stats, ""


# ------------------------ Main pipeline ------------------------ #

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--repo_root", type=str, default=str(Path(__file__).resolve().parents[1]),
                    help="PyTorch-VAE repo root (default: parent of scripts/)")
    ap.add_argument("--ot_root", type=str, default="",
                    help="Directory containing pyOMT_raw.py (if not importable by default)")

    ap.add_argument("--ae_config", type=str, required=True)
    ap.add_argument("--ae_ckpt", type=str, required=True)

    ap.add_argument("--features_pt", type=str, required=True, help="Training latent bank .pt (contains latents/lengths)")
    ap.add_argument("--ot_h", type=str, required=True, help="Trained OT parameter h.pt")

    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--run_name", type=str, default="", help="Optional run folder name. If empty, timestamp is used.")

    # How many candidate curves you want to decode+filter (ONE-SHOT)
    ap.add_argument("--n_generate", type=int, required=True,
                    help="How many candidate curves to decode+filter ONCE. "
                         "If OT produces more, we select n_generate. If OT produces fewer, we use all (no refill).")
    ap.add_argument("--select_random", action="store_true",
                    help="If set, randomly select n_generate candidates when OT produces more than needed. "
                         "If not set, take the first n_generate.")
    ap.add_argument("--seed", type=int, default=0)

    # OT sampling
    ap.add_argument("--num_gen_x", type=int, default=100000,
                    help="Candidate x count (multiple of ot_bat_size_n). "
                         "You may set this larger than n_generate to ensure enough OT candidates.")
    ap.add_argument("--ot_bat_size_n", type=int, default=10000)
    ap.add_argument("--ot_thresh", type=float, default=0.3)
    ap.add_argument("--ot_device", type=str, default="cpu", help="cpu or cuda")

    # Decoding
    ap.add_argument("--decode_device", type=str, default="cuda")
    ap.add_argument("--decode_batch_size", type=int, default=64)
    ap.add_argument("--latent_key", type=str, default="latents")
    ap.add_argument("--min_len_clamp", type=int, default=1)
    ap.add_argument("--max_len_clamp", type=int, default=0)
    ap.add_argument("--gen_len_fallback", type=int, default=128)

    # Filtering thresholds (aligned with your filter_curves.py)
    ap.add_argument("--min_length", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=0)
    ap.add_argument("--min_pairwise_dist", type=float, default=2.0)
    ap.add_argument("--neighbor_exclude", type=int, default=2)
    ap.add_argument("--min_beta_run", type=int, default=0)
    ap.add_argument("--min_beta_total", type=int, default=0)
    ap.add_argument("--beta_channel", type=int, default=1)
    ap.add_argument("--min_beta_sheet_fraction", type=float, default=0.0)
    ap.add_argument("--max_isolated_beta_strands", type=int, default=-1)
    ap.add_argument("--min_strand_len", type=int, default=3)

    # Output
    ap.add_argument("--name_pattern", type=str, default="gen_{idx:06d}.npy")
    ap.add_argument("--save_raw_decoded", action="store_true",
                    help="Also save all decoded curves (raw) before filtering.")

    return ap.parse_args()


def main():
    args = parse_args()

    repo_root = str(Path(args.repo_root).resolve())
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name.strip()
    if not run_name:
        run_name = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = run_dir / "decoded_npy" if args.save_raw_decoded else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    filtered_dir = run_dir / "filtered_npy"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "filtered_manifest.jsonl"
    summary_path = run_dir / "summary.json"

    # Save args
    with (run_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    t0 = time.time()

    # Imports
    pyOMT_raw_cls = _import_ot(args.ot_root if args.ot_root else None)
    build_experiment_from_yaml = _import_build_experiment(repo_root)

    # Load AE
    decode_device = torch.device(args.decode_device if (args.decode_device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f"[info] Loading AE on {decode_device} ...")
    ae, latent_tokens, code_dim = _safe_load_ae(build_experiment_from_yaml, args.ae_config, args.ae_ckpt, decode_device)
    flat_dim_expected = int(latent_tokens * code_dim)
    print(f"[info] AE latent_tokens={latent_tokens} code_dim={code_dim} flat_dim={flat_dim_expected}")

    # 1) OT generate latents ONCE
    t_ot0 = time.time()
    ot_out = ot_generate_latents(
        pyOMT_raw_cls=pyOMT_raw_cls,
        features_pt=args.features_pt,
        ot_h_path=args.ot_h,
        num_gen_x=int(args.num_gen_x),
        bat_size_n=int(args.ot_bat_size_n),
        thresh=float(args.ot_thresh),
        latent_key=str(args.latent_key),
        lengths_key="lengths",
        ot_device=str(args.ot_device),
    )
    t_ot1 = time.time()

    z_in = ot_out["latents"]        # [M, D]
    lengths = ot_out["lengths"]     # [M]
    if z_in.ndim != 2:
        raise RuntimeError(f"OT output latents must be [N, D], got {tuple(z_in.shape)}")
    if z_in.shape[1] != flat_dim_expected:
        raise RuntimeError(f"Latent dim mismatch: OT D={z_in.shape[1]} vs AE expected {flat_dim_expected}")

    M = int(z_in.shape[0])
    if M == 0:
        print("[warn] No OT candidates generated. Exiting.")
        summary = {
            "run_dir": str(run_dir),
            "ot_candidates": 0,
            "decoded": 0,
            "kept": 0,
            "reject_counts": {},
            "timing_sec": {"total": float(time.time() - t0), "ot": float(t_ot1 - t_ot0)},
            "outputs": {
                "filtered_dir": str(filtered_dir),
                "filtered_manifest": str(manifest_path),
                "summary": str(summary_path),
                "raw_decoded_dir": str(raw_dir) if raw_dir is not None else "",
            },
        }
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        return

    lengths = _clamp_lengths(lengths, int(args.min_len_clamp), int(args.max_len_clamp))
    print(f"[info] OT candidates (after angle+unique): {M}")
    print(f"[info] OT length min/mean/max = {int(lengths.min())}/{float(lengths.float().mean()):.2f}/{int(lengths.max())}")

    # 2) Select exactly n_generate candidates (or fewer if not enough). NO refill.
    n_generate = int(args.n_generate)
    if n_generate <= 0:
        raise ValueError("--n_generate must be > 0")

    if M < n_generate:
        print(f"[warn] OT produced only {M} candidates < n_generate={n_generate}. Will use all {M} (NO refill).")
        N = M
        z_sel = z_in
        len_sel = lengths
    else:
        N = n_generate
        if args.select_random:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(args.seed))
            perm = torch.randperm(M, generator=g)
            sel = perm[:N]
            z_sel = z_in.index_select(0, sel)
            len_sel = lengths.index_select(0, sel)
            print(f"[info] Selected {N}/{M} candidates randomly for decode+filter (seed={args.seed}).")
        else:
            z_sel = z_in[:N]
            len_sel = lengths[:N]
            print(f"[info] Selected first {N}/{M} candidates for decode+filter.")

    # Counters
    kept = 0
    total_decoded = 0
    reject_counts = {
        "too_short": 0,
        "too_long": 0,
        "bond_out": 0,
        "angle_out": 0,
        "point_collision": 0,
        "segment_collision": 0,
        "ss_reject": 0,
        "bad_shape": 0,
        "nan_inf": 0,
        "other": 0,
    }

    mf = open(manifest_path, "w", encoding="utf-8")

    # 3) Decode + Filter (single pass, streaming)
    t_df0 = time.time()
    bs = int(args.decode_batch_size)
    pbar = tqdm(total=N, desc="Decode+Filter (one-shot)", ncols=110)

    for i0 in range(0, N, bs):
        i1 = min(N, i0 + bs)

        z_flat = z_sel[i0:i1].to(decode_device, non_blocking=True).float()
        b_lengths = len_sel[i0:i1].to(decode_device, non_blocking=True)

        # reshape to tokens
        z_tokens = z_flat.view(i1 - i0, latent_tokens, code_dim).contiguous()

        # mask
        mask = _build_mask_from_lengths(b_lengths, decode_device)

        # decode -> [B, Lmax, 6] (xyz + ss_logits)
        with torch.no_grad():
            recons = ae.decode(z_tokens, mask=mask)

        coords = recons[..., :3].float()
        ss_logits = recons[..., 3:].float()
        ss_idx = torch.argmax(ss_logits, dim=-1)
        ss_one_hot = F.one_hot(ss_idx, num_classes=3).float()
        arr6 = torch.cat([coords, ss_one_hot], dim=-1).cpu().numpy().astype(np.float32)

        for bi in range(i1 - i0):
            global_idx = i0 + bi  # 0..N-1 in this run
            L = int(b_lengths[bi].item()) if b_lengths.numel() > 0 else int(args.gen_len_fallback)
            curve6 = arr6[bi, :L]

            # optionally save raw decoded
            if raw_dir is not None:
                raw_path = raw_dir / args.name_pattern.format(idx=global_idx)
                np.save(str(raw_path), curve6, allow_pickle=False)

            passed, stats, reason = curve_pass_filter(curve6, args)
            total_decoded += 1

            if not passed:
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue

            out_path = filtered_dir / args.name_pattern.format(idx=global_idx)
            np.save(str(out_path), curve6, allow_pickle=False)

            rec = {
                "i": int(global_idx),
                "recon_path": str(out_path),
                "length_recon": int(stats.get("length_recon", L)),
                "ot_thresh": float(args.ot_thresh),
                "num_gen_x": int(args.num_gen_x),
                "ot_bat_size_n": int(args.ot_bat_size_n),
                "ae_ckpt": str(args.ae_ckpt),
                "features_pt": str(args.features_pt),
                "ot_h": str(args.ot_h),
            }
            rec.update(stats)
            mf.write(json.dumps(rec) + "\n")
            kept += 1

        pbar.update(i1 - i0)

    pbar.close()
    t_df1 = time.time()
    mf.close()

    t1 = time.time()

    summary = {
        "run_dir": str(run_dir),
        "ot_candidates_after_angle_unique": int(M),
        "selected_for_decode_filter": int(N),
        "kept": int(kept),
        "total_decoded": int(total_decoded),
        "reject_counts": {k: int(v) for k, v in reject_counts.items()},
        "timing_sec": {
            "ot": float(t_ot1 - t_ot0),
            "decode_filter": float(t_df1 - t_df0),
            "total": float(t1 - t0),
        },
        "outputs": {
            "filtered_dir": str(filtered_dir),
            "filtered_manifest": str(manifest_path),
            "summary": str(summary_path),
            "raw_decoded_dir": str(raw_dir) if raw_dir is not None else "",
        },
    }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n[done] Timing:")
    print(json.dumps(summary["timing_sec"], indent=2))
    print(f"[done] selected={N}, kept={kept}, total_decoded={total_decoded}")
    print(f"[done] outputs: {run_dir}")


if __name__ == "__main__":
    main()
