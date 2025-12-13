#!/usr/bin/env python3
# coding: utf-8

import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
Example:

python scripts/probe_pdb_in_tsne.py \
  --pdb /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFDbase-chainA \
        /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFDbeta-chainA \
        /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFDfilter-chainA
"""

# ----------------------------------------------------------------------
# Hard-coded paths and config (edit these to match your environment)
# ----------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Path to prp-data env
PRP_ENV_PREFIX = Path(
    "/public/home/zhangyangroup/chengshiz/run/20250717_prp-data/prp-data/.pixi/envs/default"
)

# VQVAE checkpoint path
CKPT_PATH = REPO_ROOT / "checkpoints" / "vq_s_gradient_ckpt_test11_15" / "epochepoch=549.ckpt"

# t-SNE cache path saved by visualize_latent_and_codebook.py
TSNE_CACHE_PATH = REPO_ROOT / "latent_analysis" / "mainly_alpha_out" / "baseline_tsne_cache.npz"

# Where to save probe figures
OUTPUT_DIR = Path(
    "/public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/examples/bb-crv_sp-probe-res"
)

# Model architecture (must match training)
HIDDEN_DIM = 512
NUM_LAYERS = 4
NUM_HEADS = 8
MAX_SEQ_LEN = 350
CODE_DIM = 128
LATENT_N_TOKENS = 48

# Other configs
USE_AMP = True
KNN_K = 10
PRP_WORKERS = 16  # number of CPU workers for prp-data process

# ----------------------------------------------------------------------
# Imports from your repo
# ----------------------------------------------------------------------

from models.vq_vae import VQVAE
from dataset import CurveDataset, pad_collate

# ----------------------------------------------------------------------
# Colors and SS mixing
# ----------------------------------------------------------------------

# Helix, sheet are saturated; loop is light and close to background
HELIX_COLOR = np.array([239, 68, 68], dtype=np.float32) / 255.0       # red-ish
SHEET_COLOR = np.array([34, 197, 94], dtype=np.float32) / 255.0       # green-ish
LOOP_COLOR = np.array([191, 219, 254], dtype=np.float32) / 255.0      # light blue
GRAY_BG = np.array([241, 245, 249], dtype=np.float32) / 255.0         # light gray


def mix_three_colors_simplex(
    helix_base,
    sheet_base,
    loop_base,
    helix_frac,
    sheet_frac,
    loop_frac,
    weight_exp: float = 1.0,
):
    """Winner-take-all style mixing with purity-controlled saturation."""
    h = np.asarray(helix_frac, dtype=np.float32)
    s = np.asarray(sheet_frac, dtype=np.float32)
    l = np.asarray(loop_frac, dtype=np.float32)

    w = np.stack([h, s, l], axis=1)
    w = np.clip(w, 0.0, 1.0)
    w_sum = np.sum(w, axis=1, keepdims=True)
    w = np.divide(w, w_sum, out=np.zeros_like(w), where=w_sum > 0.0)

    max_w = np.max(w, axis=1)
    purity = (max_w - 1.0 / 3.0) / (1.0 - 1.0 / 3.0)
    purity = np.clip(purity, 0.0, 1.0)

    if weight_exp != 1.0:
        purity = np.power(purity, weight_exp)

    winner_idx = np.argmax(w, axis=1)
    base_colors = np.stack([helix_base, sheet_base, loop_base], axis=0)
    winner_colors = base_colors[winner_idx]

    colors = GRAY_BG[None, :] * (1.0 - purity)[:, None] + winner_colors * purity[:, None]
    colors = np.clip(colors, 0.0, 1.0)
    return colors


def generate_simplex_palette(
    out_png: Path,
    helix_color,
    sheet_color,
    loop_color,
    size: int = 400,
    padding: int = 40,
    weight_exp: float = 1.0,
):
    """Draw a simplex triangle palette using the same SS mixing rule."""
    bg_color = np.array([248, 250, 252], dtype=np.float32) / 255.0
    img = np.tile(bg_color[None, None, :], (size, size, 1))

    v1 = np.array([size / 2.0, padding], dtype=np.float32)            # helix (top)
    v2 = np.array([size - padding, size - padding], dtype=np.float32) # sheet (bottom-right)
    v3 = np.array([padding, size - padding], dtype=np.float32)        # loop (bottom-left)

    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    px = grid_x
    py = grid_y

    detT = (v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * (v1[1] - v3[1])
    w1 = ((v2[1] - v3[1]) * (px - v3[0]) + (v3[0] - v2[0]) * (py - v3[1])) / detT
    w2 = ((v3[1] - v1[1]) * (px - v3[0]) + (v1[0] - v3[0]) * (py - v3[1])) / detT
    w3 = 1.0 - w1 - w2

    mask = (w1 >= -0.005) & (w2 >= -0.005) & (w3 >= -0.005)

    cw1 = np.clip(w1, 0.0, 1.0)
    cw2 = np.clip(w2, 0.0, 1.0)
    cw3 = np.clip(w3, 0.0, 1.0)
    sum_w = cw1 + cw2 + cw3
    sum_w[sum_w == 0.0] = 1.0
    nw1 = cw1 / sum_w
    nw2 = cw2 / sum_w
    nw3 = cw3 / sum_w

    h_flat = nw1[mask].ravel()
    s_flat = nw2[mask].ravel()
    l_flat = nw3[mask].ravel()
    colors = mix_three_colors_simplex(
        helix_color,
        sheet_color,
        loop_color,
        h_flat,
        s_flat,
        l_flat,
        weight_exp=weight_exp,
    )
    img[mask] = colors

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=220)
    ax.imshow(img, origin="upper")
    ax.set_xlim(0, size - 1)
    ax.set_ylim(size - 1, 0)
    ax.axis("off")
    ax.set_title("SS simplex palette", fontsize=12)

    tri_x = [v1[0], v2[0], v3[0], v1[0]]
    tri_y = [v1[1], v2[1], v3[1], v1[1]]
    ax.plot(tri_x, tri_y, color="#334155", linewidth=1.5)

    ax.text(
        v1[0],
        v1[1] - 10,
        "Helix",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#0f172a",
    )
    ax.text(
        v2[0] + 10,
        v2[1] + 5,
        "Sheet",
        ha="left",
        va="top",
        fontsize=8,
        color="#0f172a",
    )
    ax.text(
        v3[0] - 10,
        v3[1] + 5,
        "Loop",
        ha="right",
        va="top",
        fontsize=8,
        color="#0f172a",
    )

    fig.tight_layout()
    fig.savefig(str(out_png), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# def plot_tsne_simplex_ss_with_queries(
#     lat2d,
#     helix_frac,
#     sheet_frac,
#     loop_frac,
#     query_coords,
#     query_labels,
#     query_groups,
#     out_png: Path,
#     title: str,
#     weight_exp: float = 1.0,
#     show_labels: bool = True,
# ):
    # """
    # Draw TSNE background colored by SS simplex, and overlay probe points.

    # query_groups is a list of group names (one per probe). Probes from different
    # groups use different marker shapes and appear in a separate legend.
    # """
    # if query_groups is None:
    #     query_groups = ["probes"] * len(query_coords)

    # assert len(query_coords) == len(query_labels) == len(query_groups)

    # # Background colors
    # colors = mix_three_colors_simplex(
    #     HELIX_COLOR,
    #     SHEET_COLOR,
    #     LOOP_COLOR,
    #     helix_frac,
    #     sheet_frac,
    #     loop_frac,
    #     weight_exp=weight_exp,
    # )

    # fig, ax = plt.subplots(figsize=(10.0, 10.0), dpi=220)

    # ax.scatter(
    #     lat2d[:, 0],
    #     lat2d[:, 1],
    #     s=4,
    #     c=colors,
    #     alpha=0.65,
    #     edgecolors="none",
    # )

    # # Assign a marker shape per group (based on first appearance order)
    # unique_groups = []
    # for g in query_groups:
    #     if g not in unique_groups:
    #         unique_groups.append(g)

    # # Markers for different groups
    # marker_cycle = ["*", "X", "o", "s", "D", "^", "v", "P"]

    # # Colors for different groups
    # color_cycle = [
    #     "yellow",      # group 0
    #     "#f97316",     # orange
    #     "#0ea5e9",     # cyan
    #     "#a855f7",     # purple
    #     "#22c55e",     # green
    # ]

    # marker_by_group = {}
    # color_by_group = {}
    # for idx, g in enumerate(unique_groups):
    #     marker_by_group[g] = marker_cycle[idx % len(marker_cycle)]
    #     color_by_group[g] = color_cycle[idx % len(color_cycle)]

    # edge_color = "none"

    # # Scatter probes (no text yet)
    # for coord, label, gname in zip(query_coords, query_labels, query_groups):
    #     qx, qy = float(coord[0]), float(coord[1])
    #     ax.scatter(
    #         [qx],
    #         [qy],
    #         s=40,
    #         c=color_by_group.get(gname, "yellow"),
    #         edgecolors=edge_color,
    #         linewidths=0.8,
    #         marker=marker_by_group.get(gname, "*"),
    #         zorder=10,
    #     )

    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()

    # ax.set_xlabel("t-SNE dim-1")
    # ax.set_ylabel("t-SNE dim-2")
    # ax.set_title(title)

    # from matplotlib.patches import Patch
    # from matplotlib.lines import Line2D

    # # Legend for SS mixture (background)
    # ss_handles = [
    #     Patch(facecolor=HELIX_COLOR, edgecolor="none", label="Helix"),
    #     Patch(facecolor=SHEET_COLOR, edgecolor="none", label="Sheet"),
    #     Patch(facecolor=LOOP_COLOR, edgecolor="none", label="Loop"),
    # ]
    # legend_ss = ax.legend(handles=ss_handles, title="SS mixture", loc="upper right")
    # ax.add_artist(legend_ss)

    # # Legend for probe groups (markers)
    # group_handles = []
    # for g in unique_groups:
    #     group_handles.append(
    #         Line2D(
    #             [0],
    #             [0],
    #             marker=marker_by_group[g],
    #             linestyle="None",
    #             markerfacecolor=color_by_group.get(g, "yellow"),
    #             markeredgecolor=edge_color,
    #             markersize=6,
    #             label=g,
    #         )
    #     )
    # ax.legend(handles=group_handles, title="Probe groups", loc="lower left")

    # fig.tight_layout()
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)

    # # Optional text labels near probes
    # if show_labels:
    #     for coord, label, gname in zip(query_coords, query_labels, query_groups):
    #         qx, qy = float(coord[0]), float(coord[1])
    #         ax.text(
    #             qx + 0.5,
    #             qy + 0.5,
    #             label,
    #             fontsize=7,
    #             color="black",
    #             weight="bold",
    #             zorder=11,
    #             clip_on=False,
    #         )
    #     ax.set_xlim(xlim)
    #     ax.set_ylim(ylim)

    # fig.savefig(str(out_png))
    # plt.close(fig)
def plot_tsne_simplex_ss_with_queries(
    lat2d,
    helix_frac,
    sheet_frac,
    loop_frac,
    query_coords,
    query_labels,
    query_groups,
    out_png: Path,
    title: str,
    weight_exp: float = 1.0,
    show_labels: bool = True,
    lengths=None,
):
    """
    Draw TSNE background colored by curve length (grayscale),
    and overlay probe points.

    SS fractions are ignored here; only lengths are used for background.
    """
    if query_groups is None:
        query_groups = ["probes"] * len(query_coords)

    assert len(query_coords) == len(query_labels) == len(query_groups)

    # ------------------------------------------------------------------
    # Background: grayscale based on length
    # ------------------------------------------------------------------
    N = lat2d.shape[0]

    if lengths is None:
        # Fallback: uniform light gray if lengths are not provided
        bg_color = np.array([209, 213, 219], dtype=np.float32) / 255.0
        colors = np.tile(bg_color[None, :], (N, 1))
    else:
        lens = np.asarray(lengths, dtype=np.float32)
        assert lens.shape[0] == N

        len_min = float(lens.min())
        len_max = float(lens.max())

        if len_max <= len_min:
            norm = np.zeros_like(lens)
        else:
            norm = (lens - len_min) / (len_max - len_min + 1e-6)

        # Map normalized length to brightness in [0.2, 0.9]
        # shorter -> darker, longer -> lighter (or vice versa if you prefer)
        brightness = 0.2 + 0.7 * norm
        brightness = np.clip(brightness, 0.0, 1.0)

        colors = np.stack([brightness, brightness, brightness], axis=1)

    fig, ax = plt.subplots(figsize=(10.0, 10.0), dpi=220)

    # Background scatter
    ax.scatter(
        lat2d[:, 0],
        lat2d[:, 1],
        s=4,
        c=colors,
        alpha=0.85,
        edgecolors="none",
    )

    # ------------------------------------------------------------------
    # Probe points (keep the original logic)
    # ------------------------------------------------------------------
    # Assign a marker shape per group (based on first appearance order)
    unique_groups = []
    for g in query_groups:
        if g not in unique_groups:
            unique_groups.append(g)

    marker_cycle = ["*", "X", "o", "s", "D", "^", "v", "P"]
    color_cycle = [
        "yellow",
        "#f97316",
        "#0ea5e9",
        "#a855f7",
        "#22c55e",
    ]

    marker_by_group = {}
    color_by_group = {}
    for idx, g in enumerate(unique_groups):
        marker_by_group[g] = marker_cycle[idx % len(marker_cycle)]
        color_by_group[g] = color_cycle[idx % len(color_cycle)]

    edge_color = "none"

    for coord, label, gname in zip(query_coords, query_labels, query_groups):
        qx, qy = float(coord[0]), float(coord[1])
        ax.scatter(
            [qx],
            [qy],
            s=40,
            c=color_by_group.get(gname, "yellow"),
            edgecolors=edge_color,
            linewidths=0.8,
            marker=marker_by_group.get(gname, "*"),
            zorder=10,
        )

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.set_xlabel("t-SNE dim-1")
    ax.set_ylabel("t-SNE dim-2")
    ax.set_title(title)

    from matplotlib.lines import Line2D

    # Legend for probe groups only (background is continuous grayscale)
    group_handles = []
    for g in unique_groups:
        group_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker_by_group[g],
                linestyle="None",
                markerfacecolor=color_by_group.get(g, "yellow"),
                markeredgecolor=edge_color,
                markersize=6,
                label=g,
            )
        )
    ax.legend(handles=group_handles, title="Probe groups", loc="lower left")

    fig.tight_layout()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Optional text labels near probes
    if show_labels:
        for coord, label, gname in zip(query_coords, query_labels, query_groups):
            qx, qy = float(coord[0]), float(coord[1])
            ax.text(
                qx + 0.5,
                qy + 0.5,
                label,
                fontsize=7,
                color="black",
                weight="bold",
                zorder=11,
                clip_on=False,
            )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.savefig(str(out_png))
    plt.close(fig)


# ----------------------------------------------------------------------
# Utility: strip prefixes and drop quantizer keys
# ----------------------------------------------------------------------

def strip_prefixes(state_dict, prefixes=("model.", "module.", "net.")):
    out = {}
    for k, v in state_dict.items():
        name = k
        for p in prefixes:
            if name.startswith(p):
                name = name[len(p):]
                break
        out[name] = v
    return out


def drop_quantizer_keys(state_dict):
    keys = [k for k in state_dict.keys() if k.startswith("quantizer.")]
    for k in keys:
        state_dict.pop(k, None)
    return state_dict


# ----------------------------------------------------------------------
# KNN interpolation from latent to t-SNE
# ----------------------------------------------------------------------

def knn_interpolate_tsne(base_latents, base_2d, z_query, k=10, eps=1e-6):
    """Project a latent vector to t-SNE space via inverse-distance KNN."""
    assert base_latents.ndim == 2 and base_2d.ndim == 2
    assert base_latents.shape[0] == base_2d.shape[0]
    assert base_latents.shape[1] == z_query.shape[0]

    diffs = base_latents - z_query[None, :]
    dists = np.linalg.norm(diffs, axis=1)

    k = max(1, min(k, base_latents.shape[0]))
    idx = np.argpartition(dists, k - 1)[:k]
    knn_dists = dists[idx]

    weights = 1.0 / (knn_dists + eps)
    weights = weights / weights.sum()

    coords = base_2d[idx]
    query_2d = (weights[:, None] * coords).sum(axis=0)
    return query_2d, idx, knn_dists


# ----------------------------------------------------------------------
# Run prp-data process in external env via mamba run -p (all PDBs at once)
# ----------------------------------------------------------------------

def run_prp_process_multi_pdb(pdb_files, tmp_root: Path, workers: int = PRP_WORKERS):
    """
    Run prp-data process once on a directory containing all PDB files.

    Args:
        pdb_files: list of Path objects pointing to PDB files
        tmp_root: temporary root directory as Path
        workers: number of CPU workers for prp-data

    Returns:
        dict mapping Path(pdb_file) -> Path(curve_npy_file)
    """
    env_prefix = PRP_ENV_PREFIX.resolve()

    pdb_input_dir = tmp_root / "pdb_input"
    curves_out_dir = tmp_root / "curves_out"
    pdb_input_dir.mkdir(parents=True, exist_ok=True)
    curves_out_dir.mkdir(parents=True, exist_ok=True)

    mapping = {}
    used_names = set()

    for idx, pdb_path in enumerate(pdb_files):
        pdb_path = pdb_path.resolve()
        base = pdb_path.name
        dest_name = base
        if dest_name in used_names:
            dest_name = "{:04d}__{}".format(idx, base)
        used_names.add(dest_name)

        dst_pdb = pdb_input_dir / dest_name
        if pdb_path != dst_pdb:
            shutil.copy2(str(pdb_path), str(dst_pdb))

        dest_stem = dst_pdb.stem
        mapping[pdb_path] = {"stem": dest_stem}

    cmd = [
        "mamba",
        "run",
        "-p",
        str(env_prefix),
        "prp-data",
        "process",
        "--input",
        str(pdb_input_dir),
        "--output",
        str(curves_out_dir),
        "--workers",
        str(int(workers)),
        "--device",
        "cpu",
        "--metadata",
        "probe_metadata.json",
    ]
    print("[PRP] Running:", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print("[PRP] stdout:\n", result.stdout)
        print("[PRP] stderr:\n", result.stderr)
        raise RuntimeError("prp-data process failed with code {}".format(result.returncode))

    print("[PRP] process finished. stdout:\n", result.stdout)

    npy_files = [p for p in curves_out_dir.iterdir() if p.suffix == ".npy"]
    if not npy_files:
        raise RuntimeError("No .npy produced by prp-data under {}".format(curves_out_dir))

    npy_files_sorted = sorted(npy_files, key=lambda x: x.name)

    pdb_to_npy = {}
    for pdb_path, meta in mapping.items():
        stem = meta["stem"]
        candidate = curves_out_dir / (stem + ".npy")

        if candidate.is_file():
            pdb_to_npy[pdb_path] = candidate
            continue

        found = None
        for fn in npy_files_sorted:
            if fn.name.startswith(stem):
                found = fn
                break

        if found is None and npy_files_sorted:
            found = npy_files_sorted[0]

        if found is None:
            raise RuntimeError(
                "Could not find any .npy for PDB {} (stem={})".format(str(pdb_path), stem)
            )

        pdb_to_npy[pdb_path] = found

    print("[PRP] Resolved {} PDBs to curve npy files".format(len(pdb_to_npy)))
    return pdb_to_npy


# ----------------------------------------------------------------------
# Encode a single batch (x, mask) to sequence-level latent
# ----------------------------------------------------------------------

@torch.no_grad()
def encode_single_batch_to_latent(model, x, mask, device, use_amp: bool):
    """Encode one curve batch into sequence-level latent vector."""
    x = x.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)

    try:
        autocast_ctx = torch.amp.autocast(
            device_type="cuda",
            enabled=(use_amp and device.type == "cuda"),
        )
    except Exception:
        autocast_ctx = torch.cuda.amp.autocast(
            enabled=(use_amp and device.type == "cuda")
        )

    model.eval()
    with autocast_ctx:
        h_fuse, _, _ = model.encode(x, mask=mask)
        z_tok = model._tokenize_to_codes(h_fuse, mask)
        z_seq = z_tok.mean(dim=1)

    z = z_seq.cpu().numpy()[0]
    return z.astype(np.float32, copy=False)


# ----------------------------------------------------------------------
# Collect pdb files and groups from args
# ----------------------------------------------------------------------

def collect_pdb_files_and_groups(pdb_args):
    """
    Collect PDB files and assign a group name for each top-level argument.

    For each argument:
      - if it is a directory: group name = directory name, all .pdb inside
      - if it is a file: group name = file stem (only if suffix is .pdb)
    """
    pdb_files = []
    pdb_groups = []

    for arg in pdb_args:
        root = Path(arg).resolve()
        if root.is_dir():
            group_name = root.name
            for fn in sorted(root.iterdir()):
                if fn.is_file() and fn.suffix.lower() == ".pdb":
                    pdb_files.append(fn)
                    pdb_groups.append(group_name)
        elif root.is_file():
            if root.suffix.lower() == ".pdb":
                group_name = root.stem
                pdb_files.append(root)
                pdb_groups.append(group_name)
            else:
                print("[Warn] Skip non-pdb file:", str(root))
        else:
            print("[Warn] Path not found, skip:", str(root))

    if not pdb_files:
        raise RuntimeError("No valid .pdb files found from: {}".format(pdb_args))

    # Keep order stable as collected
    return pdb_files, pdb_groups


def derive_group_name(pdb_args, pdb_files):
    """Name for output PNGs."""
    if len(pdb_args) == 1:
        only = Path(pdb_args[0]).resolve()
        if only.is_dir():
            return only.name
        else:
            return only.stem
    else:
        return "multi_{}_pdbs".format(len(pdb_files))


# ----------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Probe one or more PDBs in latent t-SNE space")
    p.add_argument(
        "--pdb",
        type=str,
        nargs="+",
        required=True,
        help="One or more PDB paths; each can be a file or a directory containing .pdb files",
    )
    return p.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()

    if not CKPT_PATH.is_file():
        raise FileNotFoundError("CKPT not found: {}".format(str(CKPT_PATH)))

    if not TSNE_CACHE_PATH.is_file():
        raise FileNotFoundError("TSNE cache not found: {}".format(str(TSNE_CACHE_PATH)))

    pdb_files, pdb_groups = collect_pdb_files_and_groups(args.pdb)
    group_name = derive_group_name(args.pdb, pdb_files)

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png_with = (out_dir / (group_name + "_probe.png")).resolve()
    out_png_nolabel = (out_dir / (group_name + "_probe_nolabel.png")).resolve()
    out_png_simplex = (out_dir / (group_name + "_simplex_palette.png")).resolve()

    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device] Using:", device)
    print("[Probe] Total PDB files:", len(pdb_files))

    cache = np.load(str(TSNE_CACHE_PATH), allow_pickle=True)
    latents = cache["latents"]
    tsne_2d = cache["tsne_2d"]
    helix_frac = cache["helix_frac"]
    sheet_frac = cache["sheet_frac"]
    loop_frac = cache["loop_frac"]
    lengths = cache["lengths"]
    labels = cache.get("labels", None)
    rel_paths = cache.get("rel_paths", None)     
    cath_full = cache.get("cath_full", None)     

    print(
        "[Cache] Loaded tsne cache: {} points, dim={}".format(
            latents.shape[0], latents.shape[1]
        )
    )

    model = VQVAE(
        input_dim=6,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_seq_len=MAX_SEQ_LEN,
        use_vq=False,
        codebook_size=1,
        code_dim=CODE_DIM,
        label_smoothing=0.0,
        ss_tv_lambda=0.0,
        usage_entropy_lambda=0.0,
        xyz_align_alpha=0.7,
        dist_lambda=0.0,
        rigid_aug_prob=0.0,
        pairwise_sample_k=32,
        noise_warmup_steps=0,
        max_noise_std=0.0,
        reinit_dead_codes=False,
        reinit_prob=0.0,
        dead_usage_threshold=0,
        codebook_init_path="",
        latent_tokens=int(LATENT_N_TOKENS),
        tokenizer_heads=NUM_HEADS,
        tokenizer_layers=2,
        tokenizer_dropout=0.1,
        print_init=False,
    ).to(device)

    ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = strip_prefixes(state)
    state = drop_quantizer_keys(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[Load] missing={} unexpected={}".format(len(missing), len(unexpected)))

    query_coords = []
    query_labels = []
    query_groups = []

    from torch.utils.data import DataLoader

    with tempfile.TemporaryDirectory(prefix="probe_pdbs_") as tmp_root_str:
        tmp_root = Path(tmp_root_str).resolve()

        pdb_to_npy = run_prp_process_multi_pdb(pdb_files, tmp_root, workers=PRP_WORKERS)

        for idx, (pdb_path, group_name_i) in enumerate(zip(pdb_files, pdb_groups)):
            pdb_name = pdb_path.name
            curve_npy_path = pdb_to_npy[pdb_path]
            npy_dir = curve_npy_path.parent
            npy_base = curve_npy_path.name

            print(
                "\n[Probe] {} / {}: {} (group: {}) (curve npy: {})".format(
                    idx + 1,
                    len(pdb_files),
                    pdb_name,
                    group_name_i,
                    str(curve_npy_path),
                )
            )

            list_path = tmp_root / ("probe_list_{}.txt".format(idx))
            with list_path.open("w") as f:
                f.write(npy_base + "\n")

            ds = CurveDataset(
                npy_dir=str(npy_dir),
                list_path=str(list_path),
                train=False,
            )
            loader = DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=pad_collate,
                drop_last=False,
            )

            batch = next(iter(loader))
            if isinstance(batch, (list, tuple)):
                x, mask = batch
            else:
                x, mask = batch, None
                mask = torch.ones((x.size(0), x.size(1)), dtype=torch.bool)

            print(
                "[Probe] Loaded curve: x shape={}, mask shape={}".format(
                    tuple(x.shape), tuple(mask.shape)
                )
            )

            z_query = encode_single_batch_to_latent(
                model=model,
                x=x,
                mask=mask,
                device=device,
                use_amp=bool(USE_AMP),
            )
            print(
                "[Probe] Latent dim={}, first 5 values {}".format(
                    z_query.shape[0], z_query[:5]
                )
            )

            query_2d, nn_idx, nn_dists = knn_interpolate_tsne(
                base_latents=latents,
                base_2d=tsne_2d,
                z_query=z_query,
                k=int(KNN_K),
            )
            print(
                "[Probe] t-SNE coord: ({:.3f}, {:.3f})".format(
                    float(query_2d[0]), float(query_2d[1])
                )
            )

            print(
                "[Probe] Nearest neighbors in latent space (top {}):".format(
                    len(nn_idx)
                )
            )
            for rank, (i, d) in enumerate(zip(nn_idx, nn_dists), start=1):
                length_i = float(lengths[i])
                hf = float(helix_frac[i])
                sf = float(sheet_frac[i])
                lf = float(loop_frac[i])
                label_i = (
                    int(labels[i])
                    if (labels is not None and i < labels.shape[0])
                    else -1
                )
                print(
                    "  #{:02d}: idx={} dist={:.4f} length={:.0f} helix={:.2f} sheet={:.2f} loop={:.2f} CATH={}".format(
                        rank,
                        int(i),
                        float(d),
                        length_i,
                        hf,
                        sf,
                        lf,
                        label_i,
                    )
                )

            query_coords.append(query_2d)
            query_labels.append(pdb_name)
            query_groups.append(group_name_i)

    if len(pdb_files) == 1:
        title = "TSNE SS simplex with probe: {}".format(query_labels[0])
    else:
        title = "TSNE SS simplex with {} probes".format(len(pdb_files))

    plot_tsne_simplex_ss_with_queries(
        lat2d=tsne_2d,
        helix_frac=helix_frac,
        sheet_frac=sheet_frac,
        loop_frac=loop_frac,
        query_coords=query_coords,
        query_labels=query_labels,
        query_groups=query_groups,
        out_png=out_png_with,
        title=title,
        weight_exp=1.0,
        show_labels=True,
        lengths=lengths,
    )

    plot_tsne_simplex_ss_with_queries(
        lat2d=tsne_2d,
        helix_frac=helix_frac,
        sheet_frac=sheet_frac,
        loop_frac=loop_frac,
        query_coords=query_coords,
        query_labels=query_labels,
        query_groups=query_groups,
        out_png=out_png_nolabel,
        title=title,
        weight_exp=1.0,
        show_labels=False,
        lengths=lengths,
    )

    generate_simplex_palette(
        out_png=out_png_simplex,
        helix_color=HELIX_COLOR,
        sheet_color=SHEET_COLOR,
        loop_color=LOOP_COLOR,
        size=400,
        padding=40,
        weight_exp=1.0,
    )

    print("\n[Done] Saved probe t-SNE figures:")
    print("  with labels   :", str(out_png_with))
    print("  without labels:", str(out_png_nolabel))
    print("  simplex       :", str(out_png_simplex))


if __name__ == "__main__":
    main()
