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
from matplotlib.lines import Line2D # Added for custom legend

"""
Example:

python scripts/probe_pdb_in_tsne2.py \
   --pdb /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFDbeta-chainA \
         /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFD100-chainA \
         /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFD1000-chainA \
         /public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/data/GPR4-RFD3000-chainA
        
Output:
1. combined_probe.png (Folder 1 + 2 + 3 on gray bg)
2. folder1_probe.png (Folder 1 on gray bg)
3. folder2_probe.png (Folder 2 on gray bg)
4. folder3_probe.png (Folder 3 on gray bg)
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

# t-SNE cache path saved by visualize_latent_and_codebook2.py
TSNE_CACHE_PATH = REPO_ROOT / "latent_analysis" / "class1" / "tsne_cache_class1_len_between_1_80.npz"

# Where to save probe figures
OUTPUT_DIR = Path(
    "/public/home/zhangyangroup/chengshiz/run/20251107_ccx-binder-fig/ccx-binder-fig/examples/bb-crv_sp-probe-res/class1"
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
PRP_WORKERS = 16 

# ----------------------------------------------------------------------
# Imports from your repo
# ----------------------------------------------------------------------

from models.vq_vae import VQVAE
from dataset import CurveDataset, pad_collate

# ----------------------------------------------------------------------
# Colors 
# ----------------------------------------------------------------------

# Define a consistent color cycle for groups
COLOR_CYCLE = [
    "#ef4444", # Red
    "#3b82f6", # Blue
    "#22c55e", # Green
    "#f97316", # Orange
    "#a855f7", # Purple
    "#eab308", # Yellow
    "#06b6d4", # Cyan
    "#ec4899", # Pink
]

MARKER_CYCLE = ["*", "X", "o", "s", "D", "^", "v", "P"]

# ----------------------------------------------------------------------
# Helper Functions
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

def knn_interpolate_tsne(base_latents, base_2d, z_query, k=10, eps=1e-6):
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
# Plotting Function (Modified for Gray Background)
# ----------------------------------------------------------------------

def plot_tsne_gray_background_with_queries(
    lat2d,
    query_coords,
    query_labels,
    query_groups,
    group_color_map,   # group_name -> color
    group_marker_map,  # group_name -> marker
    out_png: Path,
    title: str,
    show_labels: bool = False,
):

    x = lat2d[:, 0]
    y = lat2d[:, 1]

    nbins = 200
    H, xedges, yedges = np.histogram2d(x, y, bins=nbins)

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xcenters, ycenters)  

    H_plot = H.T  
    vmax = H_plot.max()
    if vmax <= 0:
        H_plot = None

    fig, ax = plt.subplots(figsize=(10.0, 10.0), dpi=220)

    ax.set_facecolor("white")

    if H_plot is not None:
        levels = np.linspace(0.0, vmax, 10)[1:]
        ax.contourf(
            Xg,
            Yg,
            H_plot,
            levels=levels,
            cmap="Greys",
            alpha=0.35,  
            zorder=1,
        )

    edge_color = "white"
    present_groups = sorted(list(set(query_groups)))

    for coord, label, gname in zip(query_coords, query_labels, query_groups):
        qx, qy = float(coord[0]), float(coord[1])
        c = group_color_map.get(gname, "black")
        m = group_marker_map.get(gname, "o")
        ax.scatter(
            [qx],
            [qy],
            s=60,
            c=c,
            edgecolors=edge_color,
            linewidths=0.8,
            marker=m,
            zorder=10,
        )
        if show_labels:
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

    ax.set_xlabel("t-SNE dim-1")
    ax.set_ylabel("t-SNE dim-2")
    ax.set_title(title)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    legend_handles = []
    for g in present_groups:
        c = group_color_map.get(g, "black")
        m = group_marker_map.get(g, "o")
        legend_handles.append(
            Line2D(
                [0], [0],
                marker=m,
                color="w",
                label=g,
                markerfacecolor=c,
                markeredgecolor=edge_color,
                markersize=8,
            )
        )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title="Probe Groups",
            loc="lower left",
            framealpha=0.9,
        )

    fig.tight_layout()
    fig.savefig(str(out_png))
    plt.close(fig)



# ----------------------------------------------------------------------
# Run prp-data process 
# ----------------------------------------------------------------------

def run_prp_process_multi_pdb(pdb_files, tmp_root: Path, workers: int = PRP_WORKERS):
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
        "mamba", "run", "-p", str(env_prefix),
        "prp-data", "process",
        "--input", str(pdb_input_dir),
        "--output", str(curves_out_dir),
        "--workers", str(int(workers)),
        "--device", "cpu",
        "--metadata", "probe_metadata.json",
    ]
    print("[PRP] Running:", " ".join(cmd))
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    if result.returncode != 0:
        print("[PRP] stdout:\n", result.stdout)
        print("[PRP] stderr:\n", result.stderr)
        raise RuntimeError("prp-data process failed with code {}".format(result.returncode))

    npy_files = [p for p in curves_out_dir.iterdir() if p.suffix == ".npy"]
    if not npy_files:
        raise RuntimeError("No .npy produced by prp-data")

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
        pdb_to_npy[pdb_path] = found

    return pdb_to_npy


@torch.no_grad()
def encode_single_batch_to_latent(model, x, mask, device, use_amp: bool):
    x = x.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    
    if use_amp and device.type == "cuda":
         dtype = torch.float16
    else:
         dtype = torch.float32

    model.eval()
    with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
        h_fuse, _, _ = model.encode(x, mask=mask)
        z_tok = model._tokenize_to_codes(h_fuse, mask)
        z_seq = z_tok.mean(dim=1)

    z = z_seq.cpu().numpy()[0]
    return z.astype(np.float32, copy=False)


def collect_pdb_files_and_groups(pdb_args):
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
    
    if not pdb_files:
        raise RuntimeError("No valid .pdb files found")

    return pdb_files, pdb_groups


def derive_group_name(pdb_args, pdb_files):
    if len(pdb_args) == 1:
        only = Path(pdb_args[0]).resolve()
        return only.name if only.is_dir() else only.stem
    else:
        return "multi_{}_pdbs".format(len(pdb_files))


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
        raise FileNotFoundError(f"CKPT not found: {CKPT_PATH}")
    if not TSNE_CACHE_PATH.is_file():
        raise FileNotFoundError(f"TSNE cache not found: {TSNE_CACHE_PATH}")

    # 1. Collect inputs
    pdb_files, pdb_groups = collect_pdb_files_and_groups(args.pdb)
    run_name = derive_group_name(args.pdb, pdb_files)
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Setup Device & Model
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device] Using:", device)

    # 3. Load Cache (Background Data)
    cache = np.load(str(TSNE_CACHE_PATH), allow_pickle=True)
    latents = cache["latents"]
    tsne_2d = cache["tsne_2d"]
    lengths = cache["lengths"]
    # We no longer need CATH subclasses for coloring, but we load TSNE coords

    print(f"[Cache] Loaded tsne cache: {latents.shape[0]} points")

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
    model.load_state_dict(state, strict=False)
    
    # 4. Process PDBs and Get Latents
    query_coords = []
    query_labels = []
    query_groups = [] # Will store group names for each point

    with tempfile.TemporaryDirectory(prefix="probe_pdbs_") as tmp_root_str:
        tmp_root = Path(tmp_root_str).resolve()
        pdb_to_npy = run_prp_process_multi_pdb(pdb_files, tmp_root, workers=PRP_WORKERS)

        for idx, (pdb_path, group_name_i) in enumerate(zip(pdb_files, pdb_groups)):
            pdb_name = pdb_path.name
            curve_npy_path = pdb_to_npy[pdb_path]
            
            print(f"[Probe] {idx+1}/{len(pdb_files)}: {pdb_name} (Group: {group_name_i})")

            # Create temp loader for single file
            list_path = tmp_root / f"probe_list_{idx}.txt"
            with list_path.open("w") as f:
                f.write(curve_npy_path.name + "\n")

            ds = CurveDataset(
                npy_dir=str(curve_npy_path.parent),
                list_path=str(list_path),
                train=False,
            )
            loader = torch.utils.data.DataLoader(
                ds, batch_size=1, shuffle=False, collate_fn=pad_collate
            )
            batch = next(iter(loader))
            if isinstance(batch, (list, tuple)):
                x, mask = batch
            else:
                x, mask = batch, torch.ones((batch.size(0), batch.size(1)), dtype=torch.bool)

            z_query = encode_single_batch_to_latent(model, x, mask, device, USE_AMP)
            
            # Interpolate to t-SNE
            query_2d, _, _ = knn_interpolate_tsne(latents, tsne_2d, z_query, k=int(KNN_K))
            
            query_coords.append(query_2d)
            query_labels.append(pdb_name)
            query_groups.append(group_name_i)

    # ------------------------------------------------------------------
    # 5. Setup Coloring and Mapping
    # ------------------------------------------------------------------
    
    unique_groups = sorted(list(set(query_groups)))
    print(f"\n[Plotting] Found {len(unique_groups)} unique groups: {unique_groups}")

    # Assign persistent colors/markers to each group so they match across plots
    group_color_map = {}
    group_marker_map = {}
    
    for i, gname in enumerate(unique_groups):
        group_color_map[gname] = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        group_marker_map[gname] = MARKER_CYCLE[i % len(MARKER_CYCLE)]

    # ------------------------------------------------------------------
    # 6. Generate Plot 1: The "Combined" Plot (All groups, one figure)
    # ------------------------------------------------------------------
    
    combined_png = out_dir / f"{run_name}_ALL_combined.png"
    print(f"[Plotting] Saving combined plot to {combined_png}")
    
    plot_tsne_gray_background_with_queries(
        lat2d=tsne_2d,
        query_coords=query_coords,
        query_labels=query_labels,
        query_groups=query_groups,
        group_color_map=group_color_map,
        group_marker_map=group_marker_map,
        out_png=combined_png,
        title=f"Combined: {', '.join(unique_groups)}",
        show_labels=False # Turn off labels for cleaner view if points are many
    )

    # ------------------------------------------------------------------
    # 7. Generate Plot 2..N: Individual Plots (One figure per group)
    # ------------------------------------------------------------------
    
    for gname in unique_groups:
        single_png = out_dir / f"{run_name}_GROUP_{gname}.png"
        print(f"[Plotting] Saving individual plot for group '{gname}' to {single_png}")
        
        # Filter data for just this group
        sub_coords = []
        sub_labels = []
        sub_groups = []
        
        for coord, label, grp in zip(query_coords, query_labels, query_groups):
            if grp == gname:
                sub_coords.append(coord)
                sub_labels.append(label)
                sub_groups.append(grp)
        
        plot_tsne_gray_background_with_queries(
            lat2d=tsne_2d,
            query_coords=sub_coords,
            query_labels=sub_labels,
            query_groups=sub_groups,
            group_color_map=group_color_map, # Re-use same map for consistent colors
            group_marker_map=group_marker_map,
            out_png=single_png,
            title=f"Group: {gname}",
            show_labels=False # Maybe show labels on individual plots
        )

    print("\n[Done] All plots generated.")

if __name__ == "__main__":
    main()