#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
'''
python scripts/latent_topology_analysis.py \
  --config configs/stage2_vq.yaml \
  --ckpt /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/vq_s_gradient_ckpt_test11_15/epochepoch=519.ckpt \
  --split train \
  --max_samples 100000 \
  --batch_size 256 \
  --num_workers 8 \
  --kmeans_k 50 \
  --tsne_subset 20000 \
  --out_dir ./latent_analysis \
  --out_prefix stage2_epoch519_train
'''
# make project root importable so "models" and "dataset" can be found
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.vq_vae import VQVAE
from dataset import CurveDataset, pad_collate


def load_config(path: str) -> dict:
    """Load YAML config and expand env vars."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    def _expand_env(obj):
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        if isinstance(obj, dict):
            return {k: _expand_env(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand_env(v) for v in obj]
        return obj

    return _expand_env(cfg)


def build_model_and_dataset(args):
    """Build VQVAE model and dataset from config + split."""
    cfg = load_config(args.config)
    model_params = cfg["model_params"]
    data_params = cfg["data_params"]

    npy_dir = data_params["npy_dir"]
    if args.split == "train":
        list_txt = data_params["train_list"]
        train_flag = True
    else:
        list_txt = data_params["val_list"]
        train_flag = False

    list_path = list_txt
    if not os.path.isabs(list_path):
        list_path = os.path.join(npy_dir, list_txt)

    dataset = CurveDataset(
        npy_dir=npy_dir,
        list_path=list_path,
        train=train_flag,
    )

    model = VQVAE(**model_params)
    return model, dataset, cfg


def load_checkpoint(model: torch.nn.Module, ckpt_path: str):
    """Load Lightning checkpoint into plain model."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    new_state = {}
    for k, v in state.items():
        # Lightning saves as "model.xxx" in training code
        if k.startswith("model."):
            new_key = k[len("model.") :]
            new_state[new_key] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[CKPT] loaded from {ckpt_path}")
    print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")


def extract_latent(
    model: VQVAE,
    dataset: CurveDataset,
    max_samples: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    """
    Encode curves to latent vectors and collect basic statistics.
    Returns:
      Z: [N, D] global latent
      Ls: [N] lengths
      Hs: [N] helix fraction
      Es: [N] sheet fraction
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_collate,
        drop_last=False,
    )

    model.to(device)
    model.eval()

    zs = []
    lengths = []
    helix_fracs = []
    sheet_fracs = []

    total = 0
    with torch.no_grad():
        for batch in loader:
            x, mask = batch  # x: [B,L,6], mask: [B,L]
            B = x.size(0)
            if total >= max_samples:
                break
            remain = max_samples - total
            if B > remain:
                x = x[:remain]
                mask = mask[:remain]
                B = remain

            x = x.to(device)
            mask = mask.to(device)

            # encode to fused token features
            h_fuse, _, _ = model.encode(x, mask=mask)  # [B,L,H]
            # tokenizer L->N
            z_tokens = model._tokenize_to_codes(h_fuse, mask)  # [B,N,D]
            # global latent: mean over tokens
            z_global = z_tokens.mean(dim=1)  # [B,D]

            zs.append(z_global.cpu().numpy())

            # secondary structure stats from ss_one_hot
            ss = x[:, :, 3:].cpu().numpy()  # [B,L,3]
            m = mask.cpu().numpy().astype(np.float32)  # [B,L]
            valid_counts = m.sum(axis=1)  # [B]
            valid_counts[valid_counts < 1.0] = 1.0  # avoid div by zero

            # assume ss_one_hot[...,0] is helix, 1 is sheet
            helix_sum = (ss[:, :, 0] * m).sum(axis=1)
            sheet_sum = (ss[:, :, 1] * m).sum(axis=1)

            helix_frac = helix_sum / valid_counts
            sheet_frac = sheet_sum / valid_counts

            helix_fracs.append(helix_frac)
            sheet_fracs.append(sheet_frac)
            lengths.append(valid_counts)

            total += B

    if zs:
        Z = np.concatenate(zs, axis=0)
    else:
        Z = np.zeros((0, 1), dtype=np.float32)

    if lengths:
        Ls = np.concatenate(lengths, axis=0)
    else:
        Ls = np.zeros((0,), dtype=np.float32)

    if helix_fracs:
        Hs = np.concatenate(helix_fracs, axis=0)
    else:
        Hs = np.zeros((0,), dtype=np.float32)

    if sheet_fracs:
        Es = np.concatenate(sheet_fracs, axis=0)
    else:
        Es = np.zeros((0,), dtype=np.float32)

    print(f"[LATENT] collected {Z.shape[0]} samples, dim={Z.shape[1] if Z.ndim == 2 else 0}")
    return Z, Ls, Hs, Es


def run_tsne(Z: np.ndarray, n_subset: int):
    """Run t-SNE on a random subset of latent vectors."""
    from sklearn.manifold import TSNE

    N = Z.shape[0]
    if N == 0:
        raise RuntimeError("No latent samples to run t-SNE.")

    if n_subset > 0 and N > n_subset:
        idx = np.random.choice(N, size=n_subset, replace=False)
        Z_use = Z[idx]
        sel_idx = idx
    else:
        Z_use = Z
        sel_idx = np.arange(N)

    print(f"[TSNE] using {Z_use.shape[0]} samples for TSNE")
    tsne = TSNE(
        n_components=2,
        perplexity=30.0,
        learning_rate=200.0,
        n_iter=2000,
        verbose=1,
        init="random",
        random_state=42,
    )
    Z2 = tsne.fit_transform(Z_use)
    return Z2, sel_idx


def save_scatter(Z2, values, title, out_path, vmin=None, vmax=None, cmap="viridis"):
    """Helper to save a colored scatter plot."""
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=values, s=3, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(sc)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT] saved {out_path}")


def maybe_kmeans(Z: np.ndarray, k: int):
    """Optional KMeans on latent space."""
    if k <= 0 or Z.shape[0] == 0:
        return None
    from sklearn.cluster import KMeans

    print(f"[KMEANS] running KMeans with K={k} on {Z.shape[0]} samples")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Z)
    return labels


def main():
    parser = argparse.ArgumentParser(description="Latent topology analysis for VQVAE.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--kmeans_k", type=int, default=0)
    parser.add_argument("--tsne_subset", type=int, default=20000)
    parser.add_argument("--out_dir", type=str, default="./latent_analysis")
    parser.add_argument("--out_prefix", type=str, default="stage2")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] using {device}")

    model, dataset, _ = build_model_and_dataset(args)
    load_checkpoint(model, args.ckpt)

    Z, Ls, Hs, Es = extract_latent(
        model=model,
        dataset=dataset,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    if Z.shape[0] == 0:
        print("[WARN] no samples collected, exit.")
        return

    # compute loop fraction
    loops = 1.0 - Hs - Es
    loops = np.clip(loops, 0.0, 1.0)

    # optional kmeans on full latent space
    labels = maybe_kmeans(Z, args.kmeans_k)
    if labels is not None:
        np.save(os.path.join(args.out_dir, f"{args.out_prefix}_kmeans_labels.npy"), labels)
        print("[KMEANS] labels saved.")

    # t-SNE on subset
    Z2, sel_idx = run_tsne(Z, args.tsne_subset)

    Ls_sel = Ls[sel_idx]
    Hs_sel = Hs[sel_idx]
    Es_sel = Es[sel_idx]
    Loops_sel = loops[sel_idx]

    # save numpy for future processing
    np.savez(
        os.path.join(args.out_dir, f"{args.out_prefix}_tsne_data.npz"),
        Z2=Z2,
        length=Ls_sel,
        helix_frac=Hs_sel,
        sheet_frac=Es_sel,
        loop_frac=Loops_sel,
        idx=sel_idx,
    )
    print("[SAVE] latent tsne data saved.")

    # base path for plots
    base = os.path.join(args.out_dir, args.out_prefix)

    # helix
    save_scatter(
        Z2,
        Hs_sel,
        "TSNE colored by helix_fraction",
        base + "_tsne_helix.png",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )

    # sheet
    save_scatter(
        Z2,
        Es_sel,
        "TSNE colored by sheet_fraction",
        base + "_tsne_sheet.png",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )

    # loop
    save_scatter(
        Z2,
        Loops_sel,
        "TSNE colored by loop_fraction",
        base + "_tsne_loop.png",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )

    # length
    save_scatter(
        Z2,
        Ls_sel,
        "TSNE colored by length",
        base + "_tsne_length.png",
        cmap="viridis",
    )

    # clusters (only drawn if kmeans_k > 0)
    if labels is not None:
        labels_sel = labels[sel_idx]
        # use a discrete colormap for clusters
        save_scatter(
            Z2,
            labels_sel,
            "TSNE colored by kmeans_cluster",
            base + "_tsne_kmeans.png",
            cmap="tab20",
        )


if __name__ == "__main__":
    main()
