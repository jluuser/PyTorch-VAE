#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
python scripts/playground/export_ze_tokens.py \
  --ckpt checkpoints/vq_token64_K1024_D512_ResidualVQ_fromscratch/epochepoch=139.ckpt \
  --config configs/stage2_vq.yaml \
  --npy_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/prp-dataset/filtered_curves_npy/ \
  --train_list train_list.txt \
  --val_list val_list.txt \
  --out_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/prp-dataset/ze_tokens_len1_80 \
  --shard_size 10000 \
  --min_len 1 \
  --max_len 80 \
  --batch_size 256 \
  --num_workers 16 \
  --device cuda \
  --amp
'''  
import os
import sys
import argparse
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

import yaml

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.vq_vae import VQVAE
from dataset import CurveDataset, pad_collate


def parse_args():
    p = argparse.ArgumentParser("Export ze token latents for full dataset")

    # model / config
    p.add_argument("--ckpt", type=str, required=True, help="VQVAE checkpoint path")
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="optional YAML config; if set, overrides model hyperparams",
    )

    # data
    p.add_argument("--npy_dir", type=str, required=True)
    p.add_argument(
        "--train_list",
        type=str,
        default="train_list.txt",
        help="relative or absolute path to train list",
    )
    p.add_argument(
        "--val_list",
        type=str,
        default="val_list.txt",
        help="relative or absolute path to val list",
    )

    # model hyperparams (will be overridden by config if provided)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=350)
    p.add_argument("--code_dim", type=int, default=512)
    p.add_argument("--codebook_size", type=int, default=1024)
    p.add_argument("--num_quantizers", type=int, default=4)
    p.add_argument("--latent_tokens", type=int, default=64)

    # export
    p.add_argument(
        "--out_dir",
        type=str,
        default="ze_exports",
        help="output directory for ze shards",
    )
    p.add_argument(
        "--shard_size",
        type=int,
        default=10000,
        help="number of curves per shard npz file",
    )

    # length filter
    p.add_argument(
        "--min_len",
        type=int,
        default=1,
        help="minimum true length to keep (inclusive)",
    )
    p.add_argument(
        "--max_len",
        type=int,
        default=1000000,
        help="maximum true length to keep (inclusive)",
    )

    # loader / device
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def strip_prefixes(state_dict, prefixes=("model.", "module.", "net.")):
    out = {}
    for k, v in state_dict.items():
        name = k
        for p in prefixes:
            if name.startswith(p):
                name = name[len(p) :]
                break
        out[name] = v
    return out


def read_list_file(list_path: str) -> List[str]:
    names = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(".json"):
                continue
            names.append(line)
    return names


@torch.no_grad()
def export_split(
    split_name: str,
    model: VQVAE,
    npy_dir: str,
    list_path: str,
    out_dir: str,
    shard_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
    min_len: int,
    max_len: int,
):
    print(f"[Export] split={split_name}, list={list_path}")
    if not os.path.isfile(list_path):
        raise FileNotFoundError(f"list file not found: {list_path}")

    rel_paths_all = read_list_file(list_path)
    rel_paths_all = np.array(rel_paths_all, dtype=object)
    num_items = rel_paths_all.shape[0]
    print(f"[Export] split={split_name}, num_items={num_items}")
    print(f"[Export] length filter: [{min_len}, {max_len}]")

    ds = CurveDataset(npy_dir=npy_dir, list_path=list_path, train=False)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_collate,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    split_out_dir = os.path.join(out_dir, split_name)
    os.makedirs(split_out_dir, exist_ok=True)

    shard_ze = []
    shard_lengths = []
    shard_rel_paths = []

    global_idx = 0
    shard_idx = 0
    kept_total = 0

    try:
        autocast_ctx = torch.amp.autocast(
            device_type="cuda", enabled=(use_amp and device.type == "cuda")
        )
    except Exception:
        autocast_ctx = torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda"))

    model.eval()
    with autocast_ctx:
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, mask = batch
            else:
                x, mask = batch, None

            B = x.size(0)
            x = x.to(device, non_blocking=True)
            if mask is not None:
                mask = mask.to(device, non_blocking=True)
            else:
                mask = torch.ones((B, x.size(1)), dtype=torch.bool, device=device)

            # dataset indices for this batch
            idx_this = np.arange(global_idx, global_idx + B)
            rel_paths_batch = rel_paths_all[idx_this]

            # true sequence lengths
            lengths = mask.sum(dim=1).to(torch.int32)  # [B]
            keep = (lengths >= min_len) & (lengths <= max_len)  # [B] bool

            if not torch.any(keep):
                global_idx += B
                continue

            # encoder + tokenizer -> ze tokens [B, N_tokens, D]
            h_fuse, _, _ = model.encode(x, mask=mask)
            z_e_tok = model._tokenize_to_codes(h_fuse, mask)  # [B, N_tokens, D]

            z_e_tok_kept = z_e_tok[keep]  # [K, N_tokens, D]
            lengths_kept = lengths[keep].cpu().numpy()
            rel_paths_batch_kept = rel_paths_batch[keep.cpu().numpy()]

            shard_ze.append(
                z_e_tok_kept.cpu().numpy().astype(np.float32, copy=False)
            )
            shard_lengths.append(lengths_kept)
            shard_rel_paths.append(rel_paths_batch_kept)

            kept_total += z_e_tok_kept.shape[0]
            global_idx += B

            # flush shard if large enough
            num_in_shard = sum(arr.shape[0] for arr in shard_ze)
            if num_in_shard >= shard_size:
                save_shard(
                    split_out_dir,
                    split_name,
                    shard_idx,
                    shard_ze,
                    shard_lengths,
                    shard_rel_paths,
                )
                shard_idx += 1
                shard_ze = []
                shard_lengths = []
                shard_rel_paths = []

    # flush last shard
    if shard_ze:
        save_shard(
            split_out_dir,
            split_name,
            shard_idx,
            shard_ze,
            shard_lengths,
            shard_rel_paths,
        )

    print(
        f"[Export] done split={split_name}, total_items={num_items}, kept_items={kept_total}"
    )


def save_shard(
    split_out_dir: str,
    split_name: str,
    shard_idx: int,
    shard_ze_list,
    shard_lengths_list,
    shard_rel_paths_list,
):
    ze = np.concatenate(shard_ze_list, axis=0)  # [M, N_tokens, D]
    lengths = np.concatenate(shard_lengths_list, axis=0)  # [M]
    rel_paths = np.concatenate(shard_rel_paths_list, axis=0)  # [M]

    out_name = f"ze_{split_name}_shard{shard_idx:03d}.npz"
    out_path = os.path.join(split_out_dir, out_name)

    np.savez(
        out_path,
        ze_tokens=ze,
        lengths=lengths,
        rel_paths=rel_paths,
    )
    print(
        f"[Shard] saved {out_path} | ze_tokens.shape={ze.shape}, lengths.shape={lengths.shape}"
    )


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    # optional config override (stage2 VQ-VAE yaml)
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        mp = (cfg or {}).get("model_params", {})
        for k_src, k_dst in [
            ("hidden_dim", "hidden_dim"),
            ("num_layers", "num_layers"),
            ("num_heads", "num_heads"),
            ("max_seq_len", "max_seq_len"),
            ("code_dim", "code_dim"),
            ("latent_tokens", "latent_tokens"),
            ("num_quantizers", "num_quantizers"),
            ("codebook_size", "codebook_size"),
        ]:
            if k_src in mp:
                setattr(args, k_dst, int(mp[k_src]))
        print(
            "[Config] override from {}: hidden_dim={}, num_layers={}, num_heads={}, "
            "max_seq_len={}, code_dim={}, latent_tokens={}, num_quantizers={}, codebook_size={}".format(
                args.config,
                args.hidden_dim,
                args.num_layers,
                args.num_heads,
                args.max_seq_len,
                args.code_dim,
                args.latent_tokens,
                args.num_quantizers,
                args.codebook_size,
            )
        )

    os.makedirs(args.out_dir, exist_ok=True)

    model = VQVAE(
        input_dim=6,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        use_vq=True,
        residual_vq=(int(args.num_quantizers) > 1),
        num_quantizers=int(args.num_quantizers),
        codebook_size=int(args.codebook_size),
        code_dim=args.code_dim,
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
        latent_tokens=int(args.latent_tokens),
        tokenizer_heads=args.num_heads,
        tokenizer_layers=2,
        tokenizer_dropout=0.1,
        print_init=False,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = strip_prefixes(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[Load] missing={} unexpected={}".format(len(missing), len(unexpected)))

    # train/val list absolute paths
    if os.path.isabs(args.train_list):
        train_list_path = args.train_list
    else:
        train_list_path = os.path.join(args.npy_dir, args.train_list)

    if os.path.isabs(args.val_list):
        val_list_path = args.val_list
    else:
        val_list_path = os.path.join(args.npy_dir, args.val_list)

    # export train split
    export_split(
        split_name="train",
        model=model,
        npy_dir=args.npy_dir,
        list_path=train_list_path,
        out_dir=args.out_dir,
        shard_size=args.shard_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        use_amp=bool(args.amp),
        min_len=int(args.min_len),
        max_len=int(args.max_len),
    )

    # export val split
    export_split(
        split_name="val",
        model=model,
        npy_dir=args.npy_dir,
        list_path=val_list_path,
        out_dir=args.out_dir,
        shard_size=args.shard_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        use_amp=bool(args.amp),
        min_len=int(args.min_len),
        max_len=int(args.max_len),
    )

    print("[Export] all splits done.")


if __name__ == "__main__":
    main()
