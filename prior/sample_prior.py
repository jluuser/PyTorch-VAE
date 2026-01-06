#!/usr/bin/env python3
# coding: utf-8

# prior/sample_prior.py

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

"""
Example:

python prior/sample_prior.py \
  --prior_ckpt /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/checkpoints/best_prior_ckpts/prior_best.pt \
  --num_samples 300 \
  --latent_len 48 \
  --min_target_len 64 \
  --target_len 350 \
  --temperature 1.0 \
  --top_k 0 \
  --top_p 1.0 \
  --indices_dtype int16 \
  --out_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/prior/new_prior_samples_300 \
  --device cuda \
  --train_manifest /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/prior/out_prior_data/train/manifest.jsonl
"""

# add repo root (PyTorch-VAE) to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from prior.models.prior_transformer import TransformerPriorLM


def _get_int(d, keys):
    for k in keys:
        if k in d:
            try:
                return int(d[k])
            except Exception:
                pass
    return None


def _infer_specials(sp: dict):
    """
    Return (K, PAD, BOS, EOS, V) with robust fallbacks.
    Assumes the common convention: PAD=K, BOS=K+1, EOS=K+2, V=K+3 if not present.
    """
    K = _get_int(sp, ["K", "codebook_size", "num_codes"])
    if K is None:
        raise KeyError(f"special_tokens missing K. keys={sorted(sp.keys())}")

    PAD = _get_int(sp, ["PAD", "pad", "pad_id", "pad_token_id"])
    BOS = _get_int(sp, ["BOS", "bos", "bos_id", "bos_token_id"])
    EOS = _get_int(sp, ["EOS", "eos", "eos_id", "eos_token_id"])
    V = _get_int(sp, ["V", "vocab_size"])

    if PAD is None:
        PAD = int(K)
    if BOS is None:
        BOS = int(K + 1)
    if EOS is None:
        EOS = int(K + 2)
    if V is None:
        V = int(K + 3)

    return int(K), int(PAD), int(BOS), int(EOS), int(V)


@torch.no_grad()
def generate_fixed_len(
    model: TransformerPriorLM,
    latent_len: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate exactly latent_len VQ codes, masking PAD/BOS/EOS."""
    tokens = torch.full((1, 1), int(bos_id), dtype=torch.long, device=device)
    attn = torch.ones_like(tokens, dtype=torch.bool)
    out_codes = []

    for _ in range(int(latent_len)):
        logits = model(tokens, attn_mask=attn)[:, -1, :]
        logits = logits / max(float(temperature), 1e-8)

        # mask specials
        logits[:, int(pad_id)] = float("-inf")
        logits[:, int(bos_id)] = float("-inf")
        logits[:, int(eos_id)] = float("-inf")

        logits = model._top_k_top_p_filtering(logits, top_k=int(top_k), top_p=float(top_p))
        probs = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        out_codes.append(int(next_tok.item()))

        tokens = torch.cat([tokens, next_tok], dim=1)
        attn = torch.ones_like(tokens, dtype=torch.bool)

    return torch.tensor(out_codes, dtype=torch.long, device=device)


def build_length_sampler(train_manifest: str, min_target_len: int, max_target_len: int):
    """
    If train_manifest is provided, build a sampler that draws target_len
    from the empirical length distribution in the manifest, restricted to
    [min_target_len, max_target_len] (filtering, not clipping).

    Falls back to None if anything is wrong or no valid lengths remain.
    """
    if not train_manifest:
        return None

    path = Path(train_manifest)
    if not path.is_file():
        print(
            f"[warn] train_manifest not found: {path}, "
            f"fallback to uniform [{min_target_len}, {max_target_len}]"
        )
        return None

    lengths_all = []
    lengths_in_range = []

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            tlen = rec.get("target_len", None)
            if tlen is None:
                tlen = rec.get("length", None)
            if tlen is None:
                continue

            try:
                L = int(tlen)
            except Exception:
                continue

            if L <= 0:
                continue

            lengths_all.append(L)
            if int(min_target_len) <= L <= int(max_target_len):
                lengths_in_range.append(L)

    if not lengths_all:
        print(
            f"[warn] no valid lengths in {path}, "
            f"fallback to uniform [{min_target_len}, {max_target_len}]"
        )
        return None

    if not lengths_in_range:
        mn = int(min(lengths_all))
        mx = int(max(lengths_all))
        print(
            f"[warn] no lengths within [{min_target_len}, {max_target_len}] in {path}. "
            f"manifest_min={mn}, manifest_max={mx}. "
            f"fallback to uniform [{min_target_len}, {max_target_len}]"
        )
        return None

    arr = np.asarray(lengths_in_range, dtype=np.int32)
    print(
        f"[info] loaded lengths from {path}: "
        f"total_valid={len(lengths_all)}, in_range={arr.shape[0]}, "
        f"in_range_min={int(arr.min())}, in_range_max={int(arr.max())}"
    )

    def sampler() -> int:
        idx = np.random.randint(0, arr.shape[0])
        return int(arr[idx])

    return sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_ckpt", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--latent_len", type=int, default=48)

    ap.add_argument(
        "--min_target_len",
        type=int,
        default=1,
        help="Min target curve length. If --train_manifest is provided, lengths are sampled "
             "from the subset within [min_target_len, target_len] (filtering, not clipping). "
             "If --train_manifest is not set, target_len is sampled uniformly in "
             "[min_target_len, target_len].",
    )
    ap.add_argument(
        "--target_len",
        type=int,
        default=350,
        help="Max target curve length.",
    )
    ap.add_argument(
        "--train_manifest",
        type=str,
        default="",
        help="Optional: manifest.jsonl from training extraction. If provided, target_len is sampled "
             "from its in-range empirical length distribution.",
    )

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--indices_dtype",
        type=str,
        default="int32",
        choices=["int16", "int32"],
    )
    args = ap.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = args.device
    else:
        device = "cpu"

    max_target_len = int(args.target_len)
    min_target_len = int(args.min_target_len)
    if max_target_len <= 0:
        raise ValueError("target_len must be > 0")
    if min_target_len <= 0:
        raise ValueError("min_target_len must be > 0")
    if min_target_len > max_target_len:
        raise ValueError("min_target_len must be <= target_len")

    ckpt = torch.load(args.prior_ckpt, map_location="cpu")
    sp = ckpt.get("special_tokens", None)
    if not isinstance(sp, dict):
        raise KeyError("checkpoint missing special_tokens dict")

    K, PAD, BOS, EOS, V = _infer_specials(sp)

    cfg = ckpt.get("cfg", {})
    mcfg = cfg.get("model", {})

    d_model = int(mcfg.get("d_model", 512))
    n_layers = int(mcfg.get("n_layers", 8))
    n_heads = int(mcfg.get("n_heads", 8))
    ffw_mult = int(mcfg.get("ffw_mult", 4))
    dropout = float(mcfg.get("dropout", 0.1))
    tie_embeddings = bool(mcfg.get("tie_embeddings", True))
    layer_norm_eps = float(mcfg.get("layer_norm_eps", 1e-5))

    model = TransformerPriorLM(
        vocab_size=V,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffw_mult=ffw_mult,
        dropout=dropout,
        tie_embeddings=tie_embeddings,
        layer_norm_eps=layer_norm_eps,
        pad_token_id=PAD,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    length_sampler = build_length_sampler(
        train_manifest=args.train_manifest,
        min_target_len=min_target_len,
        max_target_len=max_target_len,
    )

    meta = []
    for i in range(int(args.num_samples)):
        toks = generate_fixed_len(
            model=model,
            latent_len=int(args.latent_len),
            bos_id=BOS,
            eos_id=EOS,
            pad_id=PAD,
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            device=device,
        ).cpu().numpy()

        if args.indices_dtype == "int16":
            arr = toks.astype(np.int16, copy=False)
            dtype_str = "int16"
        else:
            arr = toks.astype(np.int32, copy=False)
            dtype_str = "int32"

        npy_path = out_dir / f"sample_prior_{i:04d}.npy"
        np.save(npy_path, arr, allow_pickle=False)

        if length_sampler is not None:
            sampled_target_len = int(length_sampler())
        else:
            sampled_target_len = int(np.random.randint(min_target_len, max_target_len + 1))

        rec = {
            "i": int(i),
            "indices_path": str(npy_path),
            "latent_len": int(args.latent_len),
            "length": int(args.latent_len),
            "target_len": int(sampled_target_len),
            "dtype": dtype_str,
        }
        meta.append(rec)

    manifest_path = out_dir / "samples_manifest.jsonl"
    with manifest_path.open("w") as f:
        for r in meta:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(meta)} sequences to {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
