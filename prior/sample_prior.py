#!/usr/bin/env python3
# coding: utf-8

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

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

    num_q = _get_int(sp, ["num_quantizers", "num_q", "nq"])
    if num_q is None:
        num_q = 4

    return int(K), int(PAD), int(BOS), int(EOS), int(V), int(num_q)


def _per_level_sampling_params(
    level: int,
    num_quantizers: int,
    base_temperature: float,
    base_top_p: float,
    base_repetition_penalty: float,
):
    """
    Heuristic schedule for RVQ sampling:
      - Level 0 (Coarse): Uses base params (high temp) for structural diversity.
      - Level 3 (Fine): Uses lower temp and higher repetition penalty to clean up noise.
    """
    level = int(level)
    num_quantizers = max(1, int(num_quantizers))

    t = float(base_temperature)
    p = float(base_top_p)
    rp = float(base_repetition_penalty)

    # Heuristic: Decrease temperature for finer levels to reduce high-frequency noise
    # e.g. L0=1.0 -> L3=0.85
    if t > 0.0:
        t = max(0.5, t - 0.05 * level)

    # Heuristic: Tighten top_p slightly for finer levels
    if 0.0 < p < 1.0:
        p = max(0.8, min(0.99, p - 0.02 * level))

    # Heuristic: Increase repetition penalty for finer levels to prevent local loops
    if rp >= 1.0:
        rp = rp + 0.05 * level

    return t, p, rp


@torch.no_grad()
def generate_fixed_len(
    model: TransformerPriorLM,
    latent_len: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    num_quantizers: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    tokens = torch.full((1, 1), int(bos_id), dtype=torch.long, device=device)
    attn = torch.ones_like(tokens, dtype=torch.bool)

    out_codes = []

    for step in range(int(latent_len)):
        logits = model(tokens, attn_mask=attn)[:, -1, :]

        # Determine current RVQ level
        level = step % max(1, int(num_quantizers))
        
        # Apply per-level sampling params
        t_l, p_l, rp_l = _per_level_sampling_params(
            level=level,
            num_quantizers=num_quantizers,
            base_temperature=temperature,
            base_top_p=top_p,
            base_repetition_penalty=repetition_penalty,
        )

        if t_l > 0.0:
            logits = logits / max(float(t_l), 1e-8)

        # Mask special tokens
        logits[:, int(pad_id)] = float("-inf")
        logits[:, int(bos_id)] = float("-inf")
        logits[:, int(eos_id)] = float("-inf")

        # Apply repetition penalty
        if rp_l > 1.0 and tokens.numel() > 0:
            prev = tokens[0].unique()
            # Exclude specials from penalty
            prev = prev[(prev != int(pad_id)) & (prev != int(bos_id)) & (prev != int(eos_id))]
            if prev.numel() > 0:
                gathered = logits.index_select(-1, prev)
                # Apply penalty: multiply if negative logit, divide if positive
                gathered = torch.where(gathered < 0, gathered * rp_l, gathered / rp_l)
                logits.scatter_(-1, prev.unsqueeze(0), gathered)

        # Filtering
        logits = model._top_k_top_p_filtering(
            logits, 
            top_k=int(top_k), 
            top_p=float(p_l)
        )
        
        probs = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        out_codes.append(int(next_tok.item()))

        tokens = torch.cat([tokens, next_tok], dim=1)
        attn = torch.ones_like(tokens, dtype=torch.bool)

    return torch.tensor(out_codes, dtype=torch.long, device=device)


def build_length_sampler(train_manifest: str, min_target_len: int, max_target_len: int):
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
            f"fallback to uniform [{min_target_len}, {max_target_len}]"
        )
        return None

    arr = np.asarray(lengths_in_range, dtype=np.int32)
    print(
        f"[info] loaded lengths from {path}: "
        f"total_valid={len(lengths_all)}, in_range={arr.shape[0]}"
    )

    def sampler() -> int:
        idx = np.random.randint(0, arr.shape[0])
        return int(arr[idx])

    return sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_ckpt", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--latent_len", type=int, default=256)

    ap.add_argument("--min_target_len", type=int, default=1)
    ap.add_argument("--target_len", type=int, default=350)
    ap.add_argument("--train_manifest", type=str, default="")

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--indices_dtype", type=str, default="int32", choices=["int16", "int32"])
    args = ap.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = args.device
    else:
        device = "cpu"

    max_target_len = int(args.target_len)
    min_target_len = int(args.min_target_len)

    print(f"[info] loading prior from {args.prior_ckpt}")
    ckpt = torch.load(args.prior_ckpt, map_location="cpu")
    sp = ckpt.get("special_tokens", None)
    if not isinstance(sp, dict):
        raise KeyError("checkpoint missing special_tokens dict")

    K, PAD, BOS, EOS, V, num_q_sp = _infer_specials(sp)

    cfg = ckpt.get("cfg", {}) or {}
    mcfg = cfg.get("model", {}) or {}
    dcfg = cfg.get("data", {}) or {}

    d_model = int(mcfg.get("d_model", 768))
    n_layers = int(mcfg.get("n_layers", 10))
    n_heads = int(mcfg.get("n_heads", 8))
    ffw_mult = int(mcfg.get("ffw_mult", 4))
    dropout = float(mcfg.get("dropout", 0.2))
    tie_embeddings = bool(mcfg.get("tie_embeddings", True))
    layer_norm_eps = float(mcfg.get("layer_norm_eps", 1e-5))
    use_hier = bool(mcfg.get("use_hierarchical_attn", True))

    num_q = int(mcfg.get("num_quantizers", num_q_sp))
    max_code_len = int(dcfg.get("max_len", int(args.latent_len)))

    model = TransformerPriorLM(
        vocab_size=V,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffw_mult=ffw_mult,
        dropout=dropout,
        max_code_len=max_code_len,
        num_quantizers=num_q,
        tie_embeddings=tie_embeddings,
        layer_norm_eps=layer_norm_eps,
        use_hierarchical_attn=use_hier,
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

    print(f"[info] generating {args.num_samples} samples to {out_dir} ...")
    meta = []

    # Cache base args
    base_temp = float(args.temperature)
    base_p = float(args.top_p)
    base_k = int(args.top_k)
    base_rep = float(args.repetition_penalty)

    for i in range(int(args.num_samples)):
        # Call generate with explicit num_quantizers to enable per-level sampling
        toks = generate_fixed_len(
            model=model,
            latent_len=int(args.latent_len),
            bos_id=BOS,
            eos_id=EOS,
            pad_id=PAD,
            num_quantizers=num_q,
            temperature=base_temp,
            top_k=base_k,
            top_p=base_p,
            repetition_penalty=base_rep,
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
            "indices_path": str(npy_path.resolve()),
            "latent_len": int(args.latent_len),
            "length": int(args.latent_len),
            "target_len": int(sampled_target_len),
            "dtype": dtype_str,
            "K": int(K),
            "num_quantizers": int(num_q),
        }
        meta.append(rec)

    manifest_path = out_dir / "samples_manifest.jsonl"
    with manifest_path.open("w") as f:
        for r in meta:
            f.write(json.dumps(r) + "\n")

    print(f"[success] Saved {len(meta)} sequences.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()