# scripts/extract_ae_latents.py
import argparse
import torch
import sys
import os
from pathlib import Path
from tqdm import tqdm

"""
Example:
python scripts/extract_ae_latents.py \
  --config configs/stage1_ae.yaml \
  --ckpt /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/checkpoints/aeot_sigmoid/epochepoch=epoch=089.ckpt \
  --out data/ae_sigmoid_latents_len1_80.pt \
  --batch_size 512 \
  --device cuda \
  --len_min 1 \
  --len_max 80
"""

# Add repo root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from experiment import build_experiment_from_yaml
from dataset import pad_collate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to stage1_ae.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to AE checkpoint")
    parser.add_argument("--out", type=str, required=True, help="Output .pt file")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")

    # Length filtering
    parser.add_argument("--len_min", type=int, default=1, help="Keep samples with length >= len_min")
    parser.add_argument("--len_max", type=int, default=80, help="Keep samples with length <= len_max")

    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    len_min = int(args.len_min)
    len_max = int(args.len_max)
    if len_min <= 0:
        raise ValueError("--len_min must be >= 1")
    if len_max > 0 and len_max < len_min:
        raise ValueError("--len_max must be >= len_min (or set <=0 to disable)")

    # 1) Load AE
    print(f"[Info] Loading AE from {args.ckpt}...")
    exp, _ = build_experiment_from_yaml(args.config)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Safely strip Lightning "model." prefix (only when it exists)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v

    exp.model.load_state_dict(new_state_dict, strict=False)

    model = exp.model
    model.eval().to(args.device)

    # 2) Setup Data
    exp.setup(stage="fit")
    dataloader = torch.utils.data.DataLoader(
        exp.train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=True,
    )

    all_latents = []
    all_lengths = []

    kept = 0
    skipped = 0

    print(f"[Info] Extracting latents with length filter: [{len_min}, {len_max}] ...")
    for batch in tqdm(dataloader):
        x, mask = batch
        x = x.to(args.device, non_blocking=True)
        mask = mask.to(args.device, non_blocking=True)

        # mask: [B, L] bool, True means valid token
        lengths = mask.long().sum(dim=1)  # [B]

        # Select indices satisfying the length range
        if len_max > 0:
            keep_mask = (lengths >= len_min) & (lengths <= len_max)
        else:
            keep_mask = (lengths >= len_min)

        if not torch.any(keep_mask):
            skipped += int(lengths.numel())
            continue

        x_kept = x[keep_mask]
        mask_kept = mask[keep_mask]
        lengths_kept = lengths[keep_mask]

        kept += int(lengths_kept.numel())
        skipped += int(lengths.numel() - lengths_kept.numel())

        # Encode: [B_kept, Tokens, D]
        h_fuse, _, _ = model.encode(x_kept, mask=mask_kept)
        z_tokens = model._tokenize_to_codes(h_fuse, mask_kept)

        # Flatten: [B_kept, Tokens * D]
        z_flat = z_tokens.reshape(z_tokens.size(0), -1)

        all_latents.append(z_flat.cpu())
        all_lengths.append(lengths_kept.cpu())

    if len(all_latents) == 0:
        raise RuntimeError("No samples matched the requested length range. Nothing to save.")

    all_latents = torch.cat(all_latents, dim=0).contiguous()  # [N, D_flat]
    all_lengths = torch.cat(all_lengths, dim=0).contiguous()  # [N]

    # 3) Compute statistics for normalization (per-dimension)
    print("[Info] Computing statistics...")
    mean = all_latents.mean(dim=0)
    std = all_latents.std(dim=0) + 1e-6  # prevent division by zero

    # 4) Save
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_dict = {
        "latents": all_latents,                  # [N, D_flat] float32 (CPU)
        "lengths": all_lengths.to(torch.int32),  # [N] int32 (CPU)
        "mean": mean,                            # [D_flat]
        "std": std,                              # [D_flat]
        "latent_tokens": getattr(model, "latent_n_tokens", None),
        "code_dim": getattr(model, "code_dim", None),
        "len_min": len_min,
        "len_max": len_max,
        "kept": kept,
        "skipped": skipped,
    }
    torch.save(save_dict, args.out)

    # Basic sanity prints
    lengths_min = int(all_lengths.min().item())
    lengths_max = int(all_lengths.max().item())
    lengths_mean = float(all_lengths.float().mean().item())

    print(f"[Info] Kept {kept} samples, skipped {skipped} samples.")
    print(f"[Info] Saved {all_latents.size(0)} samples to {args.out}")
    print(f"[Info] Latent shape: {tuple(all_latents.shape)}")
    print(f"[Info] Lengths: min={lengths_min}, mean={lengths_mean:.2f}, max={lengths_max}")
    print(f"[Info] Mean norm: {mean.norm():.4f}, Std mean: {std.mean():.4f}")


if __name__ == "__main__":
    main()
