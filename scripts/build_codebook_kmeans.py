# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Build multi-level (residual) VQ codebook from Stage-1 AE token latents.

Example:

python scripts/build_codebook_kmeans.py \
  --ckpt /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/checkpoints/ae_1024_512_residualVQ/last.ckpt \
  --data_dir /public/home/zhangyangroup/chengshiz/keyuan.zhou/prp-dataset/filtered_curves_npy \
  --train_list train_list.txt \
  --out /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/scripts/kmeans_residual_centroids_L4x1024x512.npy \
  --codebook_size 1024 \
  --code_dim 512 \
  --latent_n_tokens 64 \
  --num_quantizers 4 \
  --batch_size 512 \
  --num_workers 16 \
  --amp \
  --max_samples 400000 \
  --threads 128 \
  --devices 4 \
  --torch_kmeans_max_iter 120 \
  --torch_kmeans_batch_size 100000
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_SKLEARN_OK = True
try:
    from sklearn.cluster import MiniBatchKMeans
except Exception:
    _SKLEARN_OK = False

_FAISS_OK = False
try:
    import faiss
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.vq_vae import VQVAE
from dataset import CurveDataset, pad_collate


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


def set_num_threads(n: int):
    n = int(max(1, n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    try:
        torch.set_num_threads(n)
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="Build residual VQ codebook from Stage-1 AE token latents.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--train_list", type=str, required=True)
    p.add_argument("--out", type=str, required=True)

    p.add_argument("--codebook_size", type=int, default=512)
    p.add_argument("--code_dim", type=int, default=128)
    p.add_argument("--latent_n_tokens", type=int, default=48)
    p.add_argument("--num_quantizers", type=int, default=4, help="Number of residual VQ levels (L).")

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=300000)

    p.add_argument("--use_faiss", action="store_true")
    p.add_argument("--kmeans_batch_size", type=int, default=100000)
    p.add_argument("--kmeans_max_iter", type=int, default=120)
    p.add_argument("--kmeans_n_init", type=int, default=1)
    p.add_argument("--faiss_niter", type=int, default=100)
    p.add_argument("--faiss_nredo", type=int, default=1)

    p.add_argument("--devices", type=int, default=1, help="Number of CUDA devices for torch k-means / assignment.")
    p.add_argument("--torch_kmeans_max_iter", type=int, default=100)
    p.add_argument("--torch_kmeans_batch_size", type=int, default=100000)
    return p.parse_args()


class LatentExtractor(nn.Module):
    def __init__(self, model: VQVAE):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        h_fuse, _, _ = self.model.encode(x, mask=mask)
        z_tokens = self.model._tokenize_to_codes(h_fuse, mask)
        return z_tokens


@torch.no_grad()
def collect_token_latents(
    model: VQVAE,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    max_rows: int,
    dp_devices: int,
) -> torch.Tensor:
    rows = []
    n_rows = 0
    try:
        autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda"))
    except Exception:
        autocast_ctx = torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda"))

    extractor = LatentExtractor(model)
    if device.type == "cuda" and int(dp_devices) > 1:
        extractor = nn.DataParallel(extractor, device_ids=list(range(int(dp_devices)))).to(device)
    else:
        extractor = extractor.to(device)

    extractor.eval()
    with autocast_ctx:
        for batch in tqdm(loader, desc="Collecting token latents", leave=True):
            if isinstance(batch, (list, tuple)):
                x, mask = batch
            else:
                x, mask = batch, None
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True) if mask is not None else None

            z_tokens = extractor(x, mask)
            z_tokens = z_tokens.contiguous().view(-1, z_tokens.size(-1))

            rows.append(z_tokens.cpu())
            n_rows += z_tokens.size(0)
            if n_rows >= max_rows:
                break

    if not rows:
        raise RuntimeError("No latents collected.")
    z = torch.cat(rows, dim=0)[:max_rows]
    return z  # [N, D]


def kmeans_sklearn(
    latents_np: np.ndarray,
    k: int,
    batch_size: int,
    max_iter: int,
    n_init: int,
) -> np.ndarray:
    if not _SKLEARN_OK:
        raise RuntimeError("scikit-learn not found.")
    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init=n_init,
        verbose=1,
        compute_labels=False,
        init="k-means++",
        reassignment_ratio=0.01,
        random_state=42,
    )
    km.fit(latents_np)
    return km.cluster_centers_.astype(np.float32)


def kmeans_faiss(latents_np: np.ndarray, k: int, niter: int, nredo: int) -> np.ndarray:
    if not _FAISS_OK:
        raise RuntimeError("FAISS not detected.")
    d = latents_np.shape[1]
    km = faiss.Kmeans(
        d=d,
        k=k,
        niter=niter,
        nredo=nredo,
        verbose=True,
        seed=42,
        gpu=faiss.get_num_gpus() > 0,
    )
    km.train(latents_np)
    return km.centroids.astype(np.float32)


def _closest_centroid_multi_gpu(x: torch.Tensor, C: torch.Tensor, ngpu: int) -> torch.Tensor:
    if ngpu <= 1 or x.size(0) < 20000:
        d2 = (
            x.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * (x @ C.t())
            + (C.pow(2).sum(dim=1, keepdim=True)).t()
        )
        return torch.argmin(d2, dim=1)

    chunks = torch.chunk(x, ngpu, dim=0)
    idx_parts = []
    C_bcasts = [C.to(f"cuda:{i}") for i in range(ngpu)]
    for i, part in enumerate(chunks):
        if part.numel() == 0:
            continue
        dev = torch.device(f"cuda:{i}")
        part = part.to(dev, non_blocking=True)
        C_i = C_bcasts[i]
        d2 = (
            part.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * (part @ C_i.t())
            + (C_i.pow(2).sum(dim=1, keepdim=True)).t()
        )
        idx_local = torch.argmin(d2, dim=1).to("cuda:0", non_blocking=True)
        idx_parts.append(idx_local)
    return torch.cat(idx_parts, dim=0)


def kmeans_torch(
    latents: torch.Tensor,
    k: int,
    max_iter: int = 100,
    batch_size: int = 100000,
    devices: int = 1,
) -> np.ndarray:
    assert latents.is_cuda, "latents must be on CUDA for torch k-means."
    N, D = latents.shape
    K = int(k)
    ngpu = int(devices)

    perm = torch.randperm(N, device=latents.device)
    C = latents[perm[:K]].contiguous()

    counts = torch.zeros(K, device="cuda:0", dtype=torch.long)
    sums = torch.zeros(K, D, device="cuda:0", dtype=latents.dtype)

    for it in range(int(max_iter)):
        counts.zero_()
        sums.zero_()

        for start in range(0, N, int(batch_size)):
            end = min(N, start + int(batch_size))
            xb = latents[start:end]

            idx = _closest_centroid_multi_gpu(xb, C, ngpu)
            sums.index_add_(0, idx, xb)
            one = torch.ones_like(idx, dtype=torch.long, device=idx.device)
            counts.index_add_(0, idx, one)

        mask = counts > 0
        C_new = C.clone()
        C_new[mask] = (sums[mask] / counts[mask].unsqueeze(1).to(sums.dtype))
        if (~mask).any():
            need = (~mask).nonzero(as_tuple=True)[0]
            repl_idx = torch.randint(0, N, (need.numel(),), device="cuda:0")
            C_new[need] = latents[repl_idx]

        shift = (C_new - C).pow(2).sum().sqrt()
        C = C_new
        if float(shift.item()) < 1e-4:
            break

    return C.detach().float().cpu().numpy()


def _subtract_closest_cpu(x: torch.Tensor, C: torch.Tensor, batch_size: int = 100000) -> torch.Tensor:
    """
    CPU fallback for residual update: x -> x - nearest(C).
    x: [N, D], C: [K, D]
    """
    N, D = x.shape
    out = torch.empty_like(x)
    K = C.size(0)
    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        xb = x[start:end]  # [B, D]
        d2 = (
            xb.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * (xb @ C.t())
            + (C.pow(2).sum(dim=1, keepdim=True)).t()
        )  # [B, K]
        idx = torch.argmin(d2, dim=1)  # [B]
        out[start:end] = xb - C[idx]
    return out


def build_residual_codebooks(latents: torch.Tensor, args) -> np.ndarray:
    """
    latents: [N, D] torch tensor on CPU.
    Returns: codebooks with shape [L, K, D], where
      L = num_quantizers (residual levels),
      K = codebook_size,
      D = code_dim.
    """
    latents = latents.float()
    N, D = latents.shape
    L = int(args.num_quantizers)
    K = int(args.codebook_size)

    if L <= 0:
        raise ValueError(f"num_quantizers must be >= 1, got {L}")

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        base_device = torch.device("cuda:0")
        residual = latents.to(base_device)
    else:
        base_device = torch.device("cpu")
        residual = latents

    all_centroids = []

    print(f"[Residual KMeans] N={N}, D={D}, levels={L}, K={K}")
    for level in range(L):
        print(f"[Level {level + 1}/{L}] Running k-means on residuals: shape={tuple(residual.shape)}")

        # 1) Run k-means on current residuals
        if args.use_faiss and _FAISS_OK:
            lat_np = residual.detach().cpu().numpy()
            C_l = kmeans_faiss(
                lat_np,
                k=K,
                niter=int(args.faiss_niter),
                nredo=int(args.faiss_nredo),
            )
        elif (not args.use_faiss) and _SKLEARN_OK:
            lat_np = residual.detach().cpu().numpy()
            C_l = kmeans_sklearn(
                latents_np=lat_np,
                k=K,
                batch_size=int(args.kmeans_batch_size),
                max_iter=int(args.kmeans_max_iter),
                n_init=int(args.kmeans_n_init),
            )
        else:
            if not use_cuda:
                raise RuntimeError("Torch k-means backend requires CUDA but CUDA is not available.")
            lat_cuda = residual.to(base_device)
            C_l = kmeans_torch(
                lat_cuda,
                k=K,
                max_iter=int(args.torch_kmeans_max_iter),
                batch_size=int(args.torch_kmeans_batch_size),
                devices=int(args.devices),
            )

        C_l = C_l.astype(np.float32)
        all_centroids.append(C_l)

        # 2) Update residuals: residual <- residual - nearest_centroid(C_l)
        print(f"[Level {level + 1}/{L}] Updating residuals...")
        if use_cuda:
            C_t = torch.from_numpy(C_l).to(base_device)
            with torch.no_grad():
                idx = _closest_centroid_multi_gpu(residual, C_t, int(args.devices))
                residual = residual - C_t[idx]
        else:
            C_t = torch.from_numpy(C_l)
            with torch.no_grad():
                residual = _subtract_closest_cpu(
                    residual,
                    C_t,
                    batch_size=int(args.kmeans_batch_size),
                )

    C_all = np.stack(all_centroids, axis=0)  # [L, K, D]
    return C_all


def main():
    args = parse_args()
    if args.threads and args.threads > 0:
        set_num_threads(args.threads)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    list_path = args.train_list if os.path.isabs(args.train_list) else os.path.join(args.data_dir, args.train_list)

    dataset = CurveDataset(npy_dir=args.data_dir, list_path=list_path, train=True)
    print(f"[Dataset] Found {len(dataset)} curves from {args.data_dir}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_collate,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    model = VQVAE(
        input_dim=6,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        max_seq_len=350,
        use_vq=False,  # pure AE to extract continuous latents
        codebook_size=args.codebook_size,
        code_dim=args.code_dim,
        beta=0.0,
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
        latent_tokens=int(args.latent_n_tokens),
        tokenizer_heads=8,
        tokenizer_layers=2,
        tokenizer_dropout=0.1,
        print_init=False,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = strip_prefixes(ckpt["state_dict"])
    else:
        state_dict = strip_prefixes(ckpt if isinstance(ckpt, dict) else {})
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Info] Loaded AE checkpoint: {args.ckpt}")
    print(f"[Info] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    t0 = time.time()
    latents = collect_token_latents(
        model=model,
        loader=loader,
        device=device,
        use_amp=bool(args.amp),
        max_rows=int(args.max_samples),
        dp_devices=int(args.devices),
    )
    t1 = time.time()
    print(f"[Info] Collected {latents.shape[0]} rows of dim {latents.shape[1]} | time={t1 - t0:.2f}s")

    C_all = build_residual_codebooks(latents, args)
    print(f"[Info] Final residual codebooks shape: {C_all.shape} (L, K, D)")

    np.save(args.out, C_all.astype(np.float32))
    print(f"[Done] Saved residual codebooks to: {args.out}")


if __name__ == "__main__":
    main()

