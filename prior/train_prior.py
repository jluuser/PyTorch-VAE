#!/usr/bin/env python3
# coding: utf-8
"""
DDP-capable training script for Transformer Prior over VQ-VAE code indices.

Launch:
  torchrun --nproc_per_node=4 prior/train_prior.py \
    --prior_cfg prior/configs/prior_config.yaml \
    --vq_yaml configs/stage2_vq.yaml

Enhancements:
- Optional length-bucketed batching (if data.bucket_boundaries is non-empty)
- Epoch-like progress estimation (steps_per_epoch, approx_epochs)
- Gradient accumulation via runtime.grad_accum_steps
- Flexible eval schedule: warm-up then early-eval then regular-eval
- Logging with elapsed time and ETA
- Lightweight checkpointing: save best & last; optional periodic saves with cleanup
"""

import os
import math
import glob
import time
from collections import deque
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

# 把项目根目录 (PyTorch-VAE) 加到 sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from prior.datasets.prior_dataset import PriorIndexDataset, collate_pad, BucketBatchSampler
from prior.models.prior_transformer import TransformerPriorLM



# ---------------------- utils ---------------------- #

def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_ddp():
    """Initialize DDP if torchrun env vars present."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def barrier():
    if is_dist():
        dist.barrier()


def try_get_codebook_from_prior_cfg(prior_cfg: Dict[str, Any]) -> Optional[int]:
    for k1 in ["model", "data", "vq", "model_params"]:
        node = prior_cfg.get(k1, {})
        if isinstance(node, dict) and "codebook_size" in node:
            return int(node["codebook_size"])
    if "codebook_size" in prior_cfg:
        return int(prior_cfg["codebook_size"])
    return None


def compute_vocab_and_specials(vq_yaml_path: Optional[str], prior_cfg: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    K = None
    if vq_yaml_path:
        vq_cfg = load_yaml(vq_yaml_path)
        K = vq_cfg.get("model_params", {}).get("codebook_size", None)
    if K is None:
        K = try_get_codebook_from_prior_cfg(prior_cfg)
    if K is None:
        raise ValueError("codebook_size not found. Provide --vq_yaml with model_params.codebook_size "
                         "or put codebook_size into prior_cfg.")
    PAD, BOS, EOS = K, K + 1, K + 2
    V = K + 3
    return K, PAD, BOS, EOS, V


def lr_lambda_builder(warmup_updates: int, max_updates: int):
    def fn(step):
        if step < warmup_updates:
            return step / max(1, warmup_updates)
        progress = (step - warmup_updates) / max(1, (max_updates - warmup_updates))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return fn


def _worker_init_fn(worker_id: int):
    base_seed = int(os.environ.get("PYTHONHASHSEED", "0")) & 0xFFFFFFFF
    rank = get_rank()
    seed = (base_seed + rank * 9973 + worker_id * 101) % 2**32
    torch.manual_seed(seed)


def build_loader_distributed(
    manifest_path: str, pad_id: int, bos_id: int, eos_id: int,
    max_len: int, batch_size: int, num_workers: int,
    shuffle: bool, pin_memory: bool, drop_last: bool, seed: int,
    is_train: bool
):
    dataset = PriorIndexDataset(
        manifest_path=manifest_path,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        max_len=max_len
    )
    if is_dist():
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=is_train,
            seed=seed
        )
        shuffle_flag = False
    else:
        sampler = None
        shuffle_flag = shuffle and is_train

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=lambda b: collate_pad(b, pad_id),
        worker_init_fn=_worker_init_fn if num_workers and num_workers > 0 else None,
        persistent_workers=bool(num_workers and num_workers > 0)
    )
    return loader, sampler, dataset


def build_loader_bucketed(
    manifest_path: str, pad_id: int, bos_id: int, eos_id: int,
    max_len: int, batch_size: int, num_workers: int,
    pin_memory: bool, drop_last: bool, seed: int,
    boundaries, is_train: bool
):
    dataset = PriorIndexDataset(
        manifest_path=manifest_path,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        max_len=max_len
    )
    sampler = BucketBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        boundaries=list(boundaries),
        seed=seed,
        shuffle=is_train,
        drop_last=drop_last,
        world_size=get_world_size(),
        rank=get_rank()
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda b: collate_pad(b, pad_id),
        worker_init_fn=_worker_init_fn if num_workers and num_workers > 0 else None,
        persistent_workers=bool(num_workers and num_workers > 0)
    )
    return loader, sampler, dataset


def set_epoch_if_possible(sampler, epoch: int):
    if sampler is not None:
        if hasattr(sampler, "set_epoch") and callable(sampler.set_epoch):
            sampler.set_epoch(epoch)


# ---------------------- eval & ckpt ---------------------- #

@torch.no_grad()
def evaluate(model: TransformerPriorLM, loader: DataLoader, sampler,
             pad_id: int, device: torch.device, amp: bool) -> float:
    model.eval()
    set_epoch_if_possible(sampler, 0)

    sum_loss_times_tokens = 0.0
    sum_tokens = 0

    for batch in loader:
        inp = batch["inp"].to(device, non_blocking=True)
        tgt = batch["tgt"].to(device, non_blocking=True)
        attn = batch["attn_mask"].to(device, non_blocking=True)

        with autocast(enabled=amp):
            logits = model(inp, attn_mask=attn)
            loss = model.loss(logits, tgt, ignore_index=pad_id)

        tok = (tgt != pad_id).sum().item()
        sum_loss_times_tokens += loss.item() * max(1, tok)
        sum_tokens += tok

    if is_dist():
        t = torch.tensor([sum_loss_times_tokens, float(sum_tokens)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        sum_loss_times_tokens = float(t[0].item())
        sum_tokens = int(t[1].item())

    model.train()
    return sum_loss_times_tokens / max(1, sum_tokens)


def save_ckpt(save_dir: str, name: str, model: TransformerPriorLM,
              cfg: Dict[str, Any], specials: Dict[str, int], step: int):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    torch.save({
        "model": model.state_dict(),
        "cfg": cfg,
        "special_tokens": specials,
        "global_step": step
    }, path)
    return path


def cleanup_old_checkpoints(ckpt_dir: str, keep_last_n: int = 2):
    pattern = os.path.join(ckpt_dir, "prior_step*_nll*.pt")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if len(files) > keep_last_n:
        for f in files[:-keep_last_n]:
            try:
                os.remove(f)
            except OSError:
                pass


# ---------------------- main ---------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_cfg", type=str, required=True, help="Path to prior/configs/prior_config.yaml")
    ap.add_argument("--vq_yaml", type=str, default="", help="Path to VQ-VAE yaml (to read codebook_size)")
    ap.add_argument("--resume", type=str, default="", help="Optional: path to prior checkpoint to resume from")
    args = ap.parse_args()

    rank, world = setup_ddp()

    # device / local rank
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    cfg = load_yaml(args.prior_cfg)
    K, PAD, BOS, EOS, V = compute_vocab_and_specials(args.vq_yaml if args.vq_yaml else None, cfg)

    seed = int(cfg["runtime"].get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # data config
    data_cfg = cfg["data"]
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg["num_workers"])
    max_len = int(data_cfg["max_len"])
    pin_memory = True
    drop_last = False
    boundaries = data_cfg.get("bucket_boundaries", []) or []

    # choose sampler mode
    use_bucket = len(boundaries) > 0

    if use_bucket:
        train_loader, train_sampler, train_ds = build_loader_bucketed(
            manifest_path=data_cfg["train_manifest"],
            pad_id=PAD, bos_id=BOS, eos_id=EOS,
            max_len=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            seed=seed,
            boundaries=boundaries,
            is_train=True
        )
        val_loader, val_sampler, val_ds = build_loader_bucketed(
            manifest_path=data_cfg["val_manifest"],
            pad_id=PAD, bos_id=BOS, eos_id=EOS,
            max_len=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            seed=seed,
            boundaries=boundaries,
            is_train=False
        )
    else:
        train_loader, train_sampler, train_ds = build_loader_distributed(
            manifest_path=data_cfg["train_manifest"],
            pad_id=PAD, bos_id=BOS, eos_id=EOS,
            max_len=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=pin_memory,
            drop_last=drop_last,
            seed=seed,
            is_train=True
        )
        val_loader, val_sampler, val_ds = build_loader_distributed(
            manifest_path=data_cfg["val_manifest"],
            pad_id=PAD, bos_id=BOS, eos_id=EOS,
            max_len=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory,
            drop_last=drop_last,
            seed=seed,
            is_train=False
        )

    # rough epoch estimation
    steps_per_epoch = (len(train_ds) + (batch_size * max(1, world) - 1)) // (batch_size * max(1, world))
    approx_epochs = (int(cfg["optim"]["max_updates"]) + steps_per_epoch - 1) // max(1, steps_per_epoch)
    if get_rank() == 0:
        mode = "Bucketed" if use_bucket else "Distributed"
        print(f"[info] Sampler mode: {mode}")
        print(f"[info] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
        print(f"[info] Batch size: {batch_size} per GPU | World size: {world}")
        print(f"[info] Steps/epoch ~= {steps_per_epoch} | Planned epochs ~= {approx_epochs}")

    # model
    model = TransformerPriorLM(
        vocab_size=V,
        d_model=int(cfg["model"]["d_model"]),
        n_layers=int(cfg["model"]["n_layers"]),
        n_heads=int(cfg["model"]["n_heads"]),
        ffw_mult=int(cfg["model"]["ffw_mult"]),
        dropout=float(cfg["model"]["dropout"]),
        tie_embeddings=bool(cfg["model"]["tie_embeddings"]),
        layer_norm_eps=float(cfg["model"]["layer_norm_eps"]),
        pad_token_id=PAD
    ).to(device)

    # optimizer / scheduler / amp
    opt = AdamW(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=float(cfg["optim"]["weight_decay"])
    )
    lr_lambda = lr_lambda_builder(
        warmup_updates=int(cfg["optim"]["warmup_updates"]),
        max_updates=int(cfg["optim"]["max_updates"])
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    scaler = GradScaler(enabled=bool(cfg["runtime"]["amp"]))
    grad_clip = float(cfg["optim"]["grad_clip_norm"])
    log_interval = int(cfg["runtime"]["log_interval"])
    save_interval = int(cfg["runtime"]["save_interval_updates"])  # reserved if you later want to use it
    eval_interval_default = int(cfg["runtime"].get("eval_interval_updates", 10000))
    ckpt_dir = str(cfg["runtime"]["ckpt_dir"])
    amp_flag = bool(cfg["runtime"]["amp"])
    grad_accum = int(cfg["runtime"].get("grad_accum_steps", 1))

    # eval schedule controls
    first_eval_at = int(cfg["runtime"].get("first_eval_at", 4000))
    early_eval_interval = int(cfg["runtime"].get("early_eval_interval_updates", 5000))
    late_eval_interval = eval_interval_default

    def should_eval(step: int) -> bool:
        if step < first_eval_at:
            return False
        interval = early_eval_interval if step < (first_eval_at * 2) else late_eval_interval
        return (step % interval) == 0 or (step == max_updates)

    # optional periodic saving (disabled by default)
    periodic_every = int(cfg["runtime"].get("periodic_save_every", 0))  # 0 = disabled
    keep_last_periodic = int(cfg["runtime"].get("keep_last_periodic", 2))

    # resume
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        global_step = int(ckpt.get("global_step", 0))
        if get_rank() == 0:
            print(f"[resume] loaded {args.resume} at step {global_step}")

    specials = {"K": K, "PAD": PAD, "BOS": BOS, "EOS": EOS, "V": V}
    best_val_nll = float("inf")

    # training loop
    epoch = 0
    model.train()
    max_updates = int(cfg["optim"]["max_updates"])

    accum_loss = 0.0
    accum_count = 0

    start_time = time.time()
    last_update_ts = start_time
    update_times = deque(maxlen=100)

    def fmt_hms(sec: float) -> str:
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    while global_step < max_updates:
        set_epoch_if_possible(train_sampler, epoch)

        for step_in_epoch, batch in enumerate(train_loader):
            if global_step >= max_updates:
                break

            inp = batch["inp"].to(device, non_blocking=True)
            tgt = batch["tgt"].to(device, non_blocking=True)
            attn = batch["attn_mask"].to(device, non_blocking=True)

            with autocast(enabled=amp_flag):
                logits = model(inp, attn_mask=attn)
                loss = model.loss(logits, tgt, ignore_index=PAD) / max(1, grad_accum)

            scaler.scale(loss).backward()
            accum_loss += float(loss.item())
            accum_count += 1

            if ((step_in_epoch + 1) % grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

                now = time.time()
                update_times.append(now - last_update_ts)
                last_update_ts = now

                # logging (rank0)
                if (global_step % log_interval == 0) and (get_rank() == 0):
                    avg_nll = accum_loss / max(1, accum_count)
                    ppl = math.exp(min(20.0, avg_nll))
                    lr_cur = sched.get_last_lr()[0]
                    elapsed = now - start_time
                    avg_update_sec = (sum(update_times) / len(update_times)) if update_times else 0.0
                    remaining_updates = max(0, max_updates - global_step)
                    eta_sec = remaining_updates * avg_update_sec
                    print(
                        f"step {global_step} | nll {avg_nll:.4f} | ppl {ppl:.2f} | lr {lr_cur:.6f} | "
                        f"elapsed {fmt_hms(elapsed)} | eta {fmt_hms(eta_sec)}"
                    )
                    accum_loss = 0.0
                    accum_count = 0

                # eval (rank0 save best/periodic)
                if should_eval(global_step):
                    val_nll = evaluate(model, val_loader, val_sampler, PAD, device, amp=amp_flag)
                    if get_rank() == 0:
                        improved = val_nll < best_val_nll
                        if improved:
                            best_val_nll = val_nll
                            best_path = save_ckpt(ckpt_dir, "prior_best.pt", model, cfg, specials, global_step)
                            print(f"[eval] step {global_step} | val_nll {val_nll:.4f} | best {best_val_nll:.4f} | saved {best_path}")
                        else:
                            print(f"[eval] step {global_step} | val_nll {val_nll:.4f} | best {best_val_nll:.4f}")

                        if periodic_every > 0 and (global_step % periodic_every == 0):
                            p_path = save_ckpt(
                                ckpt_dir, f"prior_step{global_step}_nll{val_nll:.4f}.pt",
                                model, cfg, specials, global_step
                            )
                            cleanup_old_checkpoints(ckpt_dir, keep_last_n=keep_last_periodic)
                            print(f"[periodic] saved {p_path} (keep last {keep_last_periodic})")

        epoch += 1

    # final save (rank0)
    if get_rank() == 0:
        last_path = save_ckpt(ckpt_dir, "prior_last.pt", model, cfg, specials, global_step)
        print(f"[final] saved {last_path} at step {global_step}")

    barrier()


if __name__ == "__main__":
    main()