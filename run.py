# -*- coding: utf-8 -*-
"""
run.py
Generic two-stage training entry for VQ-VAE (Stage1: AE pretrain, Stage2: VQ fine-tune).

Usage examples:

  # Stage 1: AE pretrain (no quantization)
  python run.py --config configs/stage1_ae.yaml

  # Stage 2: VQ fine-tune with warm start and codebook init
  python run.py \
      --config configs/stage2_vq.yaml \
      --warm_start_ckpt /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/ae_new_checkpoints_test/epochepoch=epoch=079.ckpt \
      --init_codebook scripts/kmeans_centroids_512x128.npy

  # Resume from a previous full checkpoint (optimizer, schedulers, etc.)
  # When --resume_ckpt is provided, warm-start and codebook init are skipped.
  python run.py \
      --config configs/stage2_vq.yaml \
      --resume_ckpt /public/home/zhangyangroup/chengshiz/keyuan.zhou/PyTorch-VAE/checkpoints/vq_s_gradient_ckpt_test11_15/epochepoch=549.ckpt
"""

import os
import yaml
import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything

from experiment import VQVAEExperiment


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def maybe_warm_start(model: torch.nn.Module, ckpt_path: str):
    """
    Optionally load model weights from a previous checkpoint (only model.* keys).
    Use this for Stage-2 to import Stage-1 AE weights. This is NOT resume.
    """
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"[Warm-start] skipped (no valid ckpt at {ckpt_path})")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    stripped = {}
    for k, v in state.items():
        if k.startswith("model."):
            stripped[k[len("model."):]] = v
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    print(f"[Warm-start] Loaded weights from {ckpt_path}")
    print(f"[Warm-start] missing={len(missing)}, unexpected={len(unexpected)}")


def maybe_init_codebook(model: torch.nn.Module, path: str):
    """
    Optionally initialize quantizer codebook from KMeans centroids (.npy).
    This should be used before training starts and is skipped on resume.
    """
    if not path or not os.path.isfile(path):
        print(f"[Codebook init] skipped (invalid path: {path})")
        return
    C = np.load(path).astype(np.float32)
    C = torch.from_numpy(C)
    device = next(model.parameters()).device
    # Defer actual copy to the model's method (it sets embedding and EMA buffers).
    if not hasattr(model, "init_codebook_from_centroids"):
        raise AttributeError("Model does not implement init_codebook_from_centroids(tensor).")
    model.init_codebook_from_centroids(C.to(device))
    print(f"[Codebook init] Loaded centroids {tuple(C.shape)} from {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE (two-stage compatible).")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--warm_start_ckpt", type=str, default="",
                        help="Optional Stage-1 AE checkpoint for warm start. Ignored if --resume_ckpt is set.")
    parser.add_argument("--init_codebook", type=str, default="",
                        help="Optional .npy for initializing codebook centroids. Ignored if --resume_ckpt is set.")
    parser.add_argument("--resume_ckpt", type=str, default="",
                        help="Resume training from a full checkpoint (model+optimizer+scheduler+epoch).")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load YAML config
    # -----------------------------------------------------------------------
    cfg = load_yaml(args.config)
    model_params = cfg["model_params"]
    exp_params = cfg["exp_params"]
    data_params = cfg["data_params"]
    trainer_params = cfg.get("trainer_params", {})
    logging_params = cfg.get("logging_params", {})

    # Seed
    seed_val = exp_params.get("manual_seed", 42)
    seed_everything(seed_val, workers=True)
    print(f"[Seed] manual_seed={seed_val}")

    # -----------------------------------------------------------------------
    # Build experiment and get model for optional initializations
    # -----------------------------------------------------------------------
    experiment = VQVAEExperiment(model_params, exp_params, data_params)
    model = experiment.model

    # Decide resume or fresh run
    is_resume = bool(args.resume_ckpt)
    if is_resume:
        if not os.path.isfile(args.resume_ckpt):
            raise FileNotFoundError(f"[Resume] ckpt not found: {args.resume_ckpt}")
        print(f"[Resume] Will resume full state from: {args.resume_ckpt}")

        # Strongly disable warm-start from YAML when resuming
        # so experiment.on_fit_start will not try to warm-start.
        experiment.exp_params["warm_start_ckpt"] = ""
    else:
        # Resolve warm_start_ckpt: CLI has priority, otherwise YAML exp_params
        warm_start_path = args.warm_start_ckpt or exp_params.get("warm_start_ckpt", "")
        if warm_start_path:
            try:
                maybe_warm_start(model, warm_start_path)
            except Exception as e:
                print(f"[Warm-start] failed: {e}")
        else:
            print("[Warm-start] skipped (no warm_start_ckpt provided).")

        # Codebook init: CLI has priority, otherwise YAML model_params.codebook_init_path
        codebook_path = args.init_codebook or model_params.get("codebook_init_path", "")
        if model_params.get("use_vq", True) and codebook_path:
            try:
                maybe_init_codebook(model, codebook_path)
            except Exception as e:
                print(f"[Codebook init] failed: {e}")
        else:
            print("[Codebook init] skipped (use_vq=False or no path provided).")

    # -----------------------------------------------------------------------
    # Logging and checkpoint callbacks
    # -----------------------------------------------------------------------
    log_dir = Path(logging_params.get("save_dir", "./logs"))
    logger_name = logging_params.get("name", model_params.get("name", "VQVAE"))
    if is_resume:
        # Mark resume runs in logger name for clarity
        logger_name = f"{logger_name}-resume"
    logger = TensorBoardLogger(save_dir=str(log_dir), name=logger_name)

    # Checkpoint directory (default matches user's prior folder)
    ckpt_dir = Path(exp_params.get("checkpoint_dir", "./ae_new_checkpoints_test"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save every N epochs and keep all
    save_every = int(exp_params.get("save_every_epochs", 10))
    filename_pat = exp_params.get("checkpoint_name_pattern", "epochepoch={epoch:03d}")
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=filename_pat,
        every_n_epochs=save_every,
        save_last=True,
        save_top_k=-1,
        verbose=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------
    clip_val = trainer_params.pop("gradient_clip_val", 5.0)
    trainer = Trainer(
        logger=logger,
        callbacks=[ckpt_cb, lr_monitor],
        gradient_clip_val=clip_val,
        **trainer_params,
    )

    # -----------------------------------------------------------------------
    # Run training
    # -----------------------------------------------------------------------
    model_name = model_params.get("name", "VQVAE")
    print("======= Training {} =======".format(model_name))
    print("use_vq =", model_params.get("use_vq", True))
    if not is_resume and (args.warm_start_ckpt or exp_params.get("warm_start_ckpt", "")):
        print("[Warm-start from]", args.warm_start_ckpt or exp_params.get("warm_start_ckpt", ""))
    if not is_resume and (args.init_codebook or model_params.get("codebook_init_path", "")):
        print("[Codebook init from]", args.init_codebook or model_params.get("codebook_init_path", ""))

    # cuDNN setup for speed
    cudnn.benchmark = bool(cfg.get("trainer_params", {}).get("benchmark", True))

    start_time = time.time()

    # Important: pass resume checkpoint to fit via ckpt_path.
    if is_resume:
        trainer.fit(experiment, ckpt_path=args.resume_ckpt)
    else:
        trainer.fit(experiment)

    end_time = time.time()
    print(f"[Done] Training completed in {(end_time - start_time)/60:.2f} minutes.")
    print(f"[Checkpoint dir] {str(ckpt_dir.resolve())}")
    print(f"[TensorBoard log] {str((log_dir / logger_name).resolve())}")


if __name__ == "__main__":
    main()
