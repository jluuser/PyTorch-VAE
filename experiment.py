# -*- coding: utf-8 -*-

import math
import torch
import numpy as np
from torch import optim
from pathlib import Path
import torch.nn.functional as F
import pytorch_lightning as pl
from random import randint

PROJECT_ROOT = Path(__file__).resolve().parent


class VAEXperiment(pl.LightningModule):
    def __init__(self, vae_model: pl.LightningModule, params: dict) -> None:
        super(VAEXperiment, self).__init__()
        self.model = vae_model
        self.params = params
        self.curr_device: torch.device = torch.device("cpu")
        self.lr = params.get("LR", 1e-3)

    # --- new: ensure geometry losses can run in real scale if supported ---
    def on_fit_start(self) -> None:
        try:
            dm = self.trainer.datamodule
            mean = getattr(dm, "mean_xyz", None)
            std = getattr(dm, "std_xyz", None)
            if (mean is not None) and (std is not None) and hasattr(self.model, "set_data_stats"):
                self.model.set_data_stats(mean, std)
                print("[Info] set_data_stats applied at fit start.")
        except Exception as e:
            print(f"[Warn] set_data_stats skipped: {e}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        return self.model(x, mask=mask, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x, mask = batch
        self.curr_device = x.device
        results = self.forward(x, mask=mask)
        train_loss = self.model.loss_function(
            *results,
            ss_weight=self.params.get("ss_weight", 1.0),
            bond_length_weight=self.params.get("bond_length_weight", 0.0),
            bond_angle_weight=self.params.get("bond_angle_weight", 0.0),
            mask=mask,
        )
        self.log_dict({f"train_{k}": (v.item() if isinstance(v, torch.Tensor) else float(v))
                       for k, v in train_loss.items()},
                      sync_dist=True)
        return train_loss

    def _avg_outputs(self, outputs):
        keys = outputs[0].keys()
        avg = {}
        for k in keys:
            vals = []
            for out in outputs:
                v = out[k]
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v)
                vals.append(v)
            avg[k] = torch.stack(vals).mean().item()
        return avg

    def training_epoch_end(self, outputs: list[dict]) -> None:
        avg = self._avg_outputs(outputs)
        e = self.current_epoch
        print(
            f"[Epoch {e:03d} TRAIN] "
            f"loss={avg['loss']:.4f}, "
            f"xyz={avg['Reconstruction_Loss_XYZ']:.4f}, "
            f"ss={avg['Reconstruction_Loss_SS']:.4f}, "
            f"vq={avg['VQ_Loss']:.4f}, "
            f"geom={avg.get('Geom_Loss', 0.0):.4f}, "
            f"ppl={avg.get('VQ_Perplexity', 0.0):.2f}, "
            f"dead={avg.get('VQ_DeadRatio', 0.0):.3f}"
        )

    def validation_step(self, batch, batch_idx):
        x, mask = batch
        self.curr_device = x.device
        results = self.forward(x, mask=mask)
        val_loss = self.model.loss_function(
            *results,
            ss_weight=self.params.get("ss_weight", 1.0),
            bond_length_weight=self.params.get("bond_length_weight", 0.0),
            bond_angle_weight=self.params.get("bond_angle_weight", 0.0),
            mask=mask,
        )
        self.log_dict({f"val_{k}": (v.item() if isinstance(v, torch.Tensor) else float(v))
                       for k, v in val_loss.items()},
                      sync_dist=True)
        return val_loss

    def validation_epoch_end(self, outputs: list[dict]) -> None:
        avg = self._avg_outputs(outputs)
        e = self.current_epoch
        print(
            f"[Epoch {e:03d}   VAL] "
            f"loss={avg['loss']:.4f}, "
            f"xyz={avg['Reconstruction_Loss_XYZ']:.4f}, "
            f"ss={avg['Reconstruction_Loss_SS']:.4f}, "
            f"vq={avg['VQ_Loss']:.4f}, "
            f"geom={avg.get('Geom_Loss', 0.0):.4f}, "
            f"ppl={avg.get('VQ_Perplexity', 0.0):.2f}, "
            f"dead={avg.get('VQ_DeadRatio', 0.0):.3f}"
        )

    def sample_images(self):
        sample_dir = PROJECT_ROOT / "sample_npy"
        sample_dir.mkdir(exist_ok=True, parents=True)
        try:
            out = self.model.sample(1, self.curr_device)[0]

            mean = np.array(self.trainer.datamodule.mean_xyz, dtype=np.float32).reshape(1, 3)
            std = np.array(self.trainer.datamodule.std_xyz, dtype=np.float32).reshape(1, 3)

            xyz = out[:, :3].detach().cpu().numpy() * std + mean
            ss_logits = out[:, 3:].detach().cpu()
            ss_idx = ss_logits.argmax(dim=-1)
            ss_onehot = F.one_hot(ss_idx, num_classes=3).float().numpy()
            full_curve = np.concatenate([xyz, ss_onehot], axis=-1)

            np.save(sample_dir / f"sample_epoch_{self.current_epoch}_0.npy", full_curve)
        except Exception as e:
            print(f"[Warning] Sample generation failed: {e}")

    def save_reconstructions(self):
        recon_dir = PROJECT_ROOT / "reconstruction_npy"
        recon_dir.mkdir(exist_ok=True, parents=True)
        orig_dir = PROJECT_ROOT / "original_npy"
        orig_dir.mkdir(exist_ok=True, parents=True)
        try:
            val_loader = self.trainer.datamodule.val_dataloader()
            batch = next(iter(val_loader))
            input_tensor, mask = batch
            input_tensor = input_tensor.to(self.curr_device)
            mask = mask.to(self.curr_device)

            results = self.model(input_tensor, mask=mask)
            recons = results[0]
            mean = np.array(self.trainer.datamodule.mean_xyz, dtype=np.float32).reshape(1, 3)
            std = np.array(self.trainer.datamodule.std_xyz, dtype=np.float32).reshape(1, 3)

            recon_curve = recons[0]
            orig_curve = input_tensor[0]

            recon_xyz = recon_curve[:, :3].detach().cpu().numpy() * std + mean
            orig_xyz = orig_curve[:, :3].detach().cpu().numpy() * std + mean

            ss_logits = recon_curve[:, 3:].detach().cpu()
            ss_idx = ss_logits.argmax(dim=-1)
            ss_onehot = F.one_hot(ss_idx, num_classes=3).float().numpy()

            orig_ss = orig_curve[:, 3:].detach().cpu().numpy()

            recon_final = np.concatenate([recon_xyz, ss_onehot], axis=-1)
            orig_final = np.concatenate([orig_xyz, orig_ss], axis=-1)

            np.save(recon_dir / f"recon_epoch_{self.current_epoch}_0.npy", recon_final)
            np.save(orig_dir / f"orig_epoch_{self.current_epoch}_0.npy", orig_final)
        except Exception as e:
            print(f"[Warning] Reconstruction save failed: {e}")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.params.get("weight_decay", 0.0)
        )

        plateau = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
            threshold=1e-4,
            cooldown=0,
            min_lr=0.0
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": plateau,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": True,
                "strict": True
            }
        }
