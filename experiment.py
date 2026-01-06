# -*- coding: utf-8 -*-
import os
import yaml
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Any, Dict, Optional, Tuple, List

from models.vq_vae import VQVAE
from dataset import CurveDataset, pad_collate


def interpolate_schedule(schedules: Dict[str, List[List[float]]], epoch: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not schedules:
        return out
    for key, pairs in schedules.items():
        if not pairs:
            continue
        val = float(pairs[0][1])
        if epoch <= pairs[0][0]:
            out[key] = val
            continue
        for i in range(1, len(pairs)):
            s0, v0 = pairs[i - 1]
            s1, v1 = pairs[i]
            if s0 <= epoch < s1:
                a = (epoch - s0) / max(1e-8, (s1 - s0))
                val = float(v0 + a * (v1 - v0))
                break
            val = float(v1)
        out[key] = val
    return out


def _resolve_path(base_dir: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(base_dir, p)


def _normalize_path(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    if isinstance(p, str) and p.strip() == "":
        return None
    return p


class VQVAEExperiment(pl.LightningModule):
    def __init__(self, model_params: dict, exp_params: dict, data_params: dict):
        super().__init__()
        self.LR = float(exp_params.get("LR", 1e-3))
        self.weight_decay = float(exp_params.get("weight_decay", 0.0))
        self.manual_seed = int(exp_params.get("manual_seed", 42))
        self.save_hyperparameters({
            "LR": self.LR,
            "weight_decay": self.weight_decay,
            "manual_seed": self.manual_seed,
            "model_name": model_params.get("name", "VQVAE"),
        })

        # Model
        self.model = VQVAE(**model_params)
        self.exp_params = exp_params
        self.data_params = data_params

        # Optional warm-start ckpt (stage-2 using stage-1 weights)
        self._warm_start_ckpt: Optional[str] = _normalize_path(exp_params.get("warm_start_ckpt", None))

        # Optional codebook init path (kmeans centroids)
        # Priority: exp_params.init_codebook_path > model_params.codebook_init_path
        self._init_codebook_path: Optional[str] = _normalize_path(
            exp_params.get("init_codebook_path", None)
        )
        if self._init_codebook_path is None:
            self._init_codebook_path = _normalize_path(model_params.get("codebook_init_path", None))

        # Schedules
        self.schedules: Dict[str, List[List[float]]] = exp_params.get("schedules", {}) or {}

        # Current weights
        self.current_weights: Dict[str, float] = {
            "ss_weight": float(exp_params.get("ss_weight", 1.0)),
            "bond_length_weight": float(exp_params.get("bond_length_weight", 0.0)),
            "bond_angle_weight": float(exp_params.get("bond_angle_weight", 0.0)),
            "xyz_tv_lambda": float(exp_params.get("xyz_tv_lambda", 0.0)),
            "dir_weight": float(exp_params.get("dir_weight", 0.0)),
            "dih_weight": float(exp_params.get("dih_weight", 0.0)),
            "rmsd_weight": float(exp_params.get("rmsd_weight", 1.0)),
            "label_smoothing": float(model_params.get("label_smoothing", 0.0)),
            "usage_entropy_lambda": float(model_params.get("usage_entropy_lambda", 0.0)),
            "beta": float(model_params.get("beta", 0.25)),
            # Optional geometry regularizers
            "pdm_weight": float(exp_params.get("pdm_weight", 0.0)),
            "win_kabsch_weight": float(exp_params.get("win_kabsch_weight", 0.0)),
            "kappa_weight": float(exp_params.get("kappa_weight", 0.0)),
            "tau_weight": float(exp_params.get("tau_weight", 0.0)),
            "lr_pdm_weight": float(exp_params.get("lr_pdm_weight", 0.0)),
            "pdm_window": float(exp_params.get("pdm_window", 8)),
            "win_kabsch_size": float(exp_params.get("win_kabsch_size", 16)),
            "win_kabsch_stride": float(exp_params.get("win_kabsch_stride", 8)),
            "lr_min_sep": float(exp_params.get("lr_min_sep", 24)),
            "lr_stride": float(exp_params.get("lr_stride", 8)),
            "lr_max_offsets": float(exp_params.get("lr_max_offsets", 8)),
        }

        self.example_input_array = (torch.zeros(1, 64, 6), torch.ones(1, 64, dtype=torch.bool))

        torch.manual_seed(self.manual_seed)
        self.train_dataset = None
        self.val_dataset = None

        self._ep_sum = {"loss": 0.0, "xyz": 0.0, "ss_loss": 0.0, "vq": 0.0, "rmsd_aln": 0.0, "rmsd_raw": 0.0}
        self._ep_n = 0

    # -------------------------
    # Data
    # -------------------------
    def setup(self, stage: Optional[str] = None):
        npy_dir = self.data_params["npy_dir"]
        train_list = _resolve_path(npy_dir, self.data_params["train_list"])
        val_list = _resolve_path(npy_dir, self.data_params["val_list"])
        self.train_dataset = CurveDataset(npy_dir=npy_dir, list_path=train_list, train=True)
        self.val_dataset = CurveDataset(npy_dir=npy_dir, list_path=val_list, train=False)
        if self.global_rank == 0:
            print(f"[Data] Train files: {len(self.train_dataset)} | Val files: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.data_params.get("train_batch_size", 256)),
            shuffle=True,
            num_workers=int(self.data_params.get("num_workers", 8)),
            pin_memory=bool(self.data_params.get("pin_memory", True)),
            collate_fn=pad_collate,
            drop_last=True,
            persistent_workers=(int(self.data_params.get("num_workers", 8)) > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.data_params.get("val_batch_size", 256)),
            shuffle=False,
            num_workers=int(self.data_params.get("num_workers", 8)),
            pin_memory=bool(self.data_params.get("pin_memory", True)),
            collate_fn=pad_collate,
            drop_last=False,
            persistent_workers=(int(self.data_params.get("num_workers", 8)) > 0),
        )

    def on_validation_epoch_start(self):
        if hasattr(self.model, "quantizer") and hasattr(self.model.quantizer, "reset_epoch_stats"):
            self.model.quantizer.reset_epoch_stats()

    def on_validation_epoch_end(self):
        if hasattr(self.model, "quantizer") and hasattr(self.model.quantizer, "get_epoch_stats"):
            stats = self.model.quantizer.get_epoch_stats()
            if self.global_rank == 0:
                print(f"[Val Stats] PPL: {stats.get('perplexity', 0):.2f}, "
                      f"Dead Ratio: {stats.get('dead_ratio', 0):.3f}")

    # -------------------------
    # Optim
    # -------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.LR, weight_decay=self.weight_decay)

        # If LR is scheduled manually by our epoch schedule, do not register a scheduler here.
        if self.schedules and "LR" in self.schedules:
            return [optimizer]

        sched_name = str(self.exp_params.get("lr_scheduler", "cosine")).lower()
        if sched_name == "none":
            return optimizer

        if sched_name == "onecycle":
            steps_per_epoch = max(1, len(self.train_dataloader()))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.LR,
                epochs=int(self.trainer.max_epochs),
                steps_per_epoch=steps_per_epoch,
                pct_start=float(self.exp_params.get("onecycle_pct_start", 0.15)),
                anneal_strategy="cos",
                div_factor=float(self.exp_params.get("onecycle_div_factor", 25.0)),
                final_div_factor=float(self.exp_params.get("onecycle_final_div", 1500.0)),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(self.trainer.max_epochs), eta_min=self.LR * 1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # -------------------------
    # Warm-start helpers
    # -------------------------
    @staticmethod
    def _strip_model_prefix(state: Dict[str, torch.Tensor], prefix: str = "model.") -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
            else:
                out[k] = v
        return out

    @staticmethod
    def _filter_state_dict_for_warmstart(
        candidate: Dict[str, torch.Tensor],
        model_state: Dict[str, torch.Tensor],
        drop_prefixes: Tuple[str, ...] = ("quantizer.",),
        require_shape_match: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
        kept: Dict[str, torch.Tensor] = {}
        skipped_prefix: List[str] = []
        skipped_shape: List[str] = []

        for k, v in candidate.items():
            if any(k.startswith(dp) for dp in drop_prefixes):
                skipped_prefix.append(k)
                continue
            if k not in model_state:
                continue
            if require_shape_match and tuple(v.shape) != tuple(model_state[k].shape):
                skipped_shape.append(k)
                continue
            kept[k] = v
        return kept, skipped_prefix, skipped_shape

    def _maybe_init_codebook(self):
        # Only meaningful when VQ is enabled and a codebook path is provided.
        if not getattr(self.model, "use_vq", False):
            return
        if self._init_codebook_path is None:
            return
        if not os.path.isfile(self._init_codebook_path):
            if self.global_rank == 0:
                print(f"[CodebookInit] Path not found: {self._init_codebook_path}")
            return

        try:
            import numpy as np
            C = np.load(self._init_codebook_path).astype("float32")
            C = torch.from_numpy(C)
            self.model.init_codebook_from_centroids(C)
            if self.global_rank == 0:
                print(f"[CodebookInit] Loaded centroids from: {self._init_codebook_path} shape={tuple(C.shape)}")
        except Exception as e:
            if self.global_rank == 0:
                print(f"[CodebookInit] Failed to init codebook: {e}")

    # -------------------------
    # Lifecycle
    # -------------------------
    def on_fit_start(self):
        # If trainer.fit(..., ckpt_path=...) is used, this is a true resume.
        # In that case, do NOT warm-start or re-init codebook.
        resume_ckpt = getattr(self.trainer, "ckpt_path", None)
        if isinstance(resume_ckpt, str) and resume_ckpt.strip() == "":
            resume_ckpt = None
        if resume_ckpt is not None:
            if self.global_rank == 0:
                lr_now = None
                try:
                    if self.trainer.optimizers:
                        lr_now = self.trainer.optimizers[0].param_groups[0]["lr"]
                except Exception:
                    pass
                print(f"[Resume] ckpt_path detected, skip warm-start/codebook-init. resume_epoch={int(self.current_epoch)} lr={lr_now}")
            return

        # Warm-start: load weights from a previous checkpoint, but DO NOT load quantizer.* states.
        if self._warm_start_ckpt and os.path.isfile(self._warm_start_ckpt):
            if self.global_rank == 0:
                print(f"[WarmStart] Loading model weights from: {self._warm_start_ckpt}")

            try:
                ckpt = torch.load(self._warm_start_ckpt, map_location="cpu")
                state = ckpt.get("state_dict", ckpt)
                state = self._strip_model_prefix(state, prefix="model.")

                model_state = self.model.state_dict()
                filtered, skipped_prefix, skipped_shape = self._filter_state_dict_for_warmstart(
                    candidate=state,
                    model_state=model_state,
                    drop_prefixes=("quantizer.",),   # critical: prevent codebook overwrite
                    require_shape_match=True,
                )

                missing, unexpected = self.model.load_state_dict(filtered, strict=False)

                if self.global_rank == 0:
                    print(f"[WarmStart] loaded kept={len(filtered)} missing={len(missing)} unexpected={len(unexpected)} "
                          f"skipped_prefix={len(skipped_prefix)} skipped_shape={len(skipped_shape)}")
                    if len(skipped_shape) > 0:
                        print(f"[WarmStart] skipped_shape_examples: {skipped_shape[:10]}")
            except Exception as e:
                if self.global_rank == 0:
                    print(f"[WarmStart] Failed to load: {e}")

        # Always apply codebook init AFTER warm-start (so it cannot be overwritten).
        self._maybe_init_codebook()

    def on_train_epoch_start(self):
        epoch = int(self.current_epoch)
        new_vals = interpolate_schedule(self.schedules, epoch) if self.schedules else {}

        for k, v in (new_vals or {}).items():
            if k in self.current_weights:
                self.current_weights[k] = float(v)

        for _k in ["pdm_window", "win_kabsch_size", "win_kabsch_stride", "lr_min_sep", "lr_stride", "lr_max_offsets"]:
            self.current_weights[_k] = int(round(float(self.current_weights.get(_k, 0))))

        self.model.label_smoothing = self.current_weights["label_smoothing"]
        self.model.usage_entropy_lambda = self.current_weights["usage_entropy_lambda"]
        if hasattr(self.model, "quantizer") and hasattr(self.model.quantizer, "reset_epoch_stats"):
            self.model.quantizer.reset_epoch_stats()

        if self.global_rank == 0:
            kw = self.current_weights
            brief_keys = ["beta", "ss_weight", "rmsd_weight"]
            brief = {k: round(float(kw[k]), 6) for k in brief_keys if k in kw}
            print(f"[Schedule] Epoch {epoch}: {brief}")

        self._ep_sum = {"loss": 0.0, "xyz": 0.0, "ss_loss": 0.0, "vq": 0.0, "rmsd_aln": 0.0, "rmsd_raw": 0.0}
        self._ep_n = 0

        if "beta" in self.current_weights:
            self.model.beta = float(self.current_weights["beta"])
            if hasattr(self.model, "quantizer"):
                self.model.quantizer.beta = float(self.model.beta)

        if "LR" in new_vals:
            new_lr = float(new_vals["LR"])
            if self.trainer and self.trainer.optimizers:
                for param_group in self.trainer.optimizers[0].param_groups:
                    param_group["lr"] = new_lr

    # -------------------------
    # Core
    # -------------------------
    def forward(self, x, mask):
        return self.model(x, mask)

    def _compute_and_log(self, batch, stage: str):
        x, mask = batch
        recons, target, vq_pack, mask_out = self.forward(x, mask)

        loss_dict = self.model.loss_function(
            recons, target, vq_pack, mask_out,
            ss_weight=self.current_weights["ss_weight"],
            bond_length_weight=self.current_weights["bond_length_weight"],
            bond_angle_weight=self.current_weights["bond_angle_weight"],
            xyz_tv_lambda=self.current_weights["xyz_tv_lambda"],
            dir_weight=self.current_weights["dir_weight"],
            dih_weight=self.current_weights["dih_weight"],
            rmsd_weight=self.current_weights["rmsd_weight"],
            pdm_weight=self.current_weights["pdm_weight"],
            win_kabsch_weight=self.current_weights["win_kabsch_weight"],
            kappa_weight=self.current_weights["kappa_weight"],
            tau_weight=self.current_weights["tau_weight"],
            lr_pdm_weight=self.current_weights["lr_pdm_weight"],
            pdm_window=self.current_weights["pdm_window"],
            win_kabsch_size=self.current_weights["win_kabsch_size"],
            win_kabsch_stride=self.current_weights["win_kabsch_stride"],
            lr_min_sep=self.current_weights["lr_min_sep"],
            lr_stride=self.current_weights["lr_stride"],
            lr_max_offsets=self.current_weights["lr_max_offsets"],
        )

        # Optional quick latent diagnostics (cheap)
        try:
            zq_raw, ze_raw, _, _, _ = vq_pack
            if mask_out is not None and ze_raw.size(1) == mask_out.size(1):
                valid = mask_out.reshape(-1)
                ze = ze_raw.reshape(-1, ze_raw.size(-1))[valid]
                zq = zq_raw.reshape(-1, zq_raw.size(-1))[valid]
            else:
                ze = ze_raw.reshape(-1, ze_raw.size(-1))
                zq = zq_raw.reshape(-1, zq_raw.size(-1))
            if ze.numel() > 0 and zq.numel() > 0:
                ze_norm = ze.norm(dim=-1).mean()
                zq_norm = zq.norm(dim=-1).mean()
                cos_ze_zq = F.cosine_similarity(ze, zq, dim=-1).mean()
            else:
                device = loss_dict["loss"].device
                ze_norm = torch.tensor(0.0, device=device)
                zq_norm = torch.tensor(0.0, device=device)
                cos_ze_zq = torch.tensor(0.0, device=device)
        except Exception:
            device = loss_dict["loss"].device
            ze_norm = torch.tensor(0.0, device=device)
            zq_norm = torch.tensor(0.0, device=device)
            cos_ze_zq = torch.tensor(0.0, device=device)

        on_step = (stage == "train")
        on_epoch = True
        pb = False
        sync = True

        self.log(f"{stage}/loss_total", loss_dict["loss"].float(), on_step=on_step, on_epoch=on_epoch, prog_bar=pb, sync_dist=sync)
        self.log(f"{stage}/loss_xyz", loss_dict["Reconstruction_Loss_XYZ"].float(), on_step=on_step, on_epoch=on_epoch, prog_bar=pb, sync_dist=sync)

        if "XYZ_MSE_Raw" in loss_dict:
            self.log(f"{stage}/xyz_mse_raw", loss_dict["XYZ_MSE_Raw"].float(), on_step=on_step, on_epoch=True, prog_bar=False, sync_dist=sync)
        if "XYZ_MSE_Aligned" in loss_dict:
            self.log(f"{stage}/xyz_mse_aln", loss_dict["XYZ_MSE_Aligned"].float(), on_step=on_step, on_epoch=True, prog_bar=False, sync_dist=sync)

        self.log(f"{stage}/vq_loss", loss_dict["VQ_Loss"].float(), on_step=on_step, on_epoch=on_epoch, prog_bar=pb, sync_dist=sync)

        if "VQ_Perplexity" in loss_dict:
            self.log(f"{stage}/perplexity", loss_dict["VQ_Perplexity"].float(), on_step=on_step, on_epoch=on_epoch, prog_bar=pb, sync_dist=sync)
        if "VQ_DeadRatio" in loss_dict:
            self.log(f"{stage}/dead_ratio", loss_dict["VQ_DeadRatio"].float(), on_step=on_step, on_epoch=on_epoch, prog_bar=pb, sync_dist=sync)
        if "SS_Accuracy" in loss_dict:
            self.log(f"{stage}/ss_acc", loss_dict["SS_Accuracy"].float(), on_step=on_step, on_epoch=on_epoch, prog_bar=pb, sync_dist=sync)

        if stage == "train" and self.trainer and self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log(f"{stage}/lr", torch.tensor(lr, device=loss_dict["loss"].device).float(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=False)

        for k in [
            "Reconstruction_Loss_SS", "Geom_Loss",
            "Geom_BondLength_Loss", "Geom_BondAngle_Loss", "Geom_Direction_Loss", "Geom_Dihedral_Loss",
            "SS_TV", "Usage_Reg", "XYZ_TV2", "RMSD_Raw", "RMSD_Aligned",
            "Geom_LocalPDM", "Geom_WinKabsch", "Geom_LongRangePDM",
            "Frenet_Kappa", "Frenet_Tau"
        ]:
            if k in loss_dict:
                on_step_k = (stage == "train") and (k in ["RMSD_Raw", "RMSD_Aligned"])
                self.log(f"{stage}/{k}", loss_dict[k].float(), on_step=on_step_k, on_epoch=True, prog_bar=False, sync_dist=sync)

        if stage == "train":
            self._ep_sum["loss"] += float(loss_dict["loss"].detach().item())
            self._ep_sum["xyz"] += float(loss_dict["Reconstruction_Loss_XYZ"].detach().item())
            self._ep_sum["rmsd_aln"] += float(loss_dict["RMSD_Aligned"].detach().item())
            self._ep_sum["rmsd_raw"] += float(loss_dict["RMSD_Raw"].detach().item())
            self._ep_sum["ss_loss"] += float(loss_dict["Reconstruction_Loss_SS"].detach().item())
            self._ep_sum["vq"] += float(loss_dict.get("VQ_Loss", torch.tensor(0.0, device=loss_dict["loss"].device)).detach().item())
            self._ep_n += 1

        return loss_dict["loss"]

    # -------------------------
    # Lightning hooks
    # -------------------------
    def training_step(self, batch, batch_idx):
        loss = self._compute_and_log(batch, stage="train")
        N = int(self.exp_params.get("print_every", 0))
        if (N > 0) and (batch_idx % N == 0) and (self.global_rank == 0):
            md = self.trainer.callback_metrics
            msg = (
                f"step={batch_idx:05d} | "
                f"loss={float(md.get('train/loss_total', 0.0)):.3f} | "
                f"xyz={float(md.get('train/loss_xyz', 0.0)):.3f} | "
                f"xyz_raw={float(md.get('train/xyz_mse_raw', 0.0)):.3f} | "
                f"xyz_aln={float(md.get('train/xyz_mse_aln', 0.0)):.3f} | "
                f"vq={float(md.get('train/vq_loss', 0.0)):.3f} | "
                f"ppl={float(md.get('train/perplexity', 0.0)):.3f} | "
                f"dead={float(md.get('train/dead_ratio', 0.0)):.3f} | "
                f"ss_acc={float(md.get('train/ss_acc', 0.0)):.3f} | "
                f"ss_loss={float(md.get('train/Reconstruction_Loss_SS', 0.0)):.3f} | "
                f"lr={float(md.get('train/lr', 0.0)):.6f}"
            )
            try:
                from pytorch_lightning.utilities import rank_zero_info
                rank_zero_info(msg)
            except Exception:
                print(msg, flush=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._compute_and_log(batch, stage="val")

    def on_train_epoch_end(self):
        if self._ep_n > 0 and self.global_rank == 0:
            mean_loss = self._ep_sum["loss"] / self._ep_n
            mean_xyz = self._ep_sum["xyz"] / self._ep_n
            mean_ss_loss = self._ep_sum["ss_loss"] / self._ep_n
            mean_vq = self._ep_sum["vq"] / self._ep_n
            mean_ra = self._ep_sum["rmsd_aln"] / self._ep_n
            mean_rr = self._ep_sum["rmsd_raw"] / self._ep_n

            lr = 0.0
            try:
                if self.trainer and self.trainer.optimizers:
                    lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])
            except Exception:
                pass

            print(
                f"[Epoch {int(self.current_epoch)}] "
                f"loss={mean_loss:.4f} xyz={mean_xyz:.4f} ss_loss={mean_ss_loss:.4f} "
                f"rmsd_aln={mean_ra:.4f}Å rmsd_raw={mean_rr:.4f}Å "
                f"vq={mean_vq:.4f} lr={lr:.6f}"
            )


def build_experiment_from_yaml(yaml_path: str) -> Tuple[pl.LightningModule, dict]:
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

        def _expand_env(obj):
            if isinstance(obj, str):
                return os.path.expandvars(obj)
            if isinstance(obj, dict):
                return {k: _expand_env(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_expand_env(v) for v in obj]
            return obj

        config = _expand_env(config)

    model_params = config["model_params"]
    exp_params = config["exp_params"]
    data_params = config["data_params"]

    experiment = VQVAEExperiment(model_params, exp_params, data_params)
    return experiment, config


if __name__ == "__main__":
    import argparse
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()

    exp, cfg = build_experiment_from_yaml(args.config)

    trainer_params = cfg.get("trainer_params", {})
    logging_params = cfg.get("logging_params", {})
    save_dir = logging_params.get("save_dir", "./logs/")
    name = logging_params.get("name", "exp")
    out_dir = os.path.join(save_dir, name)

    os.makedirs(out_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir,
        filename="epoch{epoch:03d}-step{step}",
        save_last=True,
        save_top_k=0,
        every_n_epochs=1,
    )

    trainer = Trainer(
        accelerator=trainer_params.get("accelerator", "gpu"),
        devices=trainer_params.get("devices", args.devices),
        strategy=trainer_params.get("strategy", "ddp"),
        precision=trainer_params.get("precision", 32),
        max_epochs=trainer_params.get("max_epochs", 40),
        log_every_n_steps=trainer_params.get("log_every_n_steps", 20),
        enable_progress_bar=trainer_params.get("enable_progress_bar", True),
        gradient_clip_val=trainer_params.get("gradient_clip_val", 1.0),
        num_sanity_val_steps=trainer_params.get("num_sanity_val_steps", 0),
        benchmark=trainer_params.get("benchmark", True),
        limit_val_batches=trainer_params.get("limit_val_batches", 1.0),
        deterministic=trainer_params.get("deterministic", False),
        callbacks=[ckpt_cb],
        default_root_dir=out_dir,
        logger=False,
    )

    trainer.fit(exp, ckpt_path=args.ckpt_path)
