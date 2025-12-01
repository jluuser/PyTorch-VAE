# -*- coding: utf-8 -*-
import math
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
EPS = 1e-8


# ---------------------------------------------
# Vector Quantizer (EMA), L2 distance
# ---------------------------------------------
class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
        decay: float = 0.98,
        eps: float = 1e-5,
        reinit_dead_codes: bool = True,
        reinit_prob: float = 1.0,
        dead_usage_threshold: int = 0,
        print_init: bool = True,
        diag_qe_cap: float = 10.0,
        diag_qe_bins: int = 64,
    ):
        super().__init__()
        self.K = int(num_embeddings)
        self.D = int(embedding_dim)
        self.beta = float(beta)
        self.decay = float(decay)
        self.eps = float(eps)
        self.use_ema = True

        self.reinit_dead_codes = bool(reinit_dead_codes)
        self.reinit_prob = float(reinit_prob)
        self.dead_usage_threshold = int(dead_usage_threshold)

        emb = torch.randn(self.K, self.D) * (1.0 / math.sqrt(self.D))
        self.register_buffer("embedding", emb)
        self.register_buffer("ema_cluster_size", torch.zeros(self.K))
        self.register_buffer("ema_embedding", torch.zeros(self.K, self.D))

        self.register_buffer("_ep_usage", torch.zeros(self.K))
        self.register_buffer("_ep_top1_sum", torch.zeros(1))
        self.register_buffer("_ep_top2_sum", torch.zeros(1))
        self.register_buffer("_ep_cnt", torch.zeros(1))
        self.register_buffer("_ep_qe_sum", torch.zeros(1))
        self.diag_qe_cap = float(diag_qe_cap)
        self.diag_qe_bins = int(diag_qe_bins)
        self.register_buffer("_ep_qe_hist", torch.zeros(self.diag_qe_bins))

        if print_init:
            print(f"[VQ] EMA (L2): K={self.K}, D={self.D}, beta={self.beta}, decay={self.decay}")

    @torch.no_grad()
    def _ema_update(self, flat_raw: Tensor, indices: Tensor):
        if flat_raw.numel() == 0 or indices.numel() == 0:
            return
        one_hot = F.one_hot(indices, num_classes=self.K).float()
        cluster_size = one_hot.sum(dim=0)
        embed_sum = one_hot.t() @ flat_raw
    
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size * (1 - self.decay))
        self.ema_embedding.mul_(self.decay).add_(embed_sum * (1 - self.decay))
    
        updated = self.ema_embedding / (self.ema_cluster_size.unsqueeze(1) + self.eps)
        self.embedding.copy_(updated)

    @torch.no_grad()
    def _maybe_reinit_dead_codes(self, flat_raw: Tensor, usage: Tensor):
        if not self.reinit_dead_codes or self.reinit_prob <= 0.0:
            return
        dead_mask = usage <= float(self.dead_usage_threshold)
        num_dead = int(dead_mask.sum().item())
        if num_dead <= 0 or flat_raw.numel() == 0:
            return
        if torch.rand(()) > self.reinit_prob:
            return
        rand_idx = torch.randint(0, flat_raw.size(0), (num_dead,), device=flat_raw.device)
        new_vecs = flat_raw[rand_idx]
        self.embedding[dead_mask] = new_vecs
        if hasattr(self, "ema_embedding"):
            self.ema_embedding[dead_mask] = new_vecs.clone()
        if hasattr(self, "ema_cluster_size"):
            self.ema_cluster_size[dead_mask] = 1.0

    @torch.no_grad()
    def reset_epoch_stats(self):
        self._ep_usage.zero_()
        self._ep_top1_sum.zero_()
        self._ep_top2_sum.zero_()
        self._ep_cnt.zero_()
        self._ep_qe_sum.zero_()
        self._ep_qe_hist.zero_()

    @torch.no_grad()
    def _accumulate_epoch_stats(
        self,
        flat_raw: Tensor,
        indices: Tensor,
        z_e_flat: Tensor,
        z_q_flat: Tensor,
        valid_mask: Optional[Tensor] = None,
    ):
        if flat_raw.numel() == 0:
            return
        device = flat_raw.device
        if valid_mask is None:
            valid_mask = torch.ones(flat_raw.size(0), dtype=torch.bool, device=device)
        if not valid_mask.any():
            return

        fn = flat_raw[valid_mask]
        idxv = indices[valid_mask]
        zev = z_e_flat[valid_mask]
        zqv = z_q_flat[valid_mask]

        usage = torch.bincount(idxv, minlength=self.K).float()
        self._ep_usage.add_(usage)

        fn_n = fn / fn.norm(dim=1, keepdim=True).clamp_min(1e-8)
        emb_n = self.embedding / self.embedding.norm(dim=1, keepdim=True).clamp_min(1e-8)
        dots = fn_n @ emb_n.t()
        top2 = torch.topk(dots, k=2, dim=1).values
        s1 = top2[:, 0]
        s2 = top2[:, 1]
        self._ep_top1_sum.add_(s1.sum())
        self._ep_top2_sum.add_(s2.sum())
        self._ep_cnt.add_(torch.tensor(float(fn_n.size(0)), device=device))

        qe = (zqv - zev).pow(2).sum(dim=1)
        self._ep_qe_sum.add_(qe.sum())

        if self.diag_qe_bins > 0:
            cap = self.diag_qe_cap if self.diag_qe_cap > 0 else float(qe.max().item() + 1e-6)
            qec = qe.clamp_min(0.0).clamp_max(cap)
            denom = max(cap, 1e-12)
            bin_idx = torch.clamp((qec / denom * (self.diag_qe_bins - 1)).long(),
                                  min=0, max=self.diag_qe_bins - 1)
            hist = torch.bincount(bin_idx, minlength=self.diag_qe_bins).float()
            self._ep_qe_hist.add_(hist)

    @torch.no_grad()
    def get_epoch_stats(self) -> dict:
        usage = self._ep_usage.detach().cpu()
        cnt = float(self._ep_cnt.item())
        if cnt <= 0:
            return {
                "usage_hist": usage, "margin_mean": 0.0, "qe_mean": 0.0, "qe_p90": 0.0,
                "n_positions": 0, "perplexity": 0.0, "dead_ratio": 0.0
            }
    
        margin_mean = float(((self._ep_top1_sum - self._ep_top2_sum) / cnt).item())
        qe_mean = float((self._ep_qe_sum / cnt).item())
    
        total = float(usage.sum().item())
        if total > 0:
            p = (usage / max(total, 1e-12)).clamp_min(1e-12)
            perplexity = float(torch.exp(-(p * p.log()).sum()).item())
            dead_ratio = float((usage == 0).float().mean().item())
        else:
            perplexity, dead_ratio = 0.0, 0.0
    
        qe_p90 = 0.0
        hist = self._ep_qe_hist.detach().cpu()
        total_hist = float(hist.sum().item())
        if total_hist > 0:
            cdf = torch.cumsum(hist, dim=0) / max(total_hist, 1e-12)
            if (cdf >= 0.9).any():
                idx = int((cdf >= 0.9).nonzero(as_tuple=True)[0][0].item())
            else:
                idx = self.diag_qe_bins - 1
            bin_w = self.diag_qe_cap / max(self.diag_qe_bins, 1)
            qe_p90 = (idx + 0.5) * bin_w
    
        return {
            "usage_hist": usage,
            "margin_mean": margin_mean,
            "qe_mean": qe_mean,
            "qe_p90": float(qe_p90),
            "n_positions": int(cnt),
            "perplexity": perplexity,
            "dead_ratio": dead_ratio,
        }

    @torch.no_grad()
    def get_embedding_snapshot(self) -> Tensor:
        return self.embedding.detach().clone()

    def forward(
        self,
        z_e: Tensor,  # [B,M,D]
        do_ema_update: bool = True,
        allow_reinit: bool = True,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        B, M, D = z_e.shape
        flat = z_e.reshape(-1, D)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * torch.matmul(flat, self.embedding.t())
            + self.embedding.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = torch.argmin(distances, dim=1)
        z_q = F.embedding(indices, self.embedding).view(B, M, D)

        if self.training and do_ema_update:
            if mask is None:
                self._ema_update(flat.detach(), indices.detach())
            else:
                valid = mask.reshape(-1)
                if valid.any():
                    self._ema_update(flat[valid].detach(), indices[valid].detach())

        z_q_st = z_e + (z_q - z_e).detach()

        with torch.no_grad():
            valid_vec = mask.reshape(-1) if mask is not None else None
            self._accumulate_epoch_stats(
                flat_raw=flat.detach(),
                indices=indices.detach(),
                z_e_flat=flat.detach(),
                z_q_flat=z_q.reshape(-1, D).detach(),
                valid_mask=valid_vec,
            )

        with torch.no_grad():
            if mask is not None:
                valid = mask.reshape(-1)
                idx_use = indices[valid] if valid.any() else indices[:0]
                usage_inst = torch.bincount(idx_use, minlength=self.K).float()
            else:
                usage_inst = torch.bincount(indices, minlength=self.K).float()
            total = usage_inst.sum().clamp_min(1.0)
            probs = usage_inst / total
            nz = probs > 0
            perplexity = torch.exp(-(probs[nz] * probs[nz].log()).sum()) if nz.any() else torch.tensor(0.0, device=z_e.device)
            dead_ratio = (usage_inst == 0).float().mean()

        stats = torch.stack([perplexity, dead_ratio])
        return z_q_st, z_q, indices.view(B, M), stats


# ---------------------------------------------
# Learnable-Query tokenizer: L -> N
# ---------------------------------------------
class LatentTokenizer(nn.Module):
    def __init__(self, d_model: int, n_tokens: int = 32, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.d = int(d_model)
        self.queries = nn.Parameter(torch.randn(self.n_tokens, self.d) * 0.02)
        self.layers = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        for _ in range(int(n_layers)):
            self.layers.append(nn.ModuleDict({
                "ln_q": nn.LayerNorm(self.d),
                "ln_kv": nn.LayerNorm(self.d),
                "attn": nn.MultiheadAttention(self.d, int(n_heads), batch_first=True, dropout=dropout),
                "ln_o": nn.LayerNorm(self.d),
                "ffn": nn.Sequential(
                    nn.Linear(self.d, 4*self.d),
                    nn.GELU(),
                    nn.Linear(4*self.d, self.d),
                ),
                "ffn_drop": nn.Dropout(dropout),
            }))

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # x: [B,L,d], key_padding_mask: [B,L] with True for PAD
        B = x.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)      # [B,N,d]
        kv = x                                                # [B,L,d]
        kpm = key_padding_mask  # True=PAD
        for blk in self.layers:
            qn  = blk["ln_q"](q)
            kvn = blk["ln_kv"](kv)
            out, _ = blk["attn"](qn, kvn, kvn, key_padding_mask=kpm, need_weights=False)
            q = q + self.drop(out)
            q = q + blk["ffn_drop"](blk["ffn"](blk["ln_o"](q)))
        return q  # [B,N,d]


# ---------------------------------------------
# Geometry helpers
# ---------------------------------------------
def _unit(v: Tensor, eps: float = 1e-8) -> Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def _random_rotation(B: int, device) -> Tensor:
    u1 = torch.rand(B, device=device)
    u2 = torch.rand(B, device=device)
    u3 = torch.rand(B, device=device)
    q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)
    x, y, z, w = q1, q2, q3, q4
    R = torch.stack([
        1 - 2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w),
        2*(x*y + z*w),   1 - 2*(x*x+z*z), 2*(y*z - x*w),
        2*(x*z - y*w),   2*(y*z + x*w),   1 - 2*(x*x+y*y),
    ], dim=-1).view(B, 3, 3)
    return R

def _dihedral_cos_sin(x: Tensor) -> Tensor:
    v1 = x[:, 1:-2, :] - x[:, :-3, :]
    v2 = x[:, 2:-1, :] - x[:, 1:-2, :]
    v3 = x[:, 3:, :]   - x[:, 2:-1, :]
    b1 = _unit(v1)
    b2 = _unit(v2)
    b3 = _unit(v3)
    n1 = _unit(torch.cross(b1, b2, dim=-1))
    n2 = _unit(torch.cross(b2, b3, dim=-1))
    m1 = torch.cross(n1, _unit(b2), dim=-1)
    cos_t = (n1 * n2).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    sin_t = (m1 * n2).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    return torch.cat([cos_t, sin_t], dim=-1)


# ---------------------------------------------
# VQ-VAE (minimal, query-tokenized) - FIXED VERSION
# ---------------------------------------------
class VQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 350,
        codebook_size: int = 512,
        code_dim: int = 128,
        beta: float = 0.25,
        use_vq: bool = True,
        label_smoothing: float = 0.0,
        ss_tv_lambda: float = 0.0,
        usage_entropy_lambda: float = 0.0,
        xyz_align_alpha: float = 0.7,
        dist_lambda: float = 0.0,
        rigid_aug_prob: float = 0.0,
        pairwise_sample_k: int = 32,
        codebook_init_path: Optional[str] = None,
        ema_decay_start: float = 0.98,
        ema_decay_end: float = 0.98,
        ema_decay_warm_steps: int = 0,
        soft_vq_use: bool = False,
        soft_vq_tau_start: float = 2.0,
        soft_vq_tau_end: float = 0.5,
        soft_vq_tau_warm_steps: int = 0,
        soft_vq_alpha_warm_steps: int = 0,
        noise_warmup_steps: int = 0,
        max_noise_std: float = 0.0,
        latent_tokens: int = 32,
        tokenizer_heads: int = 8,
        tokenizer_layers: int = 2,
        tokenizer_dropout: float = 0.1,
        reinit_dead_codes: bool = True,
        reinit_prob: float = 1.0,
        dead_usage_threshold: int = 0,
        ema_update_freeze_steps: int = 0,
        print_init: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.code_dim = int(code_dim)
        self.max_seq_len = int(max_seq_len)
        self._beta = float(beta)
        self.use_vq = bool(use_vq)
        self.label_smoothing = float(label_smoothing)
        self.ss_tv_lambda = float(ss_tv_lambda)
        self.usage_entropy_lambda = float(usage_entropy_lambda)
        self.xyz_align_alpha = float(xyz_align_alpha)
        self.rigid_aug_prob = float(rigid_aug_prob)
        self.dist_lambda = float(dist_lambda)
        self.pairwise_sample_k = int(pairwise_sample_k)
        self.ema_decay_start = float(ema_decay_start)
        self.ema_decay_end = float(ema_decay_end)
        self.ema_decay_warm_steps = int(ema_decay_warm_steps)

        self.soft_vq_use = bool(soft_vq_use)
        self.soft_vq_tau_start = float(soft_vq_tau_start)
        self.soft_vq_tau_end = float(soft_vq_tau_end)
        self.soft_vq_tau_warm_steps = int(soft_vq_tau_warm_steps)
        self.soft_vq_alpha_warm_steps = int(soft_vq_alpha_warm_steps)

        self.noise_warmup_steps = int(noise_warmup_steps)
        self.max_noise_std = float(max_noise_std)

        self.codebook_init_path = codebook_init_path
        self.ema_update_freeze_steps = int(ema_update_freeze_steps)

        self._curr_epoch = 0
        self.training_steps = 0
        self._ema_decay_override = None
        self._data_mean = None
        self._data_std = None

        # Encoder
        self.input_proj = nn.Linear(3, self.hidden_dim)
        self.ss_input_proj = nn.Linear(3, self.hidden_dim)
        self.inp_dropout = nn.Dropout(p=0.1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=num_heads, batch_first=True, dropout=0.1, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.enc_ln = nn.LayerNorm(self.hidden_dim)
        self.to_code = nn.Linear(self.hidden_dim, self.code_dim)
        self.ln_geo = nn.LayerNorm(self.hidden_dim)
        self.ln_ss = nn.LayerNorm(self.hidden_dim)
        ss_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, 
            nhead=num_heads, 
            batch_first=True, 
            dropout=0.1, 
            norm_first=True
        )
        self.ss_encoder = nn.TransformerEncoder(ss_enc_layer, num_layers=2) 
        self._grad_monitor_enabled = False
        self._grad_hooks = []

        # sinusoidal pos enc
        pe = torch.zeros(self.max_seq_len, self.hidden_dim)
        pos = torch.arange(0, self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * (-math.log(10000.0) / self.hidden_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_enc", pe.unsqueeze(0))

        # Tokenizer L->N
        self.latent_n_tokens = int(latent_tokens)
        self.tokenizer = LatentTokenizer(
            d_model=self.hidden_dim,
            n_tokens=self.latent_n_tokens,
            n_heads=int(tokenizer_heads),
            n_layers=int(tokenizer_layers),
            dropout=float(tokenizer_dropout),
        )

        fuse_in = self.hidden_dim * 2
        self.fuse_mlp = nn.Sequential(
            nn.Linear(fuse_in, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # Quantizer
        self.quantizer = VectorQuantizerEMA(
            num_embeddings=codebook_size,
            embedding_dim=self.code_dim,
            beta=beta,
            decay=0.98,
            eps=1e-5,
            reinit_dead_codes=reinit_dead_codes,
            reinit_prob=reinit_prob,
            dead_usage_threshold=dead_usage_threshold,
            print_init=print_init,
        )
        self.quantizer.beta = self._beta

        # Decoder
        self.from_code = nn.Linear(self.code_dim, self.hidden_dim)
        self.mem_ln = nn.LayerNorm(self.hidden_dim)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim, nhead=num_heads, batch_first=True, dropout=0.1, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.query_embed = nn.Embedding(self.max_seq_len, self.hidden_dim)
        nn.init.normal_(self.query_embed.weight, std=0.02)

        self.head_xyz = nn.Linear(self.hidden_dim, 3)
        self.head_ss = nn.Linear(self.hidden_dim, 3)

        # optional codebook init
        if self.use_vq and self.codebook_init_path:
            try:
                C = np.load(self.codebook_init_path).astype(np.float32)
                C = torch.from_numpy(C)
                self.init_codebook_from_centroids(C)
                if print_init:
                    print(f"[VQ] Codebook init from {self.codebook_init_path} shape={tuple(C.shape)}")
            except Exception as e:
                if print_init:
                    print(f"[VQ] Failed to load codebook: {e}")

        if print_init:
            print(
                f"[Model] VQVAE(minimal): H={self.hidden_dim}, Dcode={self.code_dim}, "
                f"use_vq={self.use_vq}, tokensN={self.latent_n_tokens}, ema_freeze_steps={self.ema_update_freeze_steps}"
            )

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = float(value)
        self.quantizer.beta = float(value)

    def set_epoch_context(self, epoch: int, steps_per_epoch: int = 1):
        self._curr_epoch = int(epoch)

    @torch.no_grad()
    def set_data_stats(self, mean_xyz: Tensor, std_xyz: Tensor):
        device = self.head_xyz.weight.device
        mean = torch.as_tensor(mean_xyz, dtype=torch.float32, device=device).view(1, 1, 3)
        std = torch.as_tensor(std_xyz, dtype=torch.float32, device=device).view(1, 1, 3)
        self._data_mean = mean
        self._data_std = std

    @torch.no_grad()
    def init_codebook_from_centroids(self, centroids: Tensor):
        if centroids.shape != (self.quantizer.K, self.code_dim):
            raise ValueError(f"Centroid shape mismatch: expected {(self.quantizer.K, self.code_dim)}, got {tuple(centroids.shape)}")
        self.quantizer.embedding.copy_(centroids)
        if hasattr(self.quantizer, "ema_embedding"):
            self.quantizer.ema_embedding.copy_(centroids)
        if hasattr(self.quantizer, "ema_cluster_size"):
            self.quantizer.ema_cluster_size.copy_(torch.ones(self.quantizer.K, device=centroids.device))
        print(f"[Codebook Init] Loaded {centroids.shape} centroids.")

    def _linear_schedule(self, target: float, warmup_steps: int) -> float:
        if warmup_steps <= 0:
            return target
        t = min(1.0, float(self.training_steps) / float(warmup_steps))
        return target * t

    def _interp_linear(self, start: float, end: float, step: int, warm_steps: int) -> float:
        if warm_steps <= 0:
            return end
        t = min(1.0, max(0.0, step) / float(warm_steps))
        return (1.0 - t) * start + t * end

    def _compute_stats(self, indices: Tensor, device: torch.device) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            usage_inst = torch.bincount(indices.reshape(-1), minlength=self.quantizer.K).float()
            total = usage_inst.sum().clamp_min(1.0)
            probs = usage_inst / total
            nz = probs > 0
            perplexity = torch.exp(-(probs[nz] * probs[nz].log()).sum()) if nz.any() else torch.tensor(0.0, device=device)
            dead_ratio = (usage_inst == 0).float().mean()
        return perplexity, dead_ratio

    def encode(self, x, mask=None):
        B, L, _ = x.shape
        xyz = x[..., :3]
        h_geo = self.input_proj(xyz)
        h_geo = self.inp_dropout(h_geo) + self.pos_enc[:, :L, :]
        h_enc_geo = self.encoder(h_geo, src_key_padding_mask=(~mask) if mask is not None else None)
        h_enc_geo = self.enc_ln(h_enc_geo)
        g = self.ln_geo(h_enc_geo)
    
        ss_onehot = x[..., 3:]
        h_ss = self.ss_input_proj(ss_onehot)
        h_ss = h_ss + self.pos_enc[:, :L, :]  # 添加位置编码
        h_enc_ss = self.ss_encoder(h_ss, src_key_padding_mask=(~mask) if mask is not None else None)
        s = self.ln_ss(h_enc_ss)
        # 添加梯度监控（简化版）
        if self.training and self._grad_monitor_enabled and hasattr(self, '_create_grad_hook_func'):
            create_hook = self._create_grad_hook_func
            self._grad_hooks.append(g.register_hook(create_hook("Geo-Branch")))
            self._grad_hooks.append(s.register_hook(create_hook("SS-Branch")))
        h_fuse_tokens = self.fuse_mlp(torch.cat([g, s], dim=-1))
        # 监控融合层输出
        if self.training and self._grad_monitor_enabled and hasattr(self, '_create_grad_hook_func'):
            self._grad_hooks.append(h_fuse_tokens.register_hook(create_hook("Fusion-Output")))
        return h_fuse_tokens, h_enc_geo, h_enc_ss

    def enable_grad_monitor(self, enabled: bool = True):
        """启用或禁用梯度监控"""
        self._grad_monitor_enabled = enabled
        
        # 移除旧的hook
        for hook in self._grad_hooks:
            hook.remove()
        self._grad_hooks.clear()
        
        if enabled:
            print(f"[Grad Monitor] Enabled")
            # 在启用时创建hook
            def create_grad_hook(name):
                def hook(grad):
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        print(f"[GRAD-ERROR] {name}: NaN or Inf detected!")
                    else:
                        grad_norm = grad.norm().item()
                        if grad_norm > 1e-6:  # 只打印有意义的梯度
                            print(f"[GRAD] {name}: norm={grad_norm:.6f}, mean={grad.mean().item():.6f}, std={grad.std().item():.6f}")
                return hook
            
            # 为关键层注册hook（在forward中动态获取tensor）
            self._create_grad_hook_func = create_grad_hook
        else:
            print(f"[Grad Monitor] Disabled")
            self._create_grad_hook_func = None
            
    def print_grad_summary(self):
        """打印模型各部分的梯度统计"""
        if not self.training:
            print("[Grad Summary] Model is in eval mode, no gradients")
            return
        
        total_params = 0
        geo_params = 0
        ss_params = 0
        fusion_params = 0
        vq_params = 0
        decoder_params = 0
        
        geo_grad_norm = 0.0
        ss_grad_norm = 0.0
        fusion_grad_norm = 0.0
        vq_grad_norm = 0.0
        decoder_grad_norm = 0.0
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_params += 1
                
                if 'encoder' in name and 'ss_' not in name:
                    geo_params += 1
                    geo_grad_norm += grad_norm
                elif 'ss_' in name or ('encoder' in name and 'ss_' in name):
                    ss_params += 1
                    ss_grad_norm += grad_norm
                elif 'fuse' in name:
                    fusion_params += 1
                    fusion_grad_norm += grad_norm
                elif 'quantizer' in name:
                    vq_params += 1
                    vq_grad_norm += grad_norm
                elif 'decoder' in name or 'head_ss' in name or 'head_xyz' in name:
                    decoder_params += 1
                    decoder_grad_norm += grad_norm
        
        print(f"[Grad Summary] Total params with grad: {total_params}")
        if geo_params > 0:
            print(f"  Geo branch: {geo_params} params, avg_grad_norm={geo_grad_norm/max(geo_params,1):.6f}")
        if ss_params > 0:
            print(f"  SS branch: {ss_params} params, avg_grad_norm={ss_grad_norm/max(ss_params,1):.6f}")
        if fusion_params > 0:
            print(f"  Fusion: {fusion_params} params, avg_grad_norm={fusion_grad_norm/max(fusion_params,1):.6f}")
        if vq_params > 0:
            print(f"  VQ: {vq_params} params, avg_grad_norm={vq_grad_norm/max(vq_params,1):.6f}")
        if decoder_params > 0:
            print(f"  Decoder: {decoder_params} params, avg_grad_norm={decoder_grad_norm/max(decoder_params,1):.6f}")

    def _tokenize_to_codes(self, h_tokens: Tensor, mask: Optional[Tensor]) -> Tensor:
        kpm = (~mask) if mask is not None else None
        h_mem = self.tokenizer(h_tokens, key_padding_mask=kpm)
        z_e_tokens = self.to_code(h_mem)
        return z_e_tokens

    def decode(self, z_for_decode: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B = z_for_decode.size(0)
        L = mask.size(1) if mask is not None else self.max_seq_len

        memory = self.mem_ln(self.from_code(z_for_decode))
        q = self.query_embed.weight[:L].unsqueeze(0).expand(B, L, -1)
        q = q + self.pos_enc[:, :L, :]

        tgt_key_padding_mask = (~mask) if mask is not None else None
        mem_key_padding_mask = None

        h_dec = self.decoder(
            tgt=q,
            memory=memory,
            tgt_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=mem_key_padding_mask,
        )
        xyz_pred = self.head_xyz(h_dec)
        ss_logits = self.head_ss(h_dec)
        return torch.cat([xyz_pred, ss_logits], dim=-1)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> List[Tensor]:
        if self.training and hasattr(self, 'training_steps'):
            if self.training_steps % 5000 == 0:  
                self.print_grad_summary()
        target = x.clone()
        x_in = x

        # rigid aug - only on input
        if self.training and self.rigid_aug_prob > 0.0 and torch.rand(()) < self.rigid_aug_prob:
            B, L, _ = x_in.shape
            device = x_in.device
            R = _random_rotation(B, device)
            t = torch.randn(B, 1, 3, device=device) * 0.02
            xyz = x_in[..., :3]
            ss = x_in[..., 3:]
            xyz_aug = torch.einsum("bij,blj->bli", R, xyz) + t
            x_in = torch.cat([xyz_aug, ss], dim=-1)

        # coord noise - only on input
        if self.training and self.max_noise_std > 0.0:
            factor = (min(1.0, self.training_steps / float(self.noise_warmup_steps))
                      if self.noise_warmup_steps > 0 else 1.0)
            noise_std = self.max_noise_std * factor
            if noise_std > 0.0:
                noise = torch.randn_like(x_in[..., :3]) * noise_std
                x_in = torch.cat([x_in[..., :3] + noise, x_in[..., 3:]], dim=-1)

        # EMA decay schedule
        if hasattr(self.quantizer, "decay"):
            if self._ema_decay_override is not None:
                self.quantizer.decay = float(self._ema_decay_override)
            else:
                self.quantizer.decay = float(
                    self._interp_linear(self.ema_decay_start, self.ema_decay_end, 
                                      self.training_steps, self.ema_decay_warm_steps)
                )

        h_fuse_tokens, h_enc_geo, h_ss_tokens = self.encode(x_in, mask=mask)
        if self.training:
            self.training_steps += 1

        z_e_tokens = self._tokenize_to_codes(h_fuse_tokens, mask)

        eta, tau = 0.0, 0.0

        if not self.use_vq:
            z_for_decode = z_e_tokens
            z_q_raw = z_e_tokens
            indices = torch.zeros(z_e_tokens.size(0), z_e_tokens.size(1), dtype=torch.long, device=z_e_tokens.device)
            ppl = torch.tensor(0.0, device=x_in.device)
            dead = torch.tensor(0.0, device=x_in.device)
        else:
            do_ema_update = self.training and (self.training_steps >= self.ema_update_freeze_steps)
            
            if self.soft_vq_use and self.training:
                Bsz, Ntok, Ddim = z_e_tokens.shape
                flat_ze = z_e_tokens.reshape(-1, Ddim)
                
                with torch.no_grad():
                    emb = self.quantizer.embedding
                
                tau = self._interp_linear(self.soft_vq_tau_start, self.soft_vq_tau_end, 
                                        self.training_steps, self.soft_vq_tau_warm_steps)
                
                diff = flat_ze.unsqueeze(1) - emb.unsqueeze(0)  
                d2 = (diff * diff).sum(dim=-1)                  
                logits = -d2 / max(1e-8, tau)                   
                probs = F.softmax(logits, dim=-1)
                
                z_soft = (probs @ emb).view(Bsz, Ntok, Ddim)
                
                with torch.no_grad():
                    indices = torch.argmin(d2, dim=1)
                    z_q_hard = F.embedding(indices, emb).view(Bsz, Ntok, Ddim)
                
                alpha = self._linear_schedule(1.0, self.soft_vq_alpha_warm_steps)
                
                z_q_mix = (1 - alpha) * z_soft + alpha * z_q_hard
                z_for_decode = z_e_tokens + (z_q_mix - z_e_tokens).detach()  
                z_q_raw = z_q_hard
                
                do_ema_update = do_ema_update and (alpha >= 0.25)
                
                if self.training and do_ema_update:
                    self.quantizer._ema_update(flat_ze.detach(), indices.detach())
                
                ppl, dead = self._compute_stats(indices.view(Bsz, Ntok), z_e_tokens.device)
                
            else:
                z_q_st, z_q_raw, indices, stats = self.quantizer(
                    z_e_tokens, do_ema_update=do_ema_update, allow_reinit=do_ema_update, mask=None
                )
                ppl, dead = stats[0], stats[1]
                z_for_decode = z_q_st
           
            if self.training and do_ema_update:
                reinit_interval = 500
                min_steps = max(self.ema_update_freeze_steps, 800)
                
                if (self.training_steps % reinit_interval == 0) and (self.training_steps >= min_steps):
                    B, M, D = z_e_tokens.shape
                    flat = z_e_tokens.reshape(-1, D)
                    usage_signal = torch.bincount(indices.reshape(-1), minlength=self.quantizer.K).float()
                    self.quantizer._maybe_reinit_dead_codes(flat.detach(), usage_signal)

        recons = self.decode(z_for_decode, mask=mask)

        if self.training and self.training_steps % 500 == 0:
            current_decay = getattr(self.quantizer, 'decay', 0.0)
            print(f"Step {self.training_steps}: decay={current_decay:.4f}, tau={tau:.4f}, beta={self.quantizer.beta:.4f}, "
                  f"ema_frozen={self.training_steps < self.ema_update_freeze_steps}")

        vq_pack = (z_q_raw, z_e_tokens, indices, ppl, dead)
        return [recons, target, vq_pack, mask]

    @staticmethod
    def _mse_per_sample(a: Tensor, b: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        diff2 = (a - b).pow(2).sum(dim=-1)
        if mask is None:
            return diff2.mean(dim=1)
        m = mask.float()
        den = m.sum(dim=1).clamp_min(1.0)
        return (diff2 * m).sum(dim=1) / den

    @staticmethod
    def _masked_mse(a: Tensor, b: Tensor, mask: Optional[Tensor]) -> Tensor:
        diff = (a - b) ** 2
        if mask is None:
            return diff.mean()
        m = mask.unsqueeze(-1).float()
        return (diff * m).sum() / m.sum().clamp_min(1.0)

    def _ss_label_smoothing_ce(self, logits: Tensor, labels: Tensor, mask: Optional[Tensor], eps: float) -> Tensor:
        B, L, C = logits.shape
        logp = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(eps / (C - 1))
            true_dist.scatter_(-1, labels.unsqueeze(-1), 1.0 - eps)
        kl = F.kl_div(logp, true_dist, reduction="none").sum(dim=-1)
        if mask is not None:
            m = mask.float()
            return (kl * m).sum() / m.sum().clamp_min(1.0)
        return kl.mean()

    @staticmethod
    def _center(x: Tensor, mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if mask is None:
            mu = x.mean(dim=1, keepdim=True)
            return x - mu, mu
        m = mask.float().unsqueeze(-1)
        den = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mu = (x * m).sum(dim=1, keepdim=True) / den
        return x - mu, mu

    @staticmethod
    def _kabsch_rt_safe(a_xyz: Tensor, b_xyz: Tensor, mask: Optional[Tensor], eps: float = 1e-8) -> Tuple[Tensor, Tensor, Tensor]:
        B, L, _ = a_xyz.shape
        device = a_xyz.device
        with torch.no_grad():
            a_c, a_mu = VQVAE._center(a_xyz, mask)
            b_c, b_mu = VQVAE._center(b_xyz, mask)
            if mask is None:
                H = torch.einsum("bli,blj->bij", a_c, b_c)
            else:
                m = mask.float().unsqueeze(-1)
                H = torch.einsum("bli,blj->bij", a_c * m, b_c)
            try:
                U, S, Vh = torch.linalg.svd(H)
                V = Vh.transpose(-2, -1)
                det = torch.det(V @ U.transpose(-2, -1))
                D = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
                D[:, -1, -1] = (det >= 0).to(D.dtype) * 2.0 - 1.0
                R = V @ D @ U.transpose(-2, -1)
            except Exception:
                R = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
            t = b_mu - torch.einsum("bli,bij->blj", a_mu, R)
            ok = torch.isfinite(R).all(dim=(1, 2)) & torch.isfinite(t).all(dim=(1, 2))
        return R.detach(), t.detach(), ok

    @staticmethod
    def _apply_rt(x: Tensor, R: Tensor, t: Tensor) -> Tensor:
        return torch.einsum("bli,bij->blj", x, R) + t

    @staticmethod
    def _pairwise_pdm(a_xyz: Tensor, b_xyz: Tensor, mask: Optional[Tensor], window: int = 8) -> Tensor:
        B, L, _ = a_xyz.shape
        if L < 2 or window <= 1:
            return torch.tensor(0.0, device=a_xyz.device)
        offs = list(range(1, window))
        loss_acc = 0.0
        cnt = 0.0
        for d in offs:
            ai = a_xyz[:, :-d, :]
            aj = a_xyz[:,  d:, :]
            bi = b_xyz[:, :-d, :]
            bj = b_xyz[:,  d:, :]
            da = (ai - aj).norm(dim=-1)
            db = (bi - bj).norm(dim=-1)
            if mask is not None:
                m = (mask[:, :-d] & mask[:, d:]).float()
                num = (m * (da - db).pow(2)).sum()
                den = m.sum().clamp_min(1.0)
                loss_acc = loss_acc + num / den
            else:
                loss_acc = loss_acc + F.mse_loss(da, db)
            cnt += 1.0
        return loss_acc / max(1.0, cnt)

    @staticmethod
    def _window_kabsch_loss(a_xyz: Tensor, b_xyz: Tensor, mask: Optional[Tensor], win: int = 16, stride: int = 8) -> Tensor:
        B, L, _ = a_xyz.shape
        if L < 3 or win < 3:
            return torch.tensor(0.0, device=a_xyz.device)
        loss_acc = 0.0
        nwin = 0
        for s in range(0, L - win + 1, max(1, stride)):
            a_win = a_xyz[:, s:s+win, :]
            b_win = b_xyz[:, s:s+win, :]
            sub_mask = (mask[:, s:s+win] if mask is not None else None)

            if sub_mask is not None:
                ok_pts = (sub_mask.sum(dim=1) >= 3)
                if not ok_pts.any():
                    continue
            else:
                ok_pts = None

            R, t, ok = VQVAE._kabsch_rt_safe(a_win, b_win, sub_mask)
            if sub_mask is not None:
                ok = ok & ok_pts.to(ok.device)
            if not ok.any():
                continue

            a_aln = VQVAE._apply_rt(a_win, R, t)
            if sub_mask is None:
                mse = ((a_aln - b_win) ** 2).mean(dim=(1, 2))
                use = torch.ones(B, dtype=torch.bool, device=a_xyz.device)
            else:
                m = sub_mask.float().unsqueeze(-1)
                den = m.sum(dim=(1, 2)).clamp_min(1.0)
                mse = ((a_aln - b_win) ** 2 * m).sum(dim=(1, 2)) / den
                use = sub_mask.sum(dim=1) >= 3

            sel = use & ok
            if sel.any():
                loss_acc = loss_acc + mse[sel].mean()
                nwin += 1

        if nwin == 0:
            return torch.tensor(0.0, device=a_xyz.device)
        return loss_acc / float(nwin)

    @staticmethod
    def _frenet_regularizers(a_xyz: Tensor, mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        B, L, _ = a_xyz.shape
        device = a_xyz.device
        if L >= 3:
            d1 = a_xyz[:, 1:, :] - a_xyz[:, :-1, :]
            d2 = d1[:, 1:, :] - d1[:, :-1, :]
            kappa = (d2.pow(2).sum(dim=-1))
            if mask is not None:
                m = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).float()
                kappa_reg = (kappa * m).sum() / m.sum().clamp_min(1.0)
            else:
                kappa_reg = kappa.mean()
        else:
            kappa_reg = torch.tensor(0.0, device=device)

        if L >= 5:
            dih = _dihedral_cos_sin(a_xyz)
            dih_next = dih[:, 1:, :]
            dih_prev = dih[:, :-1, :]
            tau_var = (dih_next - dih_prev).pow(2).sum(dim=-1)
            if mask is not None:
                m = (mask[:, 4:] & mask[:, 3:-1] & mask[:, 2:-2] & mask[:, 1:-3] & mask[:, :-4]).float()
                tau_reg = (tau_var * m).sum() / m.sum().clamp_min(1.0)
            else:
                tau_reg = tau_var.mean()
        else:
            tau_reg = torch.tensor(0.0, device=device)
        return kappa_reg, tau_reg

    @staticmethod
    def _long_range_pdm(a_xyz: Tensor, b_xyz: Tensor, mask: Optional[Tensor],
                        min_sep: int = 24, stride: int = 8, max_offsets: int = 8) -> Tensor:
        B, L, _ = a_xyz.shape
        if L < min_sep + 1:
            return torch.tensor(0.0, device=a_xyz.device)
        total = 0.0
        cnt = 0
        for off_i in range(0, max(1, max_offsets)):
            for i in range(0, L, max(1, stride)):
                j = i + min_sep + off_i
                if j >= L:
                    break
                da = (a_xyz[:, j, :] - a_xyz[:, i, :]).norm(dim=-1)
                db = (b_xyz[:, j, :] - b_xyz[:, i, :]).norm(dim=-1)
                if mask is not None:
                    m = (mask[:, j] & mask[:, i]).float()
                    num = (m * (da - db).pow(2)).sum()
                    den = m.sum().clamp_min(1.0)
                    total = total + num / den
                else:
                    total = total + F.mse_loss(da, db)
                cnt += 1
        if cnt == 0:
            return torch.tensor(0.0, device=a_xyz.device)
        return total / float(cnt)

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        target = args[1]
        vq_pack = args[2]
        mask = args[3] if len(args) > 3 else None

        zq_raw, ze_raw, indices, ppl_unmasked, dead_unmasked = vq_pack

        ss_weight = float(kwargs.get("ss_weight", 1.0))
        bond_len_w = float(kwargs.get("bond_length_weight", 0.0))
        bond_ang_w = float(kwargs.get("bond_angle_weight", 0.0))
        xyz_tv_lambda = float(kwargs.get("xyz_tv_lambda", 0.0))
        dir_weight = float(kwargs.get("dir_weight", 0.0))
        dih_weight = float(kwargs.get("dih_weight", 0.0))
        rmsd_weight = float(kwargs.get("rmsd_weight", 1.0))
        pdm_weight = float(kwargs.get("pdm_weight", 0.0))
        win_kabsch_weight = float(kwargs.get("win_kabsch_weight", 0.0))
        kappa_weight = float(kwargs.get("kappa_weight", 0.0))
        tau_weight = float(kwargs.get("tau_weight", 0.0))
        lr_pdm_weight = float(kwargs.get("lr_pdm_weight", 0.0))
        pdm_window = int(kwargs.get("pdm_window", 8))
        win_kabsch_size = int(kwargs.get("win_kabsch_size", 16))
        win_kabsch_stride = int(kwargs.get("win_kabsch_stride", 8))
        lr_min_sep = int(kwargs.get("lr_min_sep", 24))
        lr_stride = int(kwargs.get("lr_stride", 8))
        lr_max_offsets = int(kwargs.get("lr_max_offsets", 8))

        re_xyz = recons[:, :, :3]
        re_ss_logits = recons[:, :, 3:]
        gt_xyz = target[:, :, :3]
        gt_ss_onehot = target[:, :, 3:]

        # Fixed RMSD and alignment calculation
        raw_mse_per_sample = self._mse_per_sample(re_xyz, gt_xyz, mask)
        loss_xyz_raw = raw_mse_per_sample.mean()
        
        loss_xyz_aligned = loss_xyz_raw
        aln_mse_per_sample = raw_mse_per_sample.clone()
        re_aln = re_xyz
        
        try:
            can_align = (re_xyz.size(1) >= 3)
            if mask is not None and can_align:
                valid_per_sample = mask.sum(dim=1) >= 3
                can_align = valid_per_sample.any()
            
            if can_align:
                R, t, ok = self._kabsch_rt_safe(re_xyz, gt_xyz, mask)
                
                if ok.any():
                    re_aln = self._apply_rt(re_xyz, R, t)
                    aln_mse_per_sample = self._mse_per_sample(re_aln, gt_xyz, mask)
                    
                    if mask is not None:
                        valid_mask = mask.sum(dim=1) >= 3
                        best_mse_per_sample = torch.where(
                            valid_mask & ok,
                            torch.minimum(raw_mse_per_sample, aln_mse_per_sample),
                            raw_mse_per_sample
                        )
                    else:
                        best_mse_per_sample = torch.where(
                            ok,
                            torch.minimum(raw_mse_per_sample, aln_mse_per_sample),
                            raw_mse_per_sample
                        )
                    
                    loss_xyz_aligned = best_mse_per_sample.mean()
                    
        except Exception as e:
            if self.training and self.training_steps % 1000 == 0:
                print(f"Kabsch alignment failed: {e}")
            pass

        loss_xyz = self.xyz_align_alpha * loss_xyz_aligned + (1.0 - self.xyz_align_alpha) * loss_xyz_raw

        # Independent RMSD evaluation metrics
        with torch.no_grad():
            rmsd_raw = torch.sqrt(raw_mse_per_sample.clamp_min(1e-12)).mean()
            
            if 'best_mse_per_sample' in locals():
                rmsd_aligned = torch.sqrt(best_mse_per_sample.clamp_min(1e-12)).mean()
            else:
                rmsd_aligned = rmsd_raw

        # SS CE
        gt_ss_labels = gt_ss_onehot.argmax(dim=-1)
        if self.label_smoothing and self.label_smoothing > 0.0:
            loss_ss = self._ss_label_smoothing_ce(re_ss_logits, gt_ss_labels, mask, eps=self.label_smoothing)
        else:
            if mask is not None:
                ce = F.cross_entropy(
                    re_ss_logits.reshape(-1, re_ss_logits.size(-1)),
                    gt_ss_labels.reshape(-1),
                    reduction="none",
                ).reshape(gt_ss_labels.shape)
                m = mask.float()
                loss_ss = (ce * m).sum() / m.sum().clamp_min(1.0)
            else:
                loss_ss = F.cross_entropy(
                    re_ss_logits.reshape(-1, re_ss_logits.size(-1)),
                    gt_ss_labels.reshape(-1)
                )

        # SS TV
        if self.ss_tv_lambda > 0.0:
            p = F.softmax(re_ss_logits, dim=-1)
            if p.size(1) >= 2:
                tv = (p[:, 1:, :] - p[:, :-1, :]).abs().sum(dim=-1)
                if mask is not None:
                    tv_mask = (mask[:, 1:] & mask[:, :-1]).float()
                    ss_tv = (tv * tv_mask).sum() / tv_mask.sum().clamp_min(1.0)
                else:
                    ss_tv = tv.mean()
            else:
                ss_tv = torch.tensor(0.0, device=recons.device)
        else:
            ss_tv = torch.tensor(0.0, device=recons.device)

        # real coords
        std = self._data_std if self._data_std is not None else None
        mean = self._data_mean if self._data_mean is not None else None
        def to_real(x):
            if std is not None:
                return x * std + (mean if mean is not None else 0.0)
            return x
        re_xyz_real = to_real(re_xyz)
        gt_xyz_real = to_real(gt_xyz)

        # bond length
        if re_xyz_real.size(1) >= 2:
            re_diff = re_xyz_real[:, 1:, :] - re_xyz_real[:, :-1, :]
            gt_diff = gt_xyz_real[:, 1:, :] - gt_xyz_real[:, :-1, :]
            re_len = torch.norm(re_diff, dim=-1)
            gt_len = torch.norm(gt_diff, dim=-1)
            if mask is not None:
                pair_mask = (mask[:, 1:] & mask[:, :-1]).float()
                bl = ((re_len - gt_len) ** 2 * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)
            else:
                bl = F.mse_loss(re_len, gt_len)
        else:
            bl = torch.tensor(0.0, device=recons.device)

        # bond angle
        if re_xyz_real.size(1) >= 3:
            v1_rec = re_xyz_real[:, 1:-1, :] - re_xyz_real[:, :-2, :]
            v2_rec = re_xyz_real[:, 2:, :]   - re_xyz_real[:, 1:-1, :]
            v1_gt  = gt_xyz_real[:, 1:-1, :] - gt_xyz_real[:, :-2, :]
            v2_gt  = gt_xyz_real[:, 2:, :]   - gt_xyz_real[:, 1:-1, :]
            def _cos(v1, v2, eps=1e-8):
                v1n = v1 / (v1.norm(dim=-1, keepdim=True) + eps)
                v2n = v2 / (v2.norm(dim=-1, keepdim=True) + eps)
                return (v1n * v2n).sum(dim=-1)
            cos_rec = _cos(v1_rec, v2_rec)
            cos_gt  = _cos(v1_gt,  v2_gt)
            if mask is not None:
                tri_mask = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).float()
                ba = ((cos_rec - cos_gt) ** 2 * tri_mask).sum() / tri_mask.sum().clamp_min(1.0)
            else:
                ba = F.mse_loss(cos_rec, cos_gt)
        else:
            ba = torch.tensor(0.0, device=recons.device)

        # direction
        if re_xyz_real.size(1) >= 2:
            u_rec = _unit(re_xyz_real[:, 1:, :] - re_xyz_real[:, :-1, :])
            u_gt  = _unit(gt_xyz_real[:, 1:, :] - gt_xyz_real[:, :-1, :])
            cos_u = (u_rec * u_gt).sum(dim=-1)
            dir_err = (1.0 - cos_u)
            if mask is not None:
                pair_mask = (mask[:, 1:] & mask[:, :-1]).float()
                dir_loss = (dir_err * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)
            else:
                dir_loss = dir_err.mean()
        else:
            dir_loss = torch.tensor(0.0, device=recons.device)

        # dihedral
        if re_xyz_real.size(1) >= 4:
            dih_rec = _dihedral_cos_sin(re_xyz_real)
            dih_gt  = _dihedral_cos_sin(gt_xyz_real)
            if mask is not None:
                di_mask = (mask[:, 3:] & mask[:, 2:-1] & mask[:, 1:-2] & mask[:, :-3]).float()
                dih = ((dih_rec - dih_gt).pow(2).sum(dim=-1) * di_mask).sum() / di_mask.sum().clamp_min(1.0)
            else:
                dih = F.mse_loss(dih_rec, dih_gt)
        else:
            dih = torch.tensor(0.0, device=recons.device)

        geom_loss = bond_len_w * bl + bond_ang_w * ba + dir_weight * dir_loss + dih_weight * dih

        # VQ loss
        if self.use_vq:
            commit = F.mse_loss(zq_raw.detach(), ze_raw)
            vq_loss = self.quantizer.beta * commit
        else:
            vq_loss = torch.tensor(0.0, device=recons.device)

        # usage entropy reg
        usage_reg = torch.tensor(0.0, device=recons.device)
        if self.usage_entropy_lambda > 0.0 and ze_raw.numel() > 0:
            D = ze_raw.size(-1)
            flat_ze = ze_raw.reshape(-1, D)
            with torch.no_grad():
                emb = self.quantizer.embedding.detach()
            logits = flat_ze @ emb.t()
            probs = F.softmax(logits, dim=-1)
            p_code = probs.mean(dim=0)
            entropy = -(p_code * (p_code.clamp_min(1e-12).log())).sum()
            usage_reg = - self.usage_entropy_lambda * entropy

        # XYZ TV
        if xyz_tv_lambda > 0.0 and re_xyz.size(1) >= 3:
            d1 = re_xyz[:, 1:, :] - re_xyz[:, :-1, :]
            d2 = d1[:, 1:, :] - d1[:, :-1, :]
            tv2 = (d2 ** 2).sum(dim=-1)
            if mask is not None:
                tv2_mask = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).float()
                xyz_tv = (tv2 * tv2_mask).sum() / tv2_mask.sum().clamp_min(1.0)
            else:
                xyz_tv = tv2.mean()
        else:
            xyz_tv = torch.tensor(0.0, device=recons.device)

        # extra geometry
        geom_local_pdm = self._pairwise_pdm(re_xyz_real, gt_xyz_real, mask, window=pdm_window) if pdm_weight > 0 else torch.tensor(0.0, device=recons.device)
        geom_winkabsch = self._window_kabsch_loss(re_xyz_real, gt_xyz_real, mask, win=win_kabsch_size, stride=win_kabsch_stride) if win_kabsch_weight > 0 else torch.tensor(0.0, device=recons.device)
        fr_kappa, fr_tau = self._frenet_regularizers(re_xyz_real, mask)
        fr_kappa = fr_kappa if kappa_weight > 0 else torch.tensor(0.0, device=recons.device)
        fr_tau   = fr_tau   if tau_weight > 0 else torch.tensor(0.0, device=recons.device)
        geom_lr_pdm = self._long_range_pdm(re_xyz_real, gt_xyz_real, mask, min_sep=lr_min_sep, stride=lr_stride, max_offsets=lr_max_offsets) if lr_pdm_weight > 0 else torch.tensor(0.0, device=recons.device)

        total_loss = (
            rmsd_weight * loss_xyz
            + ss_weight * loss_ss
            + vq_loss
            + geom_loss
            + self.ss_tv_lambda * ss_tv
            + usage_reg
            + xyz_tv_lambda * xyz_tv
            + pdm_weight * geom_local_pdm
            + win_kabsch_weight * geom_winkabsch
            + kappa_weight * fr_kappa
            + tau_weight * fr_tau
            + lr_pdm_weight * geom_lr_pdm
        )

        with torch.no_grad():
            ppl = ppl_unmasked
            dead = dead_unmasked
            pred_labels = re_ss_logits.argmax(dim=-1)
            if mask is not None:
                correct = (pred_labels == gt_ss_labels) & mask
                ss_acc = correct.sum().float() / mask.sum().float().clamp_min(1.0)
            else:
                ss_acc = (pred_labels == gt_ss_labels).float().mean()

        out = {
            "loss": total_loss,
            "Reconstruction_Loss_XYZ": loss_xyz.detach(),
            "Reconstruction_Loss_SS": loss_ss.detach(),
            "SS_Accuracy": ss_acc.detach(),
            "VQ_Loss": vq_loss.detach(),
            "Geom_BondLength_Loss": bl.detach(),
            "Geom_BondAngle_Loss": ba.detach(),
            "Geom_Direction_Loss": dir_loss.detach(),
            "Geom_Dihedral_Loss": dih.detach(),
            "Geom_Loss": geom_loss.detach(),
            "SS_TV": ss_tv.detach(),
            "Usage_Reg": usage_reg.detach(),
            "XYZ_TV2": xyz_tv.detach(),
            "VQ_Perplexity": ppl.detach(),
            "VQ_DeadRatio": dead.detach(),
            "RMSD_Raw": rmsd_raw.detach(),
            "RMSD_Aligned": rmsd_aligned.detach(),
        }
        if pdm_weight > 0:
            out["Geom_LocalPDM"] = geom_local_pdm.detach()
        if win_kabsch_weight > 0:
            out["Geom_WinKabsch"] = geom_winkabsch.detach()
        if kappa_weight > 0:
            out["Frenet_Kappa"] = fr_kappa.detach()
        if tau_weight > 0:
            out["Frenet_Tau"] = fr_tau.detach()
        if lr_pdm_weight > 0:
            out["Geom_LongRangePDM"] = geom_lr_pdm.detach()
        return out

    @torch.no_grad()
    def generate(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs):
        return self.forward(x, mask=mask)[0]

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device, out_len: Optional[int] = None):
        N = int(self.latent_n_tokens)
        K = int(self.quantizer.K)
        idx = torch.randint(0, K, (num_samples, N), device=device)
        z_q = F.embedding(idx, self.quantizer.embedding)
        L = out_len if (out_len is not None) else self.max_seq_len
        mask = torch.ones(num_samples, L, dtype=torch.bool, device=device)
        out = self.decode(z_q, mask=mask)
        return out