import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from .types_ import Tensor


# ----------------------- Quantizers ----------------------- #

class VectorQuantizerEMA(nn.Module):
    """
    EMA codebook VQ with straight-through estimator.
    Returns z_q_st (with STE), z_q_raw, indices, and stats (perplexity, dead_ratio).
    Includes optional dead-code recycling to mitigate collapse.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        reinit_dead_codes: bool = True,
        reinit_prob: float = 1.0,
        dead_usage_threshold: int = 0,
    ):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.use_ema = True

        # dead-code recycle knobs
        self.reinit_dead_codes = bool(reinit_dead_codes)
        self.reinit_prob = float(reinit_prob)
        self.dead_usage_threshold = int(dead_usage_threshold)

        embedding = torch.randn(self.K, self.D)
        embedding = embedding / embedding.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("embedding", embedding)                # (K, D)
        self.register_buffer("ema_cluster_size", torch.zeros(self.K))
        self.register_buffer("ema_embedding", torch.zeros(self.K, self.D))

        # one-time info
        print(f"[VQ] Using VectorQuantizerEMA: K={self.K}, D={self.D}, beta={self.beta}, "
              f"decay={self.decay}, reinit_dead_codes={self.reinit_dead_codes}")

    @torch.no_grad()
    def _ema_update(self, flat_norm: Tensor, indices: Tensor):
        # flat_norm: (B*L, D), indices: (B*L,)
        one_hot = torch.zeros(indices.shape[0], self.K, device=flat_norm.device, dtype=flat_norm.dtype)
        one_hot.scatter_(1, indices.unsqueeze(1), 1.0)
        cluster_size = one_hot.sum(dim=0)                           # (K,)
        embed_sum = flat_norm.t() @ one_hot                         # (D, K)
        embed_sum = embed_sum.t()                                   # (K, D)

        self.ema_cluster_size.mul_(self.decay).add_((1 - self.decay) * cluster_size)
        self.ema_embedding.mul_(self.decay).add_((1 - self.decay) * embed_sum)

        n = self.ema_cluster_size.sum()
        # Laplace smoothing to avoid empty clusters
        smoothed = (self.ema_cluster_size + self.eps) / (n + self.K * self.eps)
        normalized_embed = self.ema_embedding / smoothed.unsqueeze(1).clamp_min(self.eps)
        # L2 normalize embeddings for stability
        normalized_embed = normalized_embed / normalized_embed.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.embedding.copy_(normalized_embed)

    @torch.no_grad()
    def _maybe_reinit_dead_codes(self, flat_norm: Tensor, usage: Tensor):
        """
        Reinitialize codes whose usage <= dead_usage_threshold in this batch.
        """
        if not self.reinit_dead_codes:
            return
        if self.reinit_prob <= 0.0:
            return
        if usage.dtype != torch.float32:
            usage = usage.float()

        dead_mask = usage <= float(self.dead_usage_threshold)
        num_dead = int(dead_mask.sum().item())
        if num_dead <= 0:
            return

        # With probability reinit_prob, refresh these codes from random normalized encoder vectors.
        if torch.rand(()) > self.reinit_prob:
            return

        num_src = flat_norm.size(0)
        if num_src == 0:
            return
        rand_idx = torch.randint(0, num_src, (num_dead,), device=flat_norm.device)
        src = flat_norm[rand_idx]  # (num_dead, D)
        src = src / src.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.embedding[dead_mask] = src

    def forward(self, z_e: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            z_e: (B, L, D)
        Returns:
            z_q_st: (B, L, D) quantized with STE
            z_q_raw: (B, L, D) raw quantized vectors
            indices: (B, L) selected code indices
            stats: (perplexity, dead_ratio) packed in a (2,) tensor
        """
        B, L, D = z_e.shape
        flat = z_e.reshape(-1, D)  # (B*L, D)

        # Normalize encoder outputs to reduce scale mismatch
        flat_norm = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-8)

        # Cosine distance (equivalently, maximize dot-product since both are normalized).
        # argmin(1 - dot) equals argmax(dot)
        dots = torch.matmul(flat_norm, self.embedding.t())  # (B*L, K)
        indices = torch.argmax(dots, dim=1)                 # (B*L,)

        z_q = F.embedding(indices, self.embedding)          # (B*L, D)
        z_q = z_q.view(B, L, D)

        # EMA update (train mode)
        if self.training:
            self._ema_update(flat_norm.detach(), indices.detach())

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Usage stats
        with torch.no_grad():
            usage = torch.bincount(indices, minlength=self.K).float()  # (K,)
            probs = usage / usage.sum().clamp_min(1.0)
            nz = probs > 0
            perplexity = torch.exp(-(probs[nz] * probs[nz].log()).sum())
            dead_ratio = (usage == 0).float().mean()

            # Optional: recycle dead/low-usage codes using current batch encoder vectors
            if self.training:
                self._maybe_reinit_dead_codes(flat_norm.detach(), usage)

        stats = torch.stack([perplexity, dead_ratio])
        return z_q_st, z_q, indices.view(B, L), stats


# ----------------------- Model ----------------------- #

class VQVAE(nn.Module):
    """
    VQ-VAE with Transformer encoder/decoder.
    - EMA VQ to prevent codebook collapse; optional dead-code recycling.
    - Decoder queries from learned query embeddings (Q != memory).
    - Optional probabilistic history drop for self-attention (with annealing).
    - Split heads for xyz and ss.
    - SS uses label smoothing and temporal smoothing regularizer.
    - Geometry losses can operate in real scale via set_data_stats.
    """
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 350,
        codebook_size: int = 1024,
        code_dim: int = 64,
        beta: float = 0.35,
        history_drop_prob: float = 0.0,
        # knobs
        zero_ss_in_encoder: bool = True,
        label_smoothing: float = 0.05,
        ss_tv_lambda: float = 0.03,
        usage_entropy_lambda: float = 3e-4,
        # quantizer recycling knobs (mirrored to EMA class)
        reinit_dead_codes: bool = True,
        reinit_prob: float = 1.0,
        dead_usage_threshold: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.max_seq_len = max_seq_len
        self.history_drop_prob = history_drop_prob

        # training knobs
        self.zero_ss_in_encoder = bool(zero_ss_in_encoder)
        self.label_smoothing = float(label_smoothing)
        self.ss_tv_lambda = float(ss_tv_lambda)
        self.usage_entropy_lambda = float(usage_entropy_lambda)

        # Optional data stats for geometry in real scale
        self._data_mean = None  # (1,1,3)
        self._data_std = None   # (1,1,3)

        # Input projection + small dropout
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.inp_dropout = nn.Dropout(p=0.1)

        # Encoder with dropout
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # To code
        self.to_code = nn.Linear(hidden_dim, code_dim)

        # Quantizer (EMA)
        self.quantizer = VectorQuantizerEMA(
            num_embeddings=codebook_size,
            embedding_dim=code_dim,
            beta=beta,
            decay=0.99,
            eps=1e-5,
            reinit_dead_codes=reinit_dead_codes,
            reinit_prob=reinit_prob,
            dead_usage_threshold=dead_usage_threshold,
        )

        # From code (latent memory)
        self.from_code = nn.Linear(code_dim, hidden_dim)

        # Decoder with dropout
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # Positional encoding
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_enc", pe.unsqueeze(0))  # (1, L, H)

        # Learned queries (Q != memory)
        self.query_embed = nn.Embedding(max_seq_len, hidden_dim)
        nn.init.normal_(self.query_embed.weight, std=0.02)

        # Split heads
        self.head_xyz = nn.Linear(hidden_dim, 3)
        self.head_ss = nn.Linear(hidden_dim, 3)

        # step counter for annealing
        self.training_steps = 0

    # ---- Optional helper to set data stats ----
    @torch.no_grad()
    def set_data_stats(self, mean_xyz: torch.Tensor, std_xyz: torch.Tensor):
        device = self.head_xyz.weight.device
        mean = torch.as_tensor(mean_xyz, dtype=torch.float32, device=device).view(1, 1, 3)
        std = torch.as_tensor(std_xyz, dtype=torch.float32, device=device).view(1, 1, 3)
        self._data_mean = mean
        self._data_std = std

    # ----------------------------------------------------

    def encode(self, x: Tensor, mask: Optional[Tensor] = None):
        # Optionally zero the SS channels to avoid leaking ground-truth SS into the encoder
        if self.zero_ss_in_encoder:
            x = torch.cat([x[..., :3], torch.zeros_like(x[..., 3:])], dim=-1)

        B, L, _ = x.shape
        h = self.input_proj(x)
        h = self.inp_dropout(h) + self.pos_enc[:, :L, :]
        h = self.encoder(h, src_key_padding_mask=(~mask) if mask is not None else None)
        z_e = self.to_code(h)  # (B, L, code_dim)
        return z_e

    def _build_self_attn_mask(self, L: int, drop_history: bool, device) -> Optional[Tensor]:
        if not drop_history:
            return None
        m = torch.full((L, L), float("-inf"), device=device)
        m.fill_diagonal_(0.0)  # only self
        return m

    def decode(self, z_q: Tensor, mask: Optional[Tensor] = None, drop_history: bool = False):
        B, L, _ = z_q.shape
        memory = self.from_code(z_q)  # (B, L, H)
        q = self.query_embed.weight[:L].unsqueeze(0).expand(B, L, -1)
        q = q + self.pos_enc[:, :L, :]

        tgt_key_padding_mask = (~mask) if mask is not None else None
        mem_key_padding_mask = (~mask) if mask is not None else None
        self_attn_mask = self._build_self_attn_mask(L, drop_history, memory.device)

        h_dec = self.decoder(
            tgt=q,
            memory=memory,
            tgt_mask=self_attn_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=mem_key_padding_mask,
        )
        xyz_pred = self.head_xyz(h_dec)
        ss_logits = self.head_ss(h_dec)
        recons = torch.cat([xyz_pred, ss_logits], dim=-1)
        return recons

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> List[Tensor]:
        # small Gaussian noise on xyz during training (data augmentation)
        if self.training:
            noise = torch.randn_like(x[..., :3]) * 0.01
            x = torch.cat([x[..., :3] + noise, x[..., 3:]], dim=-1)

        z_e = self.encode(x, mask=mask)
        z_q_st, z_q_raw, indices, stats = self.quantizer(z_e)

        # annealed history drop: from 0.5 -> 0.1
        drop = False
        if self.history_drop_prob > 0.0 and self.training:
            steps = getattr(self, "training_steps", 0)
            max_steps = 100000
            t = min(1.0, steps / max_steps)
            p = 0.5 * (1.0 - t) + 0.1 * t
            drop = torch.rand(()) < p
            self.training_steps = steps + 1

        recons = self.decode(z_q_st, mask=mask, drop_history=bool(drop))

        # vq_pack: (z_q_raw, z_e_raw, indices, perplexity, dead_ratio)
        vq_pack = (z_q_raw, z_e, indices, stats[0], stats[1])
        return [recons, x, vq_pack, mask]

    @staticmethod
    def _masked_mse(a: Tensor, b: Tensor, mask: Optional[Tensor]):
        diff = (a - b) ** 2
        if mask is None:
            return diff.mean()
        m = mask.unsqueeze(-1).float()
        return (diff * m).sum() / m.sum().clamp_min(1.0)

    def _ss_label_smoothing_ce(self, logits: Tensor, labels: Tensor, mask: Optional[Tensor], eps: float):
        """
        Label-smoothed CE implemented via KLDivLoss with log_softmax.
        logits: (B, L, C), labels: (B, L) long, mask: (B, L) bool
        """
        B, L, C = logits.shape
        logp = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(eps / (C - 1))
            true_dist.scatter_(-1, labels.unsqueeze(-1), 1.0 - eps)
        kl = F.kl_div(logp, true_dist, reduction="none").sum(dim=-1)  # (B, L)
        if mask is not None:
            m = mask.float()
            return (kl * m).sum() / m.sum().clamp_min(1.0)
        return kl.mean()

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Expects: recons, target, vq_pack, mask
        vq_pack = (z_q_raw, z_e_raw, indices, ppl_unmasked, dead_unmasked)
        """
        recons = args[0]
        target = args[1]
        vq_pack = args[2]
        mask = args[3] if len(args) > 3 else None

        zq_raw, ze_raw, indices, ppl_unmasked, dead_unmasked = vq_pack

        ss_weight = kwargs.get("ss_weight", 1.0)
        bond_len_w = kwargs.get("bond_length_weight", 0.0)
        bond_ang_w = kwargs.get("bond_angle_weight", 0.0)
        xyz_tv_lambda = kwargs.get("xyz_tv_lambda", 0.02)  # new: coordinate smoothness

        re_xyz = recons[:, :, :3]
        re_ss_logits = recons[:, :, 3:]
        gt_xyz = target[:, :, :3]
        gt_ss_onehot = target[:, :, 3:]

        # ---------------- Reconstruction: XYZ ----------------
        loss_xyz = self._masked_mse(re_xyz, gt_xyz, mask)

        # ---------------- Reconstruction: SS ----------------
        gt_ss_labels = gt_ss_onehot.argmax(dim=-1)
        if self.label_smoothing and self.label_smoothing > 0.0:
            loss_ss = self._ss_label_smoothing_ce(
                re_ss_logits, gt_ss_labels, mask, eps=self.label_smoothing
            )
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
                    gt_ss_labels.reshape(-1),
                )

        # Temporal smoothing (TV) on SS probabilities
        if self.ss_tv_lambda > 0.0:
            p = F.softmax(re_ss_logits, dim=-1)
            if p.size(1) >= 2:
                tv = (p[:, 1:, :] - p[:, :-1, :]).abs().sum(dim=-1)  # (B, L-1)
                if mask is not None:
                    tv_mask = (mask[:, 1:] & mask[:, :-1]).float()
                    ss_tv = (tv * tv_mask).sum() / tv_mask.sum().clamp_min(1.0)
                else:
                    ss_tv = tv.mean()
            else:
                ss_tv = torch.tensor(0.0, device=recons.device)
        else:
            ss_tv = torch.tensor(0.0, device=recons.device)

        # ---------------- Geometry losses ----------------
        std = self._data_std if self._data_std is not None else None
        mean = self._data_mean if self._data_mean is not None else None

        def to_real(x):
            if std is not None:
                return x * std + (mean if mean is not None else 0.0)
            return x

        re_xyz_real = to_real(re_xyz)
        gt_xyz_real = to_real(gt_xyz)

        # Bond length
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

        # Bond angle
        if re_xyz_real.size(1) >= 3:
            v1_rec = re_xyz_real[:, 1:-1, :] - re_xyz_real[:, :-2, :]
            v2_rec = re_xyz_real[:, 2:, :] - re_xyz_real[:, 1:-1, :]
            v1_gt = gt_xyz_real[:, 1:-1, :] - gt_xyz_real[:, :-2, :]
            v2_gt = gt_xyz_real[:, 2:, :] - gt_xyz_real[:, 1:-1, :]

            def _cos(v1, v2, eps=1e-8):
                v1n = v1 / (v1.norm(dim=-1, keepdim=True) + eps)
                v2n = v2 / (v2.norm(dim=-1, keepdim=True) + eps)
                return (v1n * v2n).sum(dim=-1)

            cos_rec = _cos(v1_rec, v2_rec)
            cos_gt = _cos(v1_gt, v2_gt)
            if mask is not None:
                tri_mask = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).float()
                ba = ((cos_rec - cos_gt) ** 2 * tri_mask).sum() / tri_mask.sum().clamp_min(1.0)
            else:
                ba = F.mse_loss(cos_rec, cos_gt)
        else:
            ba = torch.tensor(0.0, device=recons.device)

        geom_loss = bond_len_w * bl + bond_ang_w * ba

        # ---------------- VQ loss (masked) ----------------
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            commit = ((zq_raw.detach() - ze_raw) ** 2 * m).sum() / m.sum().clamp_min(1.0)
            embed = ((zq_raw - ze_raw.detach()) ** 2 * m).sum() / m.sum().clamp_min(1.0)
        else:
            commit = F.mse_loss(zq_raw.detach(), ze_raw)
            embed = F.mse_loss(zq_raw, ze_raw.detach())

        if getattr(self.quantizer, "use_ema", False):
            vq_loss = self.quantizer.beta * commit  # EMA variant: commitment only
        else:
            vq_loss = self.quantizer.beta * commit + embed

        # ---------------- Usage entropy regularizer ----------------
        if self.usage_entropy_lambda > 0.0:
            with torch.no_grad():
                if mask is not None:
                    idx_mask = mask.reshape(-1)  # (B*L,)
                    idx_flat = indices.reshape(-1)
                    idx_flat = idx_flat[idx_mask]
                    usage = torch.bincount(idx_flat, minlength=self.quantizer.K).float()
                else:
                    usage = torch.bincount(indices.reshape(-1), minlength=self.quantizer.K).float()
                probs = usage / usage.sum().clamp_min(1.0)
                nz = probs > 0
                entropy = -(probs[nz] * probs[nz].log()).sum()
            usage_reg = - self.usage_entropy_lambda * entropy
        else:
            usage_reg = torch.tensor(0.0, device=recons.device)

        # ---------------- XYZ second-difference smoothness ----------------
        if xyz_tv_lambda > 0.0 and re_xyz.size(1) >= 3:
            d1 = re_xyz[:, 1:, :] - re_xyz[:, :-1, :]
            d2 = d1[:, 1:, :] - d1[:, :-1, :]
            tv2 = (d2 ** 2).sum(dim=-1)  # (B, L-2)
            if mask is not None:
                tv2_mask = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).float()
                xyz_tv = (tv2 * tv2_mask).sum() / tv2_mask.sum().clamp_min(1.0)
            else:
                xyz_tv = tv2.mean()
        else:
            xyz_tv = torch.tensor(0.0, device=recons.device)

        # ---------------- Total ----------------
        total_loss = (
            loss_xyz
            + ss_weight * loss_ss
            + vq_loss
            + geom_loss
            + self.ss_tv_lambda * ss_tv
            + usage_reg
            + xyz_tv_lambda * xyz_tv
        )

        # Prefer masked stats if mask is given
        with torch.no_grad():
            if mask is not None:
                idx_mask = mask.reshape(-1)
                idx_flat = indices.reshape(-1)
                idx_flat = idx_flat[idx_mask]
                usage = torch.bincount(idx_flat, minlength=self.quantizer.K).float()
                probs = usage / usage.sum().clamp_min(1.0)
                nz = probs > 0
                ppl = torch.exp(-(probs[nz] * probs[nz].log()).sum())
                dead = (usage == 0).float().mean()
            else:
                ppl = ppl_unmasked
                dead = dead_unmasked

        return {
            "loss": total_loss,
            "Reconstruction_Loss_XYZ": loss_xyz.detach(),
            "Reconstruction_Loss_SS": loss_ss.detach(),
            "VQ_Loss": vq_loss.detach(),
            "Geom_BondLength_Loss": bl.detach(),
            "Geom_BondAngle_Loss": ba.detach(),
            "Geom_Loss": geom_loss.detach(),
            "SS_TV": ss_tv.detach(),
            "Usage_Reg": usage_reg.detach(),
            "XYZ_TV2": xyz_tv.detach(),
            "VQ_Perplexity": ppl.detach(),
            "VQ_DeadRatio": dead.detach(),
        }

    @torch.no_grad()
    def generate(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs):
        return self.forward(x, mask=mask)[0]

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device):
        """
        Random sampling by drawing code indices uniformly from the codebook.
        This is only a smoke test and NOT a trained index prior.
        """
        L = self.max_seq_len
        idx = torch.randint(0, self.quantizer.K, (num_samples, L), device=device)
        z_q = F.embedding(idx, self.quantizer.embedding)
        out = self.decode(z_q, mask=None, drop_history=False)
        return out
