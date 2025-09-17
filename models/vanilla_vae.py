import math
import torch
from torch import nn
from torch.nn import functional as F
from .types_ import Tensor
from typing import List, Optional
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder


class VanillaVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        latent_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 512,
        kld_weight: float = 1.0,
        prior_mem_tokens: int = 8,
        p_memdrop: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight
        self.max_seq_len = max_seq_len

        # memory dropout and prior memory
        self.p_memdrop = float(p_memdrop)
        self.prior_mem_tokens = int(prior_mem_tokens)

        # input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # map pooled encoding to mu and logvar (global latent)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # latent to decoder query embedding (H)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        # transformer decoder with cross-attention
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

        # prior memory generator: z -> K tokens in H
        self.prior_mem = nn.Linear(latent_dim, hidden_dim * self.prior_mem_tokens)

        # positional encoding
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_enc", pe.unsqueeze(0))  # (1, L, H)

        # output projection back to input_dim
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def encode(self, input: Tensor, mask: Optional[Tensor] = None) -> List[Tensor]:
        # input: (B, L, input_dim)
        B, L, _ = input.shape
        x = self.input_proj(input) + self.pos_enc[:, :L, :]

        enc_out = self.encoder(
            x,
            src_key_padding_mask=(~mask) if mask is not None else None
        )  # (B, L, H)

        # masked mean pooling -> global representation
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()  # (B, L, 1)
            lengths = mask_f.sum(dim=1).clamp_min(1.0)  # (B, 1)
            pooled = (enc_out * mask_f).sum(dim=1) / lengths  # (B, H)
        else:
            pooled = enc_out.mean(dim=1)

        mu = self.fc_mu(pooled)      # (B, D)
        logvar = self.fc_var(pooled) # (B, D)
        logvar = torch.clamp(logvar, min=-10, max=10)

        return [mu, logvar, enc_out, mask]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # (B, D)

    def _build_memory(self, z: Tensor, enc_out: Optional[Tensor], mask: Optional[Tensor]) -> (Tensor, Optional[Tensor]):
        """
        Build memory for the decoder. If training and memory-dropout triggers,
        or enc_out is None, fall back to prior memory generated from z.
        Returns memory (B, M, H) and its key_padding_mask (B, M) or None.
        """
        B = z.size(0)
        H = self.output_proj.in_features  # hidden_dim

        # prior memory from z: (B, K, H)
        prior = self.prior_mem(z).view(B, self.prior_mem_tokens, H)

        use_enc = enc_out is not None
        if self.training and self.p_memdrop > 0.0:
            # stochastic gate
            if torch.rand(1, device=z.device).item() < self.p_memdrop:
                use_enc = False

        if use_enc:
            # concatenate encoder tokens with prior memory tokens
            memory = torch.cat([enc_out, prior], dim=1)  # (B, L+K, H)
            if mask is not None:
                kpm_left = (~mask)  # (B, L)
                kpm_right = torch.zeros(B, self.prior_mem_tokens, dtype=torch.bool, device=mask.device)  # K valid
                memory_kpm = torch.cat([kpm_left, kpm_right], dim=1)  # (B, L+K)
            else:
                memory_kpm = None
        else:
            memory = prior  # (B, K, H)
            memory_kpm = torch.zeros(B, self.prior_mem_tokens, dtype=torch.bool, device=z.device)

        return memory, memory_kpm

    def decode(
        self,
        z: Tensor,
        enc_out: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None
    ) -> Tensor:
        """
        z: (B, D) global latent for queries
        enc_out: (B, L, H) encoder tokens as optional memory
        """
        if z is None and enc_out is None:
            raise ValueError("At least one of z or enc_out must be provided.")

        B = z.size(0) if z is not None else enc_out.size(0)
        L = lengths.max().item() if lengths is not None else self.max_seq_len

        # queries from z
        if z is None:
            # fallback: zeros queries (should not happen in normal training)
            H = self.output_proj.in_features
            tgt = torch.zeros(B, L, H, device=enc_out.device)
        else:
            H = self.output_proj.in_features
            q = self.decoder_input(z)              # (B, H)
            tgt = q.unsqueeze(1).expand(-1, L, -1) # (B, L, H)
            tgt = tgt + self.pos_enc[:, :L, :]

        # key padding mask over target length
        if lengths is not None:
            idx = torch.arange(L, device=tgt.device).unsqueeze(0)  # (1, L)
            tgt_key_padding_mask = idx >= lengths.unsqueeze(1)     # (B, L)
        else:
            tgt_key_padding_mask = None

        # build memory
        memory, memory_kpm = self._build_memory(z if z is not None else enc_out.mean(dim=1), enc_out, mask)

        dec_out = self.decoder(
            tgt, memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_kpm
        )  # (B, L, H)

        return self.output_proj(dec_out)  # (B, L, input_dim)

    def forward(self, input: Tensor, mask: Optional[Tensor] = None, **kwargs) -> List[Tensor]:
        mu, logvar, enc_out, mask = self.encode(input, mask=mask)
        z = self.reparameterize(mu, logvar)
        lengths = mask.sum(dim=1) if mask is not None else None
        recons = self.decode(z, enc_out=enc_out, mask=mask, lengths=lengths)
        return [recons, input, mu, logvar, mask]

    @staticmethod
    def _masked_mse(a: Tensor, b: Tensor, mask: Tensor) -> Tensor:
        diff = (a - b) ** 2
        if mask is None:
            return diff.mean()
        # mask over last dim broadcast
        m = mask.unsqueeze(-1).float()
        num = (diff * m).sum()
        den = m.sum().clamp_min(1.0)
        return num / den

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]    # (B, L, 6)
        input_ = args[1]    # (B, L, 6)
        mu, logvar = args[2], args[3]  # (B, D)
        mask = args[4] if len(args) > 4 else kwargs.get("mask", None)

        kld_weight = kwargs.get("M_N", 1.0)
        ss_weight = kwargs.get("ss_weight", 1.0)
        bond_len_w = kwargs.get("bond_length_weight", 0.0)
        bond_ang_w = kwargs.get("bond_angle_weight", 0.0)
        free_bits_nats = kwargs.get("free_bits_nats", 0.0)

        # reconstruction (masked)
        re_xyz = recons[:, :, :3]
        re_ss_logits = recons[:, :, 3:]
        gt_xyz = input_[:, :, :3]
        gt_ss_onehot = input_[:, :, 3:]

        # xyz MSE
        loss_xyz = self._masked_mse(re_xyz, gt_xyz, mask if mask is not None else None)

        # ss CE
        gt_ss_labels = gt_ss_onehot.argmax(dim=-1)  # (B, L)
        if mask is not None:
            # compute CE with masking: reduce='sum' then divide by valid tokens
            ce = F.cross_entropy(
                re_ss_logits.reshape(-1, re_ss_logits.size(-1)),
                gt_ss_labels.reshape(-1),
                reduction="none"
            ).reshape(gt_ss_labels.shape)
            m = mask.float()
            loss_ss = (ce * m).sum() / m.sum().clamp_min(1.0)
        else:
            loss_ss = F.cross_entropy(re_ss_logits.reshape(-1, re_ss_logits.size(-1)),
                                      gt_ss_labels.reshape(-1),
                                      reduction="mean")

        # KL (global latent)
        kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B,)
        if free_bits_nats > 0.0:
            kld_per_sample = torch.relu(kld_per_sample - free_bits_nats)
        kld_loss = kld_per_sample.mean()

        # geometric constraints (masked, computed on (B, L, 3))
        # bond length (adjacent)
        if re_xyz.size(1) >= 2:
            re_diff = re_xyz[:, 1:, :] - re_xyz[:, :-1, :]
            gt_diff = gt_xyz[:, 1:, :] - gt_xyz[:, :-1, :]
            re_len = torch.norm(re_diff, dim=-1)  # (B, L-1)
            gt_len = torch.norm(gt_diff, dim=-1)  # (B, L-1)
            if mask is not None:
                pair_mask = (mask[:, 1:] & mask[:, :-1]).float()
                bl_num = ((re_len - gt_len) ** 2 * pair_mask).sum()
                bl_den = pair_mask.sum().clamp_min(1.0)
                bond_length_loss = bl_num / bl_den
            else:
                bond_length_loss = F.mse_loss(re_len, gt_len, reduction="mean")
        else:
            bond_length_loss = torch.tensor(0.0, device=recons.device)

        # bond angle (triplets): compare cosine of angle between consecutive segments
        if re_xyz.size(1) >= 3:
            v1_rec = re_xyz[:, 1:-1, :] - re_xyz[:, :-2, :]
            v2_rec = re_xyz[:, 2:, :] - re_xyz[:, 1:-1, :]
            v1_gt = gt_xyz[:, 1:-1, :] - gt_xyz[:, :-2, :]
            v2_gt = gt_xyz[:, 2:, :] - gt_xyz[:, 1:-1, :]

            def _cos(v1, v2, eps=1e-8):
                v1n = v1 / (v1.norm(dim=-1, keepdim=True) + eps)
                v2n = v2 / (v2.norm(dim=-1, keepdim=True) + eps)
                return (v1n * v2n).sum(dim=-1)

            cos_rec = _cos(v1_rec, v2_rec)  # (B, L-2)
            cos_gt = _cos(v1_gt, v2_gt)    # (B, L-2)

            if mask is not None:
                tri_mask = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).float()
                ba_num = ((cos_rec - cos_gt) ** 2 * tri_mask).sum()
                ba_den = tri_mask.sum().clamp_min(1.0)
                bond_angle_loss = ba_num / ba_den
            else:
                bond_angle_loss = F.mse_loss(cos_rec, cos_gt, reduction="mean")
        else:
            bond_angle_loss = torch.tensor(0.0, device=recons.device)

        geom_loss = bond_len_w * bond_length_loss + bond_ang_w * bond_angle_loss

        total_loss = loss_xyz + ss_weight * loss_ss + kld_weight * kld_loss + geom_loss

        return {
            "loss": total_loss,
            "Reconstruction_Loss_XYZ": loss_xyz.detach(),
            "Reconstruction_Loss_SS": loss_ss.detach(),
            "KLD": kld_loss.detach(),
            "Geom_BondLength_Loss": bond_length_loss.detach(),
            "Geom_BondAngle_Loss": bond_angle_loss.detach(),
            "Geom_Loss": geom_loss.detach(),
        }

    def generate(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        return self.forward(x, mask=mask, **kwargs)[0]
