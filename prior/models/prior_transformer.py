import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PriorModelConfig:
    vocab_size: int
    d_model: int = 768
    n_layers: int = 10
    n_heads: int = 8
    ffw_mult: int = 4
    dropout: float = 0.2
    max_code_len: int = 256
    num_quantizers: int = 4
    tie_embeddings: bool = True
    layer_norm_eps: float = 1e-5
    use_hierarchical_attn: bool = True
    pad_token_id: Optional[int] = None
    geo_dim: int = 0


class TransformerPriorLM(nn.Module):
    """
    Autoregressive Transformer LM over discrete VQ-VAE codes.

    Sequence layout:
        inp = [BOS] + codes (length = 1 + max_code_len)
        tgt = codes + [EOS] (same length)

    Causal training:
        Uses an upper-triangular attention mask so each position attends
        only to itself and previous positions.

    RVQ structure:
        For T time steps, positions 1..T-1 are interpreted as flattened
        RVQ positions with:
            level = (code_index % num_quantizers)
            pos_tok = (code_index // num_quantizers)
        BOS uses level id = num_quantizers.

    Optional hierarchical attention:
        For code positions (t >= 1), a lower level cannot attend to
        higher levels in the past. This encourages coarse-to-fine flow.

    Optional geometry head:
        If geo_dim > 0, a linear head maps hidden states to geometry
        descriptors of dimension geo_dim.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 10,
        n_heads: int = 8,
        ffw_mult: int = 4,
        dropout: float = 0.2,
        max_code_len: int = 256,
        num_quantizers: int = 4,
        tie_embeddings: bool = True,
        layer_norm_eps: float = 1e-5,
        use_hierarchical_attn: bool = True,
        pad_token_id: Optional[int] = None,
        geo_dim: int = 0,
    ):
        super().__init__()

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.max_code_len = int(max_code_len)
        self.max_seq_len = int(max_code_len) + 1
        self.num_quantizers = max(1, int(num_quantizers))
        self.use_hierarchical_attn = bool(use_hierarchical_attn)
        self.pad_token_id = pad_token_id
        self.geo_dim = int(geo_dim)

        self.token_embed = nn.Embedding(self.vocab_size, self.d_model)

        max_token_positions = int(math.ceil(self.max_code_len / float(self.num_quantizers)))
        self.pos_tok_embed = nn.Embedding(max_token_positions + 1, self.d_model)

        self.level_embed = nn.Embedding(self.num_quantizers + 1, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(ffw_mult * self.d_model),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=float(layer_norm_eps),
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=int(n_layers)
        )

        self.ln_f = nn.LayerNorm(self.d_model, eps=float(layer_norm_eps))
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.token_embed.weight

        self.geo_head: Optional[nn.Linear]
        if self.geo_dim > 0:
            self.geo_head = nn.Linear(self.d_model, self.geo_dim)
        else:
            self.geo_head = None

        self._mask_cache: Dict[Tuple[int, str, int, bool, int], torch.Tensor] = {}
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_tok_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.level_embed.weight, mean=0.0, std=0.02)
        if self.lm_head.weight is not self.token_embed.weight:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        if self.geo_head is not None:
            nn.init.normal_(self.geo_head.weight, mean=0.0, std=0.02)
            if self.geo_head.bias is not None:
                nn.init.zeros_(self.geo_head.bias)

    def _device_cache_key(self, T: int, device: torch.device) -> Tuple[int, str, int, bool, int]:
        dev_type = device.type
        dev_idx = int(device.index) if (dev_type == "cuda" and device.index is not None) else -1
        return (int(T), dev_type, dev_idx, bool(self.use_hierarchical_attn), int(self.num_quantizers))

    def _level_and_pos_ids(self, T: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            level: [T] long, 0..num_quantizers-1 for codes, num_quantizers for BOS
            pos_tok: [T] long, token position index for codes, 0 for BOS
        """
        if T <= 0:
            raise ValueError("T must be > 0")

        level = torch.full((T,), self.num_quantizers, dtype=torch.long, device=device)
        pos_tok = torch.zeros((T,), dtype=torch.long, device=device)

        if T > 1:
            code_idx = torch.arange(T - 1, device=device, dtype=torch.long)
            level[1:] = code_idx % self.num_quantizers
            pos_tok[1:] = code_idx // self.num_quantizers

        pos_tok = pos_tok.clamp_(0, self.pos_tok_embed.num_embeddings - 1)
        return level, pos_tok

    def _build_attn_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Returns a boolean mask [T, T] where True means "blocked".
        Combines causal mask and optional hierarchical level constraints.
        """
        key = self._device_cache_key(T, device)
        if key in self._mask_cache:
            return self._mask_cache[key]

        causal = torch.triu(
            torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1
        )

        if (not self.use_hierarchical_attn) or self.num_quantizers <= 1 or T <= 2:
            mask = causal
        else:
            lvl = torch.full((T,), -1, dtype=torch.long, device=device)
            if T > 1:
                lvl[1:] = torch.arange(T - 1, device=device, dtype=torch.long) % self.num_quantizers

            lvl_q = lvl[:, None]
            lvl_k = lvl[None, :]

            hier = (lvl_k > lvl_q) & (lvl_q >= 0) & (lvl_k >= 0)
            mask = causal | hier

        self._mask_cache[key] = mask
        return mask

    def forward_with_hidden(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns both logits and hidden states.

        Args:
            input_ids: LongTensor [B, T]
            attn_mask / attention_mask: Bool or LongTensor [B, T],
                1/True = valid, 0/False = padding

        Returns:
            logits: FloatTensor [B, T, vocab_size]
            hidden: FloatTensor [B, T, d_model]
        """
        if attention_mask is None:
            attention_mask = attn_mask

        B, T = input_ids.shape

        if T > self.max_seq_len:
            keep = self.max_seq_len
            if keep >= 2:
                input_ids = torch.cat(
                    [input_ids[:, :1], input_ids[:, -(keep - 1):]], dim=1
                )
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask[:, :1], attention_mask[:, -(keep - 1):]], dim=1
                    )
            else:
                input_ids = input_ids[:, -keep:]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, -keep:]
            T = input_ids.size(1)

        device = input_ids.device

        level_ids, pos_tok_ids = self._level_and_pos_ids(T, device)
        level_ids = level_ids.unsqueeze(0).expand(B, T)
        pos_tok_ids = pos_tok_ids.unsqueeze(0).expand(B, T)

        x = (
            self.token_embed(input_ids)
            + self.pos_tok_embed(pos_tok_ids)
            + self.level_embed(level_ids)
        )

        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask != 0
            src_key_padding_mask = ~attention_mask
        else:
            src_key_padding_mask = None

        attn = self._build_attn_mask(T, device)
        h = self.transformer(x, mask=attn, src_key_padding_mask=src_key_padding_mask)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits, h

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standard forward that returns only logits.
        """
        logits, _ = self.forward_with_hidden(
            input_ids=input_ids,
            attn_mask=attn_mask,
            attention_mask=attention_mask,
        )
        return logits

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """
        Cross-entropy loss with optional label smoothing.
        """
        ls = float(label_smoothing)
        logits_2d = logits.reshape(-1, logits.size(-1))
        targets_1d = targets.reshape(-1)

        if ls <= 0.0:
            return F.cross_entropy(
                logits_2d,
                targets_1d,
                ignore_index=int(ignore_index),
            )

        try:
            return F.cross_entropy(
                logits_2d,
                targets_1d,
                ignore_index=int(ignore_index),
                label_smoothing=ls,
            )
        except TypeError:
            log_probs = F.log_softmax(logits_2d, dim=-1)
            nll = -log_probs.gather(dim=-1, index=targets_1d.unsqueeze(-1)).squeeze(-1)
            smooth = -log_probs.mean(dim=-1)
            loss_vec = (1.0 - ls) * nll + ls * smooth
            mask = targets_1d.ne(int(ignore_index))
            loss_vec = loss_vec[mask]
            if loss_vec.numel() == 0:
                return loss_vec.new_tensor(0.0)
            return loss_vec.mean()

    @staticmethod
    def _top_k_top_p_filtering(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -1e10,
    ) -> torch.Tensor:
        """
        Top-k and/or nucleus (top-p) filtering on logits.

        Args:
            logits: [B, V]
        """
        if logits.dim() != 2:
            raise ValueError("logits must be [B, V]")

        B, V = logits.shape
        out = logits.clone()

        if top_k > 0:
            top_k = min(int(top_k), V)
            values, _ = torch.topk(out, top_k, dim=-1)
            min_values = values[:, -1].unsqueeze(-1)
            out = torch.where(
                out < min_values,
                torch.full_like(out, filter_value),
                out,
            )

        if 0.0 < float(top_p) < 1.0:
            sorted_logits, sorted_indices = torch.sort(out, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            mask = cumulative_probs > float(top_p)
            mask[:, 1:] = mask[:, :-1].clone()
            mask[:, 0] = False

            sorted_logits = torch.where(
                mask,
                torch.full_like(sorted_logits, filter_value),
                sorted_logits,
            )
            out = torch.full_like(out, filter_value)
            out.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

        return out


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -1e10,
) -> torch.Tensor:
    return TransformerPriorLM._top_k_top_p_filtering(
        logits, top_k=top_k, top_p=top_p, filter_value=filter_value
    )


def build_prior_from_config(cfg: Dict[str, Any]) -> TransformerPriorLM:
    """
    Convenience builder from a prior config dict.
    """
    vq_cfg = cfg.get("vq", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    data_cfg = cfg.get("data", {}) or {}

    code_vocab_k = int(vq_cfg.get("codebook_size", 0))
    if code_vocab_k <= 0:
        raise ValueError("cfg.vq.codebook_size must be set")

    vocab_size = int(code_vocab_k + 3)

    num_quantizers = int(model_cfg.get("num_quantizers", vq_cfg.get("num_quantizers", 4)))
    max_code_len = int(data_cfg.get("max_len", model_cfg.get("max_code_len", 256)))
    geo_dim = int(model_cfg.get("geo_dim", 0))

    return TransformerPriorLM(
        vocab_size=vocab_size,
        d_model=int(model_cfg.get("d_model", 768)),
        n_layers=int(model_cfg.get("n_layers", 10)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        ffw_mult=int(model_cfg.get("ffw_mult", 4)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        max_code_len=max_code_len,
        num_quantizers=num_quantizers,
        tie_embeddings=bool(model_cfg.get("tie_embeddings", True)),
        layer_norm_eps=float(model_cfg.get("layer_norm_eps", 1e-5)),
        use_hierarchical_attn=bool(model_cfg.get("use_hierarchical_attn", True)),
        pad_token_id=model_cfg.get("pad_token_id", None),
        geo_dim=geo_dim,
    )
