# prior/models/prior_transformer.py
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerPriorLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        ffw_mult: int = 4,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
        layer_norm_eps: float = 1e-5,
        pad_token_id: Optional[int] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(4096, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffw_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, inp: torch.Tensor, attn_mask: torch.Tensor):
        # inp: [B, T] long
        B, T = inp.shape
        pos = torch.arange(T, device=inp.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(inp) + self.pos_emb(pos)

        # causal mask
        # shape [T, T], True means blocked
        causal = torch.triu(torch.ones((T, T), device=inp.device, dtype=torch.bool), diagonal=1)

        # key_padding_mask expects True for padding positions to be masked
        key_padding_mask = ~attn_mask  # invert: True for pads

        h = self.transformer(
            x,
            mask=causal,
            src_key_padding_mask=key_padding_mask
        )
        h = self.norm(h)
        logits = self.lm_head(h)  # [B, T, vocab]

        return logits

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int):
        # logits: [B, T, V], targets: [B, T]
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=ignore_index
        )

    @torch.no_grad()
    def generate(self, max_len: int, bos_id: int, eos_id: int, pad_id: int,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                 device: str = "cuda"):
        tokens = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        attn = torch.ones_like(tokens, dtype=torch.bool)

        for _ in range(max_len):
            logits = self.forward(tokens, attn_mask=attn)[:, -1, :] / max(temperature, 1e-8)
            probs = torch.softmax(self._top_k_top_p_filtering(logits, top_k, top_p), dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # [1,1]
            tokens = torch.cat([tokens, next_tok], dim=1)
            attn = torch.ones_like(tokens, dtype=torch.bool)
            if next_tok.item() == eos_id:
                break
        return tokens.squeeze(0)  # [T_gen]

    @staticmethod
    def _top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
        # logits: [B, V]
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            min_values = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_values, torch.full_like(logits, filter_value), logits)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumprobs > top_p
            mask[..., 0] = False
            sorted_logits = torch.where(mask, torch.full_like(sorted_logits, filter_value), sorted_logits)
            logits = torch.full_like(logits, filter_value).scatter(-1, sorted_indices, sorted_logits)
        return logits