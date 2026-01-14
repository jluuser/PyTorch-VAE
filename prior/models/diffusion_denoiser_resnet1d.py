# coding: utf-8

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding.

    t: [B] int/float
    returns: [B, dim]
    """
    if t.dim() != 1:
        t = t.view(-1)
    device = t.device
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device).float() / max(half - 1, 1))
    ang = t.float().unsqueeze(1) * freq.unsqueeze(0)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        time_dim: int,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.time_proj = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.drop = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return x + h


@dataclass
class DenoiserConfig:
    code_dim: int
    hidden_channels: int = 256
    time_embed_dim: int = 256
    num_blocks: int = 12
    dropout: float = 0.0
    cond_dim: int = 0


class DiffusionDenoiserResNet1D(nn.Module):
    """1D residual conv denoiser for token sequences.

    Inputs/outputs use token-major layout: [B, M, D]. Internally uses Conv1d with
    channel-major layout: [B, C, M].
    """

    def __init__(self, cfg: DenoiserConfig):
        super().__init__()
        self.cfg = cfg
        D = int(cfg.code_dim)
        C = int(cfg.hidden_channels)
        T = int(cfg.time_embed_dim)

        self.in_proj = nn.Conv1d(D, C, kernel_size=1)
        self.out_proj = nn.Conv1d(C, D, kernel_size=1)

        self.time_mlp = nn.Sequential(
            nn.Linear(T, T * 4),
            nn.SiLU(),
            nn.Linear(T * 4, T),
        )

        self.cond_proj: Optional[nn.Linear] = None
        if int(cfg.cond_dim) > 0:
            self.cond_proj = nn.Linear(int(cfg.cond_dim), C)

        blocks = []
        dilations = [2 ** (i % 6) for i in range(int(cfg.num_blocks))]
        for d in dilations:
            blocks.append(ResBlock1D(C, time_dim=T, dilation=int(d), dropout=float(cfg.dropout)))
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if xt.dim() != 3:
            raise ValueError(f"xt must be [B,M,D], got {xt.shape}")
        B, M, D = xt.shape

        if mask is not None:
            if mask.shape != (B, M):
                raise ValueError(f"mask must be [B,M]={B,M}, got {mask.shape}")

        t_emb = sinusoidal_time_embedding(t, self.cfg.time_embed_dim)
        t_emb = self.time_mlp(t_emb)

        x = xt
        if mask is not None:
            x = x * mask[:, :, None].to(x.dtype)

        h = self.in_proj(x.transpose(1, 2))  # [B,C,M]

        if cond is not None and self.cond_proj is not None:
            if cond.shape[:2] != (B, M):
                raise ValueError(f"cond must be [B,M,*], got {cond.shape}")
            c = self.cond_proj(cond).transpose(1, 2)
            h = h + c

        for blk in self.blocks:
            h = blk(h, t_emb)

        out = self.out_proj(h).transpose(1, 2)  # [B,M,D]
        if mask is not None:
            out = out * mask[:, :, None].to(out.dtype)
        return out