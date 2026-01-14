# coding: utf-8

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Extract per-sample coefficients.

    a: [T]
    t: [B] int
    returns: [B, 1, 1,...] broadcastable to x_shape
    """
    if t.dim() != 1:
        raise ValueError(f"t must be 1D, got {t.shape}")
    out = a.gather(0, t.clamp(0, a.numel() - 1))
    while out.dim() < len(x_shape):
        out = out.unsqueeze(-1)
    return out


@dataclass
class GaussianDiffusion:
    betas: torch.Tensor

    def __post_init__(self):
        betas = self.betas
        betas = betas.float()
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.alphas_cumprod.device, dtype=self.alphas_cumprod.dtype),
            self.alphas_cumprod[:-1]],
            dim=0,
        )


        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def num_steps(self) -> int:
        return int(self.betas.numel())

    def to(self, device: torch.device) -> "GaussianDiffusion":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omab = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_omab * noise

    def predict_x0_from_eps(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, t, xt.shape)
        sqrt_omab = _extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        return (xt - sqrt_omab * eps) / torch.clamp(sqrt_ab, min=1e-12)

    @torch.no_grad()
    def ddim_step(
        self,
        model,
        xt: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DDIM step with eta=0.

        t and t_prev are [B] int tensors.
        """
        eps = model(xt, t, cond=cond, mask=mask)
        x0 = self.predict_x0_from_eps(xt, t, eps)

        ab_prev = _extract(self.alphas_cumprod, t_prev, xt.shape)
        omab_prev = torch.sqrt(1.0 - ab_prev)
        return torch.sqrt(ab_prev) * x0 + omab_prev * eps

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model,
        shape: Tuple[int, ...],
        steps: int,
        device: torch.device,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        steps = int(steps)
        if steps <= 1:
            raise ValueError("steps must be > 1")

        T = self.num_steps
        t_seq = torch.linspace(T - 1, 0, steps, device=device).long()
        x = torch.randn(shape, device=device)

        B = int(shape[0])
        for i in range(steps - 1):
            t = t_seq[i].expand(B)
            t_prev = t_seq[i + 1].expand(B)
            x = self.ddim_step(model, x, t=t, t_prev=t_prev, cond=cond, mask=mask)
            if mask is not None:
                x = x * mask[:, :, None].to(x.dtype)

        t = t_seq[-1].expand(B)
        eps = model(x, t, cond=cond, mask=mask)
        x0 = self.predict_x0_from_eps(x, t, eps)
        if mask is not None:
            x0 = x0 * mask[:, :, None].to(x0.dtype)
        return x0

    def training_loss(
        self,
        model,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)

        if mask is not None:
            noise = noise * mask[:, :, None].to(noise.dtype)

        xt = self.q_sample(x0, t, noise=noise)
        pred = model(xt, t, cond=cond, mask=mask)
        loss = F.mse_loss(pred, noise, reduction="none")
        if mask is not None:
            loss = loss * mask[:, :, None].to(loss.dtype)
            denom = torch.clamp(mask.sum() * x0.size(-1), min=1.0)
            return loss.sum() / denom
        return loss.mean()