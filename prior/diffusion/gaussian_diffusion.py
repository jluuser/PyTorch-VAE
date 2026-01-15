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
    """DDPM-style Gaussian diffusion with x-prediction parameterization.

    - Forward process (q_sample) is the standard DDPM noising:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

    - The model is assumed to predict clean x_0 directly:
        x0_pred = model(x_t, t, ...)

    - Training loss is MSE(x0_pred, x0) with optional masking.

    - Sampling uses a DDIM-like deterministic update where x0_pred is
      converted to eps_pred internally.
    """

    betas: torch.Tensor

    def __post_init__(self):
        betas = self.betas
        betas = betas.float()
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [
                torch.ones(
                    1,
                    device=self.alphas_cumprod.device,
                    dtype=self.alphas_cumprod.dtype,
                ),
                self.alphas_cumprod[:-1],
            ],
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
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            device
        )
        return self

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sample x_t from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omab = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_omab * noise

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
        """DDIM step with eta=0 using x-prediction.

        t and t_prev are [B] int tensors.
        The model is assumed to output x0_pred.
        """
        # Predict clean x_0
        x0_pred = model(xt, t, cond=cond, mask=mask)

        # Compute eps_pred from x0_pred and x_t
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, t, xt.shape)
        sqrt_omab = _extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        eps_pred = (xt - sqrt_ab * x0_pred) / torch.clamp(
            sqrt_omab, min=1e-12
        )

        # DDIM update to time t_prev
        ab_prev = _extract(self.alphas_cumprod, t_prev, xt.shape)
        omab_prev = torch.sqrt(torch.clamp(1.0 - ab_prev, min=1e-12))

        x_prev = torch.sqrt(ab_prev) * x0_pred + omab_prev * eps_pred
        if mask is not None:
            x_prev = x_prev * mask[:, :, None].to(x_prev.dtype)
        return x_prev

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
        """Deterministic DDIM sampling loop using x-prediction."""
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
            x = self.ddim_step(
                model,
                x,
                t=t,
                t_prev=t_prev,
                cond=cond,
                mask=mask,
            )

        # Final step: directly output x0_pred at t=0
        t = t_seq[-1].expand(B)
        x0_pred = model(x, t, cond=cond, mask=mask)
        if mask is not None:
            x0_pred = x0_pred * mask[:, :, None].to(x0_pred.dtype)
        return x0_pred

    def training_loss(
        self,
        model,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Training loss with x-prediction.

        The model is assumed to predict clean x_0:
            x0_pred = model(x_t, t, ...)

        Loss is MSE(x0_pred, x0) with optional token mask.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        if mask is not None:
            noise = noise * mask[:, :, None].to(noise.dtype)

        xt = self.q_sample(x0, t, noise=noise)
        x0_pred = model(xt, t, cond=cond, mask=mask)

        if mask is not None:
            loss = (x0_pred - x0) ** 2
            loss = loss * mask[:, :, None].to(loss.dtype)
            denom = torch.clamp(mask.sum() * x0.size(-1), min=1.0)
            return loss.sum() / denom

        return F.mse_loss(x0_pred, x0)
