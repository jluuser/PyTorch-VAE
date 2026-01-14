# coding: utf-8

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


ScheduleName = Literal["linear", "cosine"]


def linear_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    num_steps = int(num_steps)
    if num_steps <= 1:
        raise ValueError("num_steps must be > 1")
    return torch.linspace(float(beta_start), float(beta_end), num_steps, dtype=torch.float64)


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal (Improved DDPM).

    Returns betas in float64 for numerical stability.
    """
    num_steps = int(num_steps)
    if num_steps <= 1:
        raise ValueError("num_steps must be > 1")

    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_steps) + float(s)) / (1.0 + float(s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-8, 0.999)


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor

    @property
    def num_steps(self) -> int:
        return int(self.betas.numel())

    @classmethod
    def build(
        cls,
        num_steps: int,
        schedule: ScheduleName = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        cosine_s: float = 0.008,
    ) -> "DiffusionSchedule":
        schedule = str(schedule).lower()
        if schedule == "linear":
            betas = linear_beta_schedule(num_steps, beta_start=beta_start, beta_end=beta_end)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(num_steps, s=cosine_s)
        else:
            raise ValueError(f"unknown schedule: {schedule}")
        return cls(betas=betas)
