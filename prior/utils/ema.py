# coding: utf-8

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class EMA:
    decay: float = 0.999

    def __post_init__(self):
        self.decay = float(self.decay)
        self.shadow: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def register(self, model: nn.Module):
        self.shadow = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        if not self.shadow:
            self.register(model)
            return
        d = self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                continue
            self.shadow[name].mul_(d).add_(param.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].data)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor], device: torch.device):
        self.shadow = {k: v.to(device) for k, v in state.items()}
