"""Exponential moving average of model weights.

Sampling from EMA weights is consistently better behaved than sampling from the
raw SGD iterate, which keeps bouncing around the optimum. Cheap insurance.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: int = 1000):
        self.decay = decay
        self.warmup = warmup
        self.step_count = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def _current_decay(self) -> float:
        # Ramp the decay in, so the average is not dominated by the random init.
        if self.warmup <= 0:
            return self.decay
        return min(self.decay, (1 + self.step_count) / (10 + self.step_count))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self._current_decay()
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src.detach(), alpha=1.0 - d)
            else:
                v.copy_(src)
        self.step_count += 1

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)
