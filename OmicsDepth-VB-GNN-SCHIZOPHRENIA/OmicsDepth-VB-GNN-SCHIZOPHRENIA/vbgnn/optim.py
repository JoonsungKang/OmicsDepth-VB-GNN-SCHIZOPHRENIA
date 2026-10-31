"""Minimal Adam optimiser for dictionaries of NumPy arrays."""
from __future__ import annotations

import numpy as np


class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 lr_overrides: dict[str, float] | None = None, maximize: bool = False):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.lr_overrides = lr_overrides or {}
        self.sign = 1.0 if maximize else -1.0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for k, g in grads.items():
            if k not in self.params:
                continue
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * g * g
            lr = self.lr_overrides.get(k, self.lr)
            self.params[k] += self.sign * lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)

    def state_dict(self) -> dict:
        return {"t": self.t, "m": self.m, "v": self.v}
