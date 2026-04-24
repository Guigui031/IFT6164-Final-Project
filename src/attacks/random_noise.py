"""Uniform L_inf random-noise attack.

Baseline against which gradient-based attacks should be strictly stronger.
If an algorithm's FGSM drop isn't meaningfully worse than its random-noise
drop at the same epsilon, the policy probably wasn't vulnerable to gradient
leakage in that setting to begin with.
"""

from __future__ import annotations

import numpy as np

from .base import Attack


class RandomNoiseAttack(Attack):
    name = "random"

    def __init__(self, epsilon: float):
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative; got {epsilon}")
        self.epsilon = float(epsilon)

    def perturb(self, obs: np.ndarray, ctx: dict) -> np.ndarray:
        noise = np.random.uniform(-self.epsilon, self.epsilon, size=obs.shape).astype(obs.dtype)
        return obs + noise
