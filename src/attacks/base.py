"""Attack base class.

Threat model: the adversary can perturb each agent's raw observation vector
inside an L_inf ball of radius epsilon. Agent IDs and past-action features
are NOT perturbed (they aren't "sensor readings" and the MAC reconstructs
them on every step anyway).
"""

from __future__ import annotations

import numpy as np


class Attack:
    """Abstract base for observation perturbations.

    Subclasses implement `perturb(obs, ctx)` which takes the clean observation
    stack shape [n_agents, obs_dim] and returns a perturbed stack of the same
    shape and dtype. `ctx` is a dict with anything the attack needs — for
    FGSM that's the MAC + its current hidden state + last-action onehot; for
    random noise it's empty.
    """

    name: str = "base"

    def perturb(self, obs: np.ndarray, ctx: dict) -> np.ndarray:
        raise NotImplementedError


class NoAttack(Attack):
    """Identity — the clean-env baseline in the same eval harness."""

    name = "none"

    def perturb(self, obs: np.ndarray, ctx: dict) -> np.ndarray:
        return obs
