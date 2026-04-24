"""Observation-perturbation attacks against MARL protagonists.

Usage:
    from src.attacks import get_attack
    attack = get_attack("fgsm", epsilon=0.1)
    obs_adv = attack.perturb(obs_clean, ctx)

Each Attack.perturb takes a numpy array of shape [n_agents, obs_dim] plus a
context dict carrying whatever the attack needs (e.g. the MAC for FGSM).
The output has the same shape and dtype as the input.
"""

from .base import Attack, NoAttack
from .random_noise import RandomNoiseAttack
from .fgsm import FGSMAttack
from .fgsm_per_agent import FGSMTransferAttack

ATTACKS = {
    "none": NoAttack,
    "random": RandomNoiseAttack,
    "fgsm": FGSMAttack,
    "fgsm_transfer": FGSMTransferAttack,
}


def get_attack(name: str, **kwargs) -> Attack:
    if name not in ATTACKS:
        raise ValueError(f"unknown attack {name!r}; known: {sorted(ATTACKS)}")
    cls = ATTACKS[name]
    if name == "none":
        return cls()
    return cls(**kwargs)


__all__ = [
    "Attack", "NoAttack", "RandomNoiseAttack", "FGSMAttack", "FGSMTransferAttack",
    "get_attack", "ATTACKS",
]
