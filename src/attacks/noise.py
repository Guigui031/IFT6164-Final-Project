import numpy as np


def no_attack(obs_list):
    return obs_list


def random_noise(obs_list, epsilon):
    """Uniform noise U(-epsilon, epsilon) added to each agent's observation.

    Returns new arrays — does not modify the env's internal _obs cache in-place,
    which keeps get_state() clean (centralized state is not agent-observable).
    MPE observations are unbounded floats so no clamping is applied.
    """
    return [
        obs + np.random.uniform(-epsilon, epsilon, size=obs.shape).astype(obs.dtype)
        for obs in obs_list
    ]
