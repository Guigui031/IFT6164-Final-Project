class ObsPerturbWrapper:
    """Wraps any MultiAgentEnv and intercepts get_obs() to apply an attack.

    get_state() is intentionally NOT overridden: GymmaWrapper.get_state()
    concatenates the env's internal _obs cache, which stays clean because
    noise.py returns new arrays rather than modifying in-place.

    Args:
        env:       A MultiAgentEnv instance (e.g. GymmaWrapper).
        attack_fn: callable(list[np.ndarray]) -> list[np.ndarray]
    """

    def __init__(self, env, attack_fn):
        self._env = env
        self._attack_fn = attack_fn

    def get_obs(self):
        return self._attack_fn(self._env.get_obs())

    def reset(self):
        result = self._env.reset()
        if hasattr(self._attack_fn, "reset_episode"):
            self._attack_fn.reset_episode()
        return result

    def __getattr__(self, name):
        return getattr(self._env, name)
