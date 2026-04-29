import numpy as np
import torch as th


def stor_step(obs_list, mac, adv_actions, epsilon, args, device):
    """STor FGSM step guided by SDor's policy perturbation suggestions.

    Finds observation perturbation ô^i that maximally aligns the protagonist's
    action distribution with SDor's suggested direction â^i:

        ô^i = o^i + ε · sign(∇_o (â^i · softmax(π^i(o^i))))

    This is protagonist-specific (uses the target MAC's gradients), so the same
    SDor can attack both shared and independent protagonists (Option B transfer).
    """
    if epsilon == 0.0:
        return obs_list

    n       = len(obs_list)
    obs_t   = th.FloatTensor(np.stack(obs_list)).to(device)
    obs_t.requires_grad_(True)

    parts = [obs_t]
    if getattr(args, "obs_agent_id", False):
        parts.append(th.eye(n, device=device))
    agent_inputs = th.cat(parts, dim=-1)

    hidden = mac.hidden_states.detach().clone()

    with th.enable_grad():
        logits, _ = mac.agent(agent_inputs, hidden)
        probs     = th.softmax(logits, dim=-1)
        adv_t     = th.FloatTensor(np.asarray(adv_actions, dtype=np.float32)).to(device)
        # Maximize dot product → gradient ascent on (adv · probs)
        loss = (adv_t * probs).sum()
        loss.backward()

    with th.no_grad():
        perturbed = obs_t + epsilon * obs_t.grad.sign()

    out = perturbed.detach().cpu().numpy()
    return [out[i].astype(obs_list[0].dtype) for i in range(n)]


class SDorSTorAttack:
    """Eval-time stochastic adversary attack (Option B).

    SDor was trained against the shared protagonist but generalises to any
    protagonist via STor's per-step FGSM (which uses the target MAC's gradients).

    Must be paired with ObsPerturbWrapper, which calls reset_episode() on
    env.reset() so SDor's GRU hidden state is properly initialised each episode.
    """

    def __init__(self, sdor, mac, args_ns, epsilon, device):
        self.sdor     = sdor
        self.mac      = mac
        self.args_ns  = args_ns
        self.epsilon  = epsilon
        self.device   = device
        self._prev_adv_obs    = None
        self._prev_adv_action = np.zeros((sdor.n_agents, sdor.n_actions), dtype=np.float32)

    def reset_episode(self):
        self.sdor.init_episode()
        self._prev_adv_obs    = None
        self._prev_adv_action = np.zeros(
            (self.sdor.n_agents, self.sdor.n_actions), dtype=np.float32
        )

    def __call__(self, obs_list):
        if self._prev_adv_obs is None:
            self._prev_adv_obs = obs_list  # first step: bootstrap from clean obs

        adv_action = self.sdor.select_action(
            self._prev_adv_obs, self._prev_adv_action, explore=False
        )
        adv_obs = stor_step(
            obs_list, self.mac, adv_action, self.epsilon, self.args_ns, self.device
        )
        self._prev_adv_obs    = adv_obs
        self._prev_adv_action = adv_action
        return adv_obs
