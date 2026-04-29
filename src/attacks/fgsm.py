import numpy as np
import torch as th
import torch.nn.functional as F


class FGSMAttack:
    """White-box FGSM on agent observations.

    obs_adv = obs + ε * sign(∇_obs CE(π(obs), argmax π(obs).detach()))

    Uses the MAC's current hidden state (detached clone) so the real
    hidden state is never modified. Works for both BasicMAC (shared RNN)
    and NonSharedMAC (RNNNSAgent with per-agent sub-networks).
    """

    def __init__(self, mac, args, epsilon, device):
        self.mac = mac
        self.args = args
        self.epsilon = epsilon
        self.device = device

    def __call__(self, obs_list):
        if self.epsilon == 0.0:
            return obs_list

        n = len(obs_list)
        obs_dim = obs_list[0].shape[0]
        obs_t = th.FloatTensor(np.stack(obs_list)).to(self.device)  # [n, obs_dim]
        obs_t.requires_grad_(True)

        # Mirror BasicMAC._build_inputs (obs_last_action=False for MAPPO/MAPPO_NS)
        parts = [obs_t]
        if getattr(self.args, "obs_agent_id", False):
            parts.append(th.eye(n, device=self.device))
        agent_inputs = th.cat(parts, dim=-1)  # [n, input_dim]

        # Detached hidden state — must not corrupt the real MAC state
        hidden = self.mac.hidden_states.detach().clone()  # [1, n, hidden_dim]

        with th.enable_grad():
            logits, _ = self.mac.agent(agent_inputs, hidden)  # [n, n_actions]
            greedy = logits.detach().argmax(dim=-1)            # [n]
            F.cross_entropy(logits, greedy).backward()

        with th.no_grad():
            perturbed = obs_t + self.epsilon * obs_t.grad.sign()

        out = perturbed.cpu().numpy()  # [n, obs_dim]
        return [out[i].astype(obs_list[0].dtype) for i in range(n)]
