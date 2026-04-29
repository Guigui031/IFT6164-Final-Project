import numpy as np
import torch as th
import torch.nn.functional as F


class FGSMTransferAttack:
    """Cross-agent FGSM transfer attack.

    Crafts the perturbation from the SOURCE agent's gradient and applies it
    only to the TARGET agent's observation. Used by exp_transfer.py to build
    the N x N matrix that tests whether perturbations crafted for agent i
    fool agent j.

    Shared-parameter networks predict an almost-symmetric matrix with strong
    off-diagonal drops (one Q-function, one gradient landscape). Independent
    networks predict strong diagonal drops only (matches per-agent FGSM) but
    weak off-diagonals.
    """

    def __init__(self, mac, args, epsilon, source, target, device):
        self.mac = mac
        self.args = args
        self.epsilon = epsilon
        self.source = source
        self.target = target
        self.device = device

    def __call__(self, obs_list):
        if self.epsilon == 0.0:
            return obs_list

        n = len(obs_list)
        obs_t = th.FloatTensor(np.stack(obs_list)).to(self.device)  # [n, obs_dim]
        obs_t.requires_grad_(True)

        parts = [obs_t]
        if getattr(self.args, "obs_agent_id", False):
            parts.append(th.eye(n, device=self.device))
        agent_inputs = th.cat(parts, dim=-1)

        hidden = self.mac.hidden_states.detach().clone()

        with th.enable_grad():
            logits, _ = self.mac.agent(agent_inputs, hidden)
            greedy = logits.detach().argmax(dim=-1)
            # Loss is computed only on the source agent's prediction so the
            # gradient w.r.t. obs_t[source] is what FGSM signs.
            F.cross_entropy(logits[self.source:self.source + 1],
                            greedy[self.source:self.source + 1]).backward()

        with th.no_grad():
            grad_sign = obs_t.grad[self.source].sign()
            out = obs_t.detach().cpu().numpy()
            # Apply the source-crafted perturbation only to the target's obs.
            out[self.target] = out[self.target] + self.epsilon * grad_sign.cpu().numpy()

        return [out[i].astype(obs_list[0].dtype) for i in range(n)]
