"""Single-agent-source FGSM variant, used by the transfer-matrix analysis.

Regular FGSMAttack crafts a perturbation per agent from that agent's own
gradient. For transferability analysis we instead craft ONE perturbation
using agent `source_agent`'s gradient, and apply it ONLY to agent
`target_agent`'s observation; all other agents receive clean observations.

Whether this perturbation hurts agent j when crafted from agent i's policy
(shared network vs N independent networks) is the direct empirical test of
whether parameter sharing creates a shared vulnerability surface.
"""

from __future__ import annotations

import numpy as np
import torch as th

from .base import Attack


class FGSMTransferAttack(Attack):
    name = "fgsm_transfer"

    def __init__(self, epsilon: float, source_agent: int, target_agent: int):
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative; got {epsilon}")
        self.epsilon = float(epsilon)
        self.source_agent = int(source_agent)
        self.target_agent = int(target_agent)

    def perturb(self, obs: np.ndarray, ctx: dict) -> np.ndarray:
        mac = ctx["mac"]
        prev_actions_onehot = ctx.get("prev_actions_onehot")
        device = next(mac.agent.parameters()).device
        n_agents = mac.n_agents
        dtype = obs.dtype

        obs_t = th.tensor(obs, device=device, dtype=th.float32, requires_grad=True)

        pieces = [obs_t]
        if getattr(mac.args, "obs_last_action", False):
            if prev_actions_onehot is not None:
                pieces.append(prev_actions_onehot.reshape(n_agents, -1).to(device))
            else:
                pieces.append(th.zeros(n_agents, mac.args.n_actions, device=device))
        if getattr(mac.args, "obs_agent_id", False):
            pieces.append(th.eye(n_agents, device=device))
        agent_inputs = th.cat(pieces, dim=1)

        hidden_in = mac.hidden_states.detach().clone() if mac.hidden_states is not None else None
        agent_outs, _ = mac.agent(agent_inputs, hidden_in)

        src_out = agent_outs[self.source_agent]  # [n_actions]
        if mac.agent_output_type == "q":
            a_star = src_out.argmax().unsqueeze(0)
            target = src_out.gather(0, a_star).sum()
        elif mac.agent_output_type == "pi_logits":
            probs = th.softmax(src_out, dim=-1)
            a_star = probs.argmax().unsqueeze(0)
            target = probs.gather(0, a_star).sum()
        else:
            raise RuntimeError(
                f"FGSM doesn't know how to attack agent_output_type={mac.agent_output_type!r}"
            )

        grad, = th.autograd.grad(target, obs_t)  # [n_agents, obs_dim]
        # Use ONLY the gradient row for the source agent — that's the direction
        # against the source's policy. Apply to the target's observation.
        perturb_vec = (-self.epsilon) * grad[self.source_agent].sign()  # [obs_dim]

        perturbed = obs_t.detach().cpu().numpy().astype(dtype)
        perturbed[self.target_agent] = (perturbed[self.target_agent]
                                        + perturb_vec.cpu().numpy().astype(dtype))
        return perturbed
