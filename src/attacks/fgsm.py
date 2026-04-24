"""Single-step FGSM observation attack.

Untargeted: the adversary minimises the protagonist's argmax-action
confidence. For Q-learning heads we decrease the Q-value of the action
the clean policy would take; for pi_logits heads we decrease that
action's softmax probability. In both cases we step obs by
epsilon * sign(grad) in the direction that reduces the target quantity.

Gradients are taken through the agent network using the mac's *current*
hidden state (the "going-in" state for this timestep). The mac's hidden
state is not advanced by this call — the subsequent clean mac.forward
during select_actions will advance it using the perturbed obs, which is
what the protagonist actually sees.
"""

from __future__ import annotations

import numpy as np
import torch as th

from .base import Attack


class FGSMAttack(Attack):
    name = "fgsm"

    def __init__(self, epsilon: float):
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative; got {epsilon}")
        self.epsilon = float(epsilon)

    def perturb(self, obs: np.ndarray, ctx: dict) -> np.ndarray:
        mac = ctx["mac"]
        prev_actions_onehot = ctx.get("prev_actions_onehot")  # optional [1, n_agents, n_actions]

        device = next(mac.agent.parameters()).device
        n_agents = mac.n_agents
        dtype = obs.dtype

        obs_t = th.tensor(obs, device=device, dtype=th.float32, requires_grad=True)  # [n_agents, obs_dim]

        # Reconstruct agent_inputs exactly like BasicMAC._build_inputs would,
        # but with obs_t as the leaf we can differentiate through.
        pieces = [obs_t]
        if getattr(mac.args, "obs_last_action", False):
            if prev_actions_onehot is not None:
                pieces.append(prev_actions_onehot.reshape(n_agents, -1).to(device))
            else:
                n_actions = mac.args.n_actions
                pieces.append(th.zeros(n_agents, n_actions, device=device))
        if getattr(mac.args, "obs_agent_id", False):
            pieces.append(th.eye(n_agents, device=device))

        agent_inputs = th.cat(pieces, dim=1)  # [n_agents, total_input_dim]

        # Use the current hidden state without advancing it. Both RNNAgent and
        # RNNNSAgent accept hidden of shape [bs, n_agents, hidden_dim]; bs=1.
        hidden_in = mac.hidden_states.detach().clone() if mac.hidden_states is not None else None
        agent_outs, _ = mac.agent(agent_inputs, hidden_in)
        # agent_outs shape: [n_agents, n_actions] (for bs=1)

        if mac.agent_output_type == "q":
            a_star = agent_outs.argmax(-1, keepdim=True)
            target = agent_outs.gather(-1, a_star).sum()
        elif mac.agent_output_type == "pi_logits":
            probs = th.softmax(agent_outs, dim=-1)
            a_star = probs.argmax(-1, keepdim=True)
            target = probs.gather(-1, a_star).sum()
        else:
            raise RuntimeError(
                f"FGSM doesn't know how to attack agent_output_type={mac.agent_output_type!r}"
            )

        grad, = th.autograd.grad(target, obs_t)
        # We want to *decrease* the target, so move obs opposite to the gradient.
        perturbation = (-self.epsilon) * grad.sign()
        return (obs_t.detach() + perturbation).cpu().numpy().astype(dtype)
