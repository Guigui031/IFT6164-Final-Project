"""Verify that EPyMARL's parameter-sharing toggle produces genuinely independent networks.

Loads the shared RNNAgent and the non-shared RNNNSAgent with identical architectural args
and prints parameter counts. For N agents, NS should have ~N× the params of shared.

Run with the project venv activated:
    python src/utils/verify_sharing.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

# make epymarl.src importable
EPYMARL_SRC = Path(__file__).resolve().parents[2] / "epymarl" / "src"
sys.path.insert(0, str(EPYMARL_SRC))

from modules.agents.rnn_agent import RNNAgent
from modules.agents.rnn_ns_agent import RNNNSAgent


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def make_args(n_agents, hidden_dim, n_actions, use_rnn):
    return SimpleNamespace(
        n_agents=n_agents,
        hidden_dim=hidden_dim,
        n_actions=n_actions,
        use_rnn=use_rnn,
    )


def main():
    # MPE simple_spread with N=3: obs is 18-dim, 5 discrete actions.
    # EPyMARL prepends agent_id one-hot (size n_agents) when obs_agent_id=True,
    # so shared input_shape = 18 + 3 = 21; non-shared input_shape = 18.
    # To make a clean *architectural* comparison we use the same input_shape for both.
    n_agents = 3
    hidden_dim = 128
    n_actions = 5
    input_shape = 18

    for use_rnn in (True, False):
        args = make_args(n_agents, hidden_dim, n_actions, use_rnn)
        shared = RNNAgent(input_shape, args)
        ns = RNNNSAgent(input_shape, args)

        shared_params = count_params(shared)
        ns_params = count_params(ns)
        ratio = ns_params / shared_params

        tag = "GRU" if use_rnn else "MLP"
        print(f"--- {tag} (use_rnn={use_rnn}) ---")
        print(f"  shared (RNNAgent):       {shared_params:>8,} params")
        print(f"  independent (RNNNSAgent): {ns_params:>8,} params  ({ratio:.2f}x shared)")
        if abs(ratio - n_agents) > 0.01:
            print(f"  WARNING: ratio {ratio:.3f} != n_agents {n_agents}; toggle may not be disjoint")
        else:
            print(f"  OK: ratio matches n_agents={n_agents} within tolerance")
        print()


if __name__ == "__main__":
    main()
