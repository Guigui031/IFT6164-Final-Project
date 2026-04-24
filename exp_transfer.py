"""Cross-agent perturbation transfer analysis.

For a trained cell, runs rollouts where an FGSM perturbation is crafted using
agent `source`'s gradient and applied only to agent `target`'s observation.
Produces an N x N matrix of mean team return per (source, target) pair.

Shared-parameter networks predict an almost-symmetric matrix with strong
off-diagonal drops (one Q-function, one gradient landscape).  Independent
networks predict strong diagonal drops (matches per-agent FGSM) but weak
off-diagonals — a perturbation crafted from agent i's net shouldn't fool
agent j's net.

Usage (from repo root, venv active):

    python exp_transfer.py --ckpt results/mpe_simple_spread/iql/shared/seed1 \
                           --epsilon 0.1 --n_episodes 50

Output:
    <ckpt>/transfer/eps<eps>/matrix.json   # N x N plus baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from exp_attack import (
    build_env_mac, find_latest_model_dir, find_sacred_config,
    load_args, rollout_one_episode,
)
from src.attacks import FGSMTransferAttack, NoAttack


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", type=Path, default=None)
    args_cli = p.parse_args()

    cfg_path = find_sacred_config(args_cli.ckpt)
    args = load_args(cfg_path)
    model_dir = find_latest_model_dir(args_cli.ckpt)
    print(f"[transfer] cell = {args_cli.ckpt}")
    print(f"[transfer] model = {model_dir}")
    print(f"[transfer] epsilon = {args_cli.epsilon}  n_episodes = {args_cli.n_episodes}")

    np.random.seed(args_cli.seed)
    th.manual_seed(args_cli.seed)
    args.seed = args_cli.seed
    args.env_args["seed"] = args_cli.seed
    args.evaluate = True
    args.checkpoint_path = str(model_dir.parent)
    args.load_step = int(model_dir.name)
    args.test_nepisode = args_cli.n_episodes
    args.test_interval = 1
    args.log_interval = int(1e18)
    args.runner_log_interval = int(1e18)
    args.learner_log_interval = int(1e18)
    args.save_model = False
    args.use_tensorboard = False

    runner, mac, learner, _, _, _ = build_env_mac(args)
    learner.load_models(str(model_dir))
    n_agents = args.n_agents

    def run_many(attack, n_eps):
        returns = []
        for _ in range(n_eps):
            returns.append(rollout_one_episode(runner, mac, attack, args))
        return returns

    t0 = time.time()
    baseline_returns = run_many(NoAttack(), args_cli.n_episodes)
    baseline_mean = float(np.mean(baseline_returns))
    print(f"[transfer] baseline (no attack) mean = {baseline_mean:.3f}")

    matrix = np.zeros((n_agents, n_agents), dtype=float)
    per_cell_returns: list[list[list[float]]] = [[[] for _ in range(n_agents)] for _ in range(n_agents)]
    for s in range(n_agents):
        for t in range(n_agents):
            atk = FGSMTransferAttack(epsilon=args_cli.epsilon, source_agent=s, target_agent=t)
            returns = run_many(atk, args_cli.n_episodes)
            matrix[s, t] = float(np.mean(returns))
            per_cell_returns[s][t] = returns
            print(f"[transfer]  source={s} -> target={t}  mean_return = {matrix[s, t]:.3f}  (drop vs clean = {baseline_mean - matrix[s, t]:+.3f})")

    elapsed = time.time() - t0

    out_dir = args_cli.out or (args_cli.ckpt / "transfer" / f"eps{args_cli.epsilon}")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "ckpt": str(args_cli.ckpt),
        "epsilon": args_cli.epsilon,
        "n_episodes": args_cli.n_episodes,
        "seed": args_cli.seed,
        "n_agents": n_agents,
        "baseline_mean": baseline_mean,
        "baseline_returns": baseline_returns,
        "matrix": matrix.tolist(),           # [source][target] = mean_return
        "returns": per_cell_returns,         # [source][target] = list of ep returns
        "model_step": int(model_dir.name),
        "elapsed_sec": elapsed,
    }
    with open(out_dir / "matrix.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[transfer] wrote {out_dir / 'matrix.json'} ({elapsed:.1f}s)")

    # Pretty-print summary.
    print("\n[transfer] matrix of mean returns (rows=source, cols=target):")
    header = "       " + "  ".join(f"t={t:2d}" for t in range(n_agents))
    print(header)
    for s in range(n_agents):
        row = " ".join(f"{matrix[s, t]:7.2f}" for t in range(n_agents))
        print(f"s={s}: {row}")
    print(f"baseline (clean) = {baseline_mean:.3f}")

    runner.close_env()
    return 0


if __name__ == "__main__":
    sys.exit(main())
