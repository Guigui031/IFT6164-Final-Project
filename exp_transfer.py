"""Cross-agent perturbation transfer analysis.

For a trained cell, runs rollouts where an FGSM perturbation is crafted using
agent `source`'s gradient and applied only to agent `target`'s observation.
Produces an N x N matrix of mean team return per (source, target) pair.

Shared-parameter networks predict an almost-symmetric matrix with strong
off-diagonal drops (one Q-function, one gradient landscape). Independent
networks predict strong diagonal drops only (matches per-agent FGSM) but
weak off-diagonals — a perturbation crafted from agent i's net shouldn't
fool agent j's net.

Usage (from repo root, venv active):
    python exp_transfer.py --algo iql --sharing shared --env mpe_simple_spread \
                           --seed 1 --epsilon 0.25 --n_episodes 50

Output:
    results/<env>/<algo>/<sharing>/seed<N>/transfer_eps<eps>.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "epymarl" / "src"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from envs import REGISTRY as env_REGISTRY

from wrappers.obs_perturb import ObsPerturbWrapper
from attacks.fgsm_transfer import FGSMTransferAttack
from attacks.noise import no_attack

from exp_attack import (
    ENV_MAP, find_checkpoint, find_sacred_config, run_episode,
    _get_epymarl_config,
)


def main():
    parser = argparse.ArgumentParser(description="Cross-agent FGSM transfer matrix")
    parser.add_argument("--algo",       choices=["iql", "ippo", "mappo", "qmix", "vdn", "maddpg"],
                        required=True)
    parser.add_argument("--sharing",    choices=["shared", "independent"], required=True)
    parser.add_argument("--env",        choices=list(ENV_MAP), required=True)
    parser.add_argument("--seed",       type=int, required=True)
    parser.add_argument("--epsilon",    type=float, default=0.25)
    parser.add_argument("--n_episodes", type=int,   default=50)
    args = parser.parse_args()

    env_info_map  = ENV_MAP[args.env]
    env_config    = env_info_map["env_config"]
    env_args_dict = env_info_map["env_args"]
    sacred_subdir = env_info_map["sacred_subdir"]
    algo_config   = _get_epymarl_config(args.algo, args.sharing)

    ckpt_path = find_checkpoint(REPO_ROOT, args.env, args.algo, args.sharing, args.seed)
    print(f"Checkpoint: {ckpt_path}")

    config_dict = find_sacred_config(
        REPO_ROOT, algo_config, sacred_subdir, args.seed,
        args.env, args.algo, args.sharing,
    )
    args_ns = SimpleNamespace(**config_dict)
    args_ns.device = "cuda" if th.cuda.is_available() else "cpu"

    common_kwargs = {
        "seed": args.seed,
        "common_reward": args_ns.common_reward,
        "reward_scalarisation": args_ns.reward_scalarisation,
    }
    if env_config == "gymma":
        common_kwargs["pretrained_wrapper"] = None
    inner_env = env_REGISTRY[env_config](**common_kwargs, **env_args_dict)
    env_info = inner_env.get_env_info()
    args_ns.n_agents  = env_info["n_agents"]
    args_ns.n_actions = env_info["n_actions"]

    scheme = {
        "state":          {"vshape": env_info["state_shape"]},
        "obs":            {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions":        {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions":  {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward":         {"vshape": (1,)},
        "terminated":     {"vshape": (1,), "dtype": th.uint8},
        "actions_onehot": {"vshape": (env_info["n_actions"],), "group": "agents"},
    }
    groups     = {"agents": args_ns.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args_ns.n_actions)])}

    mac = mac_REGISTRY[args_ns.mac](scheme, groups, args_ns)
    mac.load_models(str(ckpt_path))
    if args_ns.device == "cuda":
        mac.cuda()

    n = args_ns.n_agents

    env = ObsPerturbWrapper(inner_env, no_attack)
    print(f"Running {args.n_episodes} clean episodes for baseline...")
    baseline_returns = [run_episode(env, mac, scheme, groups, preprocess, args_ns)
                        for _ in range(args.n_episodes)]
    baseline_mean = float(np.mean(baseline_returns))
    print(f"  baseline mean return = {baseline_mean:.3f}")

    matrix = np.zeros((n, n), dtype=float)
    per_cell_returns = [[[] for _ in range(n)] for _ in range(n)]
    for s in range(n):
        for t in range(n):
            attack_fn = FGSMTransferAttack(mac, args_ns, args.epsilon, s, t, args_ns.device)
            env = ObsPerturbWrapper(inner_env, attack_fn)
            returns = [run_episode(env, mac, scheme, groups, preprocess, args_ns)
                       for _ in range(args.n_episodes)]
            matrix[s, t] = float(np.mean(returns))
            per_cell_returns[s][t] = returns
            print(f"  source={s} -> target={t}  mean={matrix[s, t]:.3f}  "
                  f"(drop = {baseline_mean - matrix[s, t]:+.3f})")

    result = {
        "algo": args.algo, "sharing": args.sharing,
        "env": args.env, "seed": args.seed,
        "epsilon": args.epsilon, "n_episodes": args.n_episodes,
        "n_agents": n,
        "baseline_mean": baseline_mean,
        "baseline_returns": baseline_returns,
        "matrix": matrix.tolist(),
        "returns": per_cell_returns,
        "checkpoint_timestep": int(ckpt_path.name),
    }
    out_dir  = REPO_ROOT / "results" / args.env / args.algo / args.sharing / f"seed{args.seed}"
    out_path = out_dir / f"transfer_eps{args.epsilon}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved -> {out_path}")

    print("\nTransfer matrix (rows=source, cols=target):")
    for s in range(n):
        row = " ".join(f"{matrix[s, t]:7.2f}" for t in range(n))
        print(f"s={s}: {row}")
    print(f"baseline (clean) = {baseline_mean:.3f}")


if __name__ == "__main__":
    main()
