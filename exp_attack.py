"""Load a trained EPyMARL checkpoint and evaluate it under an observation attack.

Usage (from repo root, venv active):

    python exp_attack.py --ckpt results/mpe_simple_spread/iql/shared/seed1 \
                         --attack random --epsilon 0.1 --n_episodes 100

Outputs:
    <ckpt>/attacks/<attack>_eps<eps>/metrics.json

The rollout loop is a re-implementation of EpisodeRunner.run() that injects
the attack on the raw per-agent observation tuple before it is written into
the episode batch. Everything downstream (MAC forward, action selection,
env.step) sees only the perturbed observation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th

REPO_ROOT = Path(__file__).resolve().parent
EPYMARL_SRC = REPO_ROOT / "epymarl" / "src"
sys.path.insert(0, str(EPYMARL_SRC))
# epymarl's modules do `from envs import REGISTRY` etc., so epymarl/src must be on sys.path
# even though we use its packages as epymarl.src in our repo layout.

# Import EPyMARL pieces (require sys.path tweak above).
from components.episode_buffer import ReplayBuffer  # noqa: E402
from components.transforms import OneHot  # noqa: E402
from controllers import REGISTRY as mac_REGISTRY  # noqa: E402
from learners import REGISTRY as le_REGISTRY  # noqa: E402
from runners import REGISTRY as r_REGISTRY  # noqa: E402
from utils.logging import get_logger  # noqa: E402

# Our attacks live outside epymarl/.
sys.path.insert(0, str(REPO_ROOT))
from src.attacks import get_attack  # noqa: E402


def find_sacred_config(ckpt_dir: Path) -> Path:
    """Sacred writes one config.json per run at .../sacred/<algo>/<map>/<run_id>/config.json."""
    candidates = list((ckpt_dir / "epymarl" / "sacred").glob("*/*/*/config.json"))
    if not candidates:
        raise FileNotFoundError(f"No sacred config.json under {ckpt_dir}/epymarl/sacred/")
    # If multiple sacred runs exist, take the one with the largest run_id.
    candidates.sort(key=lambda p: int(p.parent.name))
    return candidates[-1]


def find_latest_model_dir(ckpt_dir: Path) -> Path:
    """EPyMARL saves under .../models/<unique_token>/<step>/agent.th. Return the dir of the
    largest step across all unique_tokens under this cell."""
    model_roots = list((ckpt_dir / "epymarl" / "models").glob("*"))
    if not model_roots:
        raise FileNotFoundError(f"No models/ dir under {ckpt_dir}/epymarl/")
    best = None
    for root in model_roots:
        for step_dir in root.iterdir():
            if step_dir.is_dir() and step_dir.name.isdigit():
                step = int(step_dir.name)
                if best is None or step > best[0]:
                    best = (step, step_dir)
    if best is None:
        raise FileNotFoundError(f"No step dirs under {ckpt_dir}/epymarl/models/*")
    return best[1]


def load_args(config_path: Path) -> SimpleNamespace:
    with open(config_path) as f:
        cfg = json.load(f)
    return SimpleNamespace(**cfg)


def build_env_mac(args: SimpleNamespace):
    """Instantiate runner, scheme, mac mirroring run_sequential() from epymarl/src/run.py."""
    # EPyMARL's config.json doesn't include `device`; run.py sets it at runtime.
    if not hasattr(args, "device"):
        args.device = "cuda" if args.use_cuda and th.cuda.is_available() else "cpu"

    logger = get_logger()
    # Silence EPyMARL's chatty logger for attack eval.
    import logging
    logger.setLevel(logging.WARNING)

    # Runner needs a logger with .log_stat etc. — but we just want the env out of it, so
    # construct a minimal logger-like object. The EPyMARL logger module has a Logger
    # class; easier to use the real one without observers.
    from utils.logging import Logger
    runner_logger = Logger(logger)

    # Build runner (gives us env_info for scheme).
    runner = r_REGISTRY[args.runner](args=args, logger=runner_logger)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    if args.common_reward:
        scheme["reward"] = {"vshape": (1,)}
    else:
        scheme["reward"] = {"vshape": (args.n_agents,)}
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme, groups, args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    learner = le_REGISTRY[args.learner](mac, buffer.scheme, runner_logger, args)

    if args.use_cuda and th.cuda.is_available():
        learner.cuda()

    return runner, mac, learner, scheme, groups, preprocess


def rollout_one_episode(runner, mac, attack, args) -> float:
    """Re-implementation of EpisodeRunner.run(test_mode=True) with attack injection.

    Differences from upstream:
      * Uses attack.perturb(obs_stack, ctx) on the raw per-agent observations
        before writing them into the episode batch.
      * Returns the episode return scalar instead of the episode batch + stats.
    """
    runner.reset()
    mac.init_hidden(batch_size=runner.batch_size)
    episode_return = 0.0 if args.common_reward else np.zeros(args.n_agents)
    t = 0
    terminated = False

    while not terminated:
        # Stack raw per-agent obs -> [n_agents, obs_dim].
        raw_obs = runner.env.get_obs()
        obs_stack = np.stack([np.asarray(o, dtype=np.float32) for o in raw_obs])

        # Build FGSM context (prev actions onehot for obs_last_action envs).
        prev_onehot = None
        if getattr(args, "obs_last_action", False) and t > 0:
            prev_onehot = runner.batch["actions_onehot"][:, t - 1]

        obs_stack = attack.perturb(obs_stack, ctx={
            "mac": mac,
            "prev_actions_onehot": prev_onehot,
            "t": t,
        })
        perturbed_obs = tuple(obs_stack[i] for i in range(args.n_agents))

        pre = {
            "state": [runner.env.get_state()],
            "avail_actions": [runner.env.get_avail_actions()],
            "obs": [perturbed_obs],
        }
        runner.batch.update(pre, ts=t)

        actions = mac.select_actions(runner.batch, t_ep=t, t_env=0, test_mode=True)
        _, reward, terminated, truncated, env_info = runner.env.step(actions[0])
        terminated = terminated or truncated
        episode_return += reward

        post = {
            "actions": actions,
            "terminated": [(terminated != env_info.get("episode_limit", False),)],
        }
        if args.common_reward:
            post["reward"] = [(reward,)]
        else:
            post["reward"] = [tuple(reward)]
        runner.batch.update(post, ts=t)
        t += 1

    return float(episode_return) if args.common_reward else episode_return.tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, type=Path,
                   help="Cell dir, e.g. results/mpe_simple_spread/iql/shared/seed1")
    p.add_argument("--attack", required=True, choices=["none", "random", "fgsm"])
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345, help="RNG seed for attack + env")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir. Default: <ckpt>/attacks/<attack>_eps<eps>/")
    args_cli = p.parse_args()

    # Load training config snapshot.
    cfg_path = find_sacred_config(args_cli.ckpt)
    args = load_args(cfg_path)
    model_dir = find_latest_model_dir(args_cli.ckpt)
    print(f"[attack] cell = {args_cli.ckpt}")
    print(f"[attack] config = {cfg_path}")
    print(f"[attack] model = {model_dir}")
    print(f"[attack] attack = {args_cli.attack}  epsilon = {args_cli.epsilon}  n_episodes = {args_cli.n_episodes}")

    # Seed RNGs.
    np.random.seed(args_cli.seed)
    th.manual_seed(args_cli.seed)
    args.seed = args_cli.seed
    args.env_args["seed"] = args_cli.seed
    args.evaluate = True
    args.checkpoint_path = str(model_dir.parent)  # unique_token dir
    args.load_step = int(model_dir.name)
    # Override t_max to 0 so run_sequential's training loop is skipped — but we don't call
    # run_sequential; we drive the rollout ourselves. Just normalize fields the runner reads.
    args.test_nepisode = args_cli.n_episodes
    args.test_interval = 1
    args.log_interval = int(1e18)
    args.runner_log_interval = int(1e18)
    args.learner_log_interval = int(1e18)
    args.save_model = False
    args.use_tensorboard = False

    # Build env + mac + learner, load weights.
    runner, mac, learner, scheme, groups, preprocess = build_env_mac(args)
    learner.load_models(str(model_dir))

    # Build attack.
    attack = get_attack(args_cli.attack, **({"epsilon": args_cli.epsilon}
                                            if args_cli.attack != "none" else {}))

    # Rollout.
    t0 = time.time()
    returns = []
    for ep in range(args_cli.n_episodes):
        r = rollout_one_episode(runner, mac, attack, args)
        returns.append(r)
    elapsed = time.time() - t0

    # Summary stats.
    if args.common_reward:
        mean = float(np.mean(returns))
        std = float(np.std(returns))
    else:
        arr = np.asarray(returns)  # [n_episodes, n_agents]
        mean = arr.mean(axis=0).tolist()
        std = arr.std(axis=0).tolist()

    out_dir = args_cli.out or (args_cli.ckpt / "attacks" / f"{args_cli.attack}_eps{args_cli.epsilon}")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "attack": args_cli.attack,
        "epsilon": args_cli.epsilon,
        "n_episodes": args_cli.n_episodes,
        "seed": args_cli.seed,
        "return_mean": mean,
        "return_std": std,
        "returns": returns,
        "elapsed_sec": elapsed,
        "ckpt": str(args_cli.ckpt),
        "model_step": int(model_dir.name),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[attack] return_mean = {mean}  return_std = {std}  ({args_cli.n_episodes} eps in {elapsed:.1f}s)")
    print(f"[attack] wrote {out_dir / 'metrics.json'}")

    runner.close_env()
    return 0


if __name__ == "__main__":
    sys.exit(main())
