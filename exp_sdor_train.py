import argparse
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "epymarl" / "src"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from envs import REGISTRY as env_REGISTRY

from attacks.sdor import SDorAgent
from attacks.sdor_stor import stor_step

ENV_MAP = {
    "mpe_simple_spread": {
        "key":        "pz-mpe-simple-spread-v3",
        "algo_config": {"shared": "mappo", "independent": "mappo_ns"},
        "time_limit": 25,
    },
}


def find_checkpoint(repo_root, env, algo, sharing, seed):
    models_root = repo_root / "results" / env / algo / sharing / f"seed{seed}" / "models"
    valid = [
        p.parent for p in models_root.glob("**/agent.th")
        if p.parent.name.isdigit()
    ]
    if not valid:
        raise FileNotFoundError(f"No checkpoint found under {models_root}")
    return max(valid, key=lambda p: int(p.name))


def _save_progress_plot(metrics: dict, path: Path, title: str):
    episodes = np.array(metrics["episodes"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, ylabel in [
        (axes[0], "ep_return",   "Team reward (under attack)"),
        (axes[1], "critic_loss", "SDor critic loss"),
        (axes[2], "actor_loss",  "SDor actor loss"),
    ]:
        vals  = np.array(metrics[key], dtype=float)
        valid = ~np.isnan(vals)
        ax.plot(episodes[valid], vals[valid])
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def find_sacred_config(repo_root, algo_config, env_key, seed, env, algo, sharing):
    sacred_dir = repo_root / "epymarl" / "results" / "sacred" / algo_config / env_key
    expected_suffix = f"{env}/{algo}/{sharing}/seed{seed}"
    matches = []
    for cfg_path in sacred_dir.glob("*/config.json"):
        cfg   = json.loads(cfg_path.read_text())
        local = cfg.get("local_results_path", "").replace("\\", "/")
        if cfg.get("seed") == seed and local.endswith(expected_suffix):
            matches.append((int(cfg_path.parent.name), cfg))
    if not matches:
        raise FileNotFoundError(
            f"No Sacred config found for seed={seed}, sharing={sharing} in {sacred_dir}"
        )
    _, config_dict = max(matches, key=lambda x: x[0])
    return config_dict


def main():
    parser = argparse.ArgumentParser(
        description="Train SDor offline against a frozen shared MAPPO protagonist"
    )
    parser.add_argument("--algo",             default="mappo",
                        choices=["iql", "ippo", "mappo", "qmix", "vdn"])
    parser.add_argument("--env",              required=True, choices=list(ENV_MAP))
    parser.add_argument("--seed",             type=int, required=True,
                        help="Protagonist checkpoint seed to train against")
    parser.add_argument("--epsilon",          type=float, default=0.1,
                        help="STor perturbation budget ε")
    parser.add_argument("--n_train_episodes", type=int, default=10_000)
    parser.add_argument("--hidden_dim",       type=int, default=64,
                        help="SDor GRU/MLP hidden size (paper default: 64)")
    parser.add_argument("--lr",               type=float, default=5e-4)
    parser.add_argument("--batch_size",       type=int, default=256)
    parser.add_argument("--buffer_size",      type=int, default=100_000)
    parser.add_argument("--update_every",     type=int, default=1,
                        help="Run SAC update every N timesteps")
    parser.add_argument("--log_interval",     type=int, default=100)
    parser.add_argument("--save_interval",    type=int, default=1000,
                        help="Save metrics JSON every N episodes")
    parser.add_argument("--plot_interval",    type=int, default=500,
                        help="Save progress PNG every N episodes (0 = disabled)")
    parser.add_argument("--out",              default="results/sdor")
    args = parser.parse_args()

    device = "cuda" if th.cuda.is_available() else "cpu"

    env_info_map = ENV_MAP[args.env]
    env_key      = env_info_map["key"]
    # SDor always trains against the shared protagonist (Option B)
    sharing      = "shared"
    algo_config  = args.algo if sharing == "shared" else f"{args.algo}_ns"

    # --- Load frozen shared protagonist ---
    ckpt_path   = find_checkpoint(REPO_ROOT, args.env, args.algo, sharing, args.seed)
    config_dict = find_sacred_config(
        REPO_ROOT, algo_config, env_key, args.seed,
        args.env, args.algo, sharing,
    )
    args_ns        = SimpleNamespace(**config_dict)
    args_ns.device = device
    print(f"Protagonist checkpoint: {ckpt_path}")

    inner_env = env_REGISTRY["gymma"](
        key=env_key,
        time_limit=env_info_map["time_limit"],
        seed=args.seed,
        common_reward=args_ns.common_reward,
        reward_scalarisation=args_ns.reward_scalarisation,
        pretrained_wrapper=None,
    )
    env_info          = inner_env.get_env_info()
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
    if device == "cuda":
        mac.cuda()
    mac.agent.eval()
    for p in mac.agent.parameters():
        p.requires_grad_(False)

    episode_limit = env_info["episode_limit"]

    # --- SDor ---
    obs_dim = env_info["obs_shape"]
    n_actions = env_info["n_actions"]
    n_agents  = env_info["n_agents"]

    sdor = SDorAgent(
        obs_dim=obs_dim, n_actions=n_actions, n_agents=n_agents,
        hidden_dim=args.hidden_dim, lr=args.lr,
        buffer_size=args.buffer_size, batch_size=args.batch_size,
        device=device,
    )

    # Path keyed by protagonist (env, algo, sharing, seed) so multiple SDors
    # trained against different protagonists at the same seed don't collide.
    out_dir = REPO_ROOT / args.out / args.env / args.algo / sharing / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_path = (
        REPO_ROOT / "training_plots"
        / f"sdor_{args.env}_{args.algo}_{sharing}_seed{args.seed}_eps{args.epsilon}.png"
    )
    plot_title = f"SDor training — {args.env} {args.algo}/{sharing} seed{args.seed} ε={args.epsilon}"

    # --- Metrics ---
    metrics = {
        "episodes":    [],
        "ep_return":   [],
        "critic_loss": [],
        "actor_loss":  [],
        "alpha":       [],
    }

    print(f"Training SDor for {args.n_train_episodes} episodes | "
          f"ε={args.epsilon} | device={device}")

    for episode in range(1, args.n_train_episodes + 1):
        mac.init_hidden(batch_size=1)
        sdor.init_episode()
        inner_env.reset()

        prev_adv_obs    = inner_env.get_obs()           # initial SDor input = clean obs
        prev_adv_action = np.zeros((n_agents, n_actions), dtype=np.float32)

        batch = EpisodeBatch(
            scheme, groups, batch_size=1,
            max_seq_length=episode_limit + 1,
            preprocess=preprocess, device=device,
        )

        ep_return  = 0.0
        terminated = False
        t = 0
        ep_losses  = {"critic_loss": [], "actor_loss": [], "alpha": []}

        while not terminated and t < episode_limit:
            clean_obs    = inner_env.get_obs()
            state        = inner_env.get_state()
            avail_actions = inner_env.get_avail_actions()

            # SDor samples based on previous adversarial observation
            adv_action = sdor.select_action(prev_adv_obs, prev_adv_action, explore=True)

            # STor: FGSM step guided by SDor's direction
            adv_obs = stor_step(clean_obs, mac, adv_action, args.epsilon, args_ns, device)

            # Protagonist acts on adversarial observations
            batch.update({
                "state":         [state],
                "avail_actions": [avail_actions],
                "obs":           [adv_obs],
            }, ts=t)
            actions = mac.select_actions(batch, t_ep=t, t_env=0, test_mode=True)
            _, reward, done, truncated, _ = inner_env.step(actions[0].cpu().numpy())
            terminated = bool(done) or bool(truncated)
            batch.update({
                "actions":    actions,
                "reward":     [[reward]],
                "terminated": [[terminated]],
            }, ts=t)

            # SDor reward = negative team reward (adversary wants to hurt protagonist)
            sdor.store(prev_adv_obs, prev_adv_action, adv_action, -reward, adv_obs, terminated)

            if sdor.can_update() and (t % args.update_every == 0):
                info = sdor.update()
                ep_losses["critic_loss"].append(info["critic_loss"])
                ep_losses["actor_loss"].append(info["actor_loss"])
                ep_losses["alpha"].append(info["alpha"])

            prev_adv_obs    = adv_obs
            prev_adv_action = adv_action
            ep_return      += reward
            t += 1

        metrics["episodes"].append(episode)
        metrics["ep_return"].append(ep_return)
        for k in ("critic_loss", "actor_loss", "alpha"):
            metrics[k].append(float(np.mean(ep_losses[k])) if ep_losses[k] else float("nan"))

        if episode % args.log_interval == 0:
            recent = metrics["ep_return"][-args.log_interval:]
            cl = metrics["critic_loss"][-1]
            al = metrics["actor_loss"][-1]
            print(f"  [{episode}/{args.n_train_episodes}] "
                  f"ep_return={np.mean(recent):.1f} | "
                  f"critic_loss={cl:.4f} | actor_loss={al:.4f} | "
                  f"alpha={sdor.alpha:.4f} | buffer={len(sdor.replay)}")

        if episode % args.save_interval == 0:
            (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))

        if args.plot_interval and episode % args.plot_interval == 0:
            _save_progress_plot(metrics, plot_path, plot_title)

    # --- Final save ---
    sdor.save(str(out_dir))
    (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    _save_progress_plot(metrics, plot_path, plot_title)
    print(f"\nSDor checkpoint → {out_dir / 'sdor.pt'}")
    print(f"Metrics          → {out_dir / 'training_metrics.json'}")


if __name__ == "__main__":
    main()
