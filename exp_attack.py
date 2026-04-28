import argparse
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th
import yaml

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "epymarl" / "src"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from envs import REGISTRY as env_REGISTRY

from attacks.noise import no_attack, random_noise
from wrappers.obs_perturb import ObsPerturbWrapper

ENV_MAP = {
    "mpe_simple_spread": {
        "key": "pz-mpe-simple-spread-v3",
        "algo_config": {"shared": "mappo", "independent": "mappo_ns"},
        "time_limit": 25,
    },
}


def build_attack(name, epsilon, mac, args_ns, device, sdor_ckpt=None):
    if name == "no_attack":
        return no_attack
    if name == "random_noise":
        return lambda obs: random_noise(obs, epsilon)
    if name == "fgsm":
        from attacks.fgsm import FGSMAttack
        return FGSMAttack(mac, args_ns, epsilon, device)
    if name == "sdor_stor":
        if sdor_ckpt is None:
            raise ValueError("--sdor_ckpt is required for sdor_stor attack")
        from attacks.sdor import SDorAgent
        from attacks.sdor_stor import SDorSTorAttack
        sdor = SDorAgent.load(sdor_ckpt, device=device)
        return SDorSTorAttack(sdor, mac, args_ns, epsilon, device)
    raise ValueError(f"Unknown attack: {name}")


def find_checkpoint(repo_root, env, algo, sharing, seed):
    models_root = repo_root / "results" / env / algo / sharing / f"seed{seed}" / "models"
    # structure: models/<unique_token>/<timestep>/agent.th  (two levels deep)
    valid = [
        p.parent for p in models_root.glob("**/agent.th")
        if p.parent.name.isdigit()
    ]
    if not valid:
        raise FileNotFoundError(f"No checkpoint found under {models_root}")
    return max(valid, key=lambda p: int(p.name))


def find_sacred_config(repo_root, algo_config, env_key, seed, env, algo, sharing):
    sacred_dir = repo_root / "epymarl" / "results" / "sacred" / algo_config / env_key
    expected_suffix = f"{env}/{algo}/{sharing}/seed{seed}"
    matches = []
    for cfg_path in sacred_dir.glob("*/config.json"):
        cfg = json.loads(cfg_path.read_text())
        local = cfg.get("local_results_path", "").replace("\\", "/")
        if cfg.get("seed") == seed and local.endswith(expected_suffix):
            matches.append((int(cfg_path.parent.name), cfg))
    if not matches:
        raise FileNotFoundError(
            f"No Sacred config found for seed={seed}, sharing={sharing} in {sacred_dir}"
        )
    _, config_dict = max(matches, key=lambda x: x[0])
    return config_dict


def run_episode(env, mac, scheme, groups, preprocess, args_ns):
    episode_limit = env.get_env_info()["episode_limit"]
    batch = EpisodeBatch(
        scheme, groups, batch_size=1,
        max_seq_length=episode_limit + 1,
        preprocess=preprocess,
        device=args_ns.device,
    )
    mac.init_hidden(batch_size=1)
    env.reset()

    ep_return, terminated, t = 0.0, False, 0
    while not terminated:
        batch.update({
            "state":         [env.get_state()],
            "avail_actions": [env.get_avail_actions()],
            "obs":           [env.get_obs()],
        }, ts=t)
        actions = mac.select_actions(batch, t_ep=t, t_env=0, test_mode=True)
        _, reward, done, truncated, _ = env.step(actions[0].cpu().numpy())
        terminated = bool(done) or bool(truncated)
        batch.update({
            "actions":    actions,
            "reward":     [[reward]],
            "terminated": [[terminated]],
        }, ts=t)
        ep_return += reward
        t += 1
    return ep_return


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint under observation attack")
    parser.add_argument("--config",     type=Path, default=None,
                        help="Path to attack YAML (alternative to CLI args)")
    parser.add_argument("--algo",       choices=["mappo"])
    parser.add_argument("--sharing",    choices=["shared", "independent"])
    parser.add_argument("--env",        choices=list(ENV_MAP))
    parser.add_argument("--seed",       type=int)
    parser.add_argument("--attack",     choices=["no_attack", "random_noise", "fgsm", "sdor_stor"])
    parser.add_argument("--epsilon",    type=float, default=None)
    parser.add_argument("--n_episodes", type=int,   default=None)
    parser.add_argument("--sdor_ckpt",  default=None,
                        help="Path to trained SDor checkpoint dir (required for sdor_stor)")
    args = parser.parse_args()

    # If --config is given, load YAML and fill in any args not set on CLI.
    # CLI args take precedence; YAML fills the gaps.
    if args.config is not None:
        if not args.config.exists():
            parser.error(f"Attack YAML not found: {args.config}")
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for key in ("algo", "sharing", "env", "seed", "attack",
                    "epsilon", "n_episodes", "sdor_ckpt"):
            if getattr(args, key) is None and key in cfg:
                setattr(args, key, cfg[key])

    # Apply defaults for anything still unset (after CLI + YAML)
    if args.epsilon    is None: args.epsilon    = 0.0
    if args.n_episodes is None: args.n_episodes = 100

    # Validate required fields
    missing = [k for k in ("algo", "sharing", "env", "seed", "attack")
               if getattr(args, k) is None]
    if missing:
        parser.error(f"Missing required field(s): {', '.join(missing)} "
                     "(provide via --config YAML or CLI args)")

    env_info_map = ENV_MAP[args.env]
    env_key      = env_info_map["key"]
    algo_config  = env_info_map["algo_config"][args.sharing]

    # --- checkpoint ---
    ckpt_path = find_checkpoint(REPO_ROOT, args.env, args.algo, args.sharing, args.seed)
    print(f"Checkpoint: {ckpt_path}")

    # --- sacred config -> args namespace ---
    config_dict = find_sacred_config(
        REPO_ROOT, algo_config, env_key, args.seed,
        args.env, args.algo, args.sharing,
    )
    args_ns = SimpleNamespace(**config_dict)
    args_ns.device = "cuda" if th.cuda.is_available() else "cpu"

    # --- inner env (no wrapper yet — need env_info before building MAC) ---
    inner_env = env_REGISTRY["gymma"](
        key=env_key,
        time_limit=env_info_map["time_limit"],
        seed=args.seed,
        common_reward=args_ns.common_reward,
        reward_scalarisation=args_ns.reward_scalarisation,
        pretrained_wrapper=None,
    )
    env_info = inner_env.get_env_info()

    # populate runtime fields not stored in Sacred config
    args_ns.n_agents  = env_info["n_agents"]
    args_ns.n_actions = env_info["n_actions"]

    # --- scheme / groups / MAC ---
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

    # --- build attack + wrap env (FGSM/SDor need MAC reference) ---
    attack_fn = build_attack(
        args.attack, args.epsilon, mac, args_ns, args_ns.device,
        sdor_ckpt=args.sdor_ckpt,
    )
    env = ObsPerturbWrapper(inner_env, attack_fn)

    # --- eval loop ---
    print(f"Running {args.n_episodes} episodes | attack={args.attack} epsilon={args.epsilon}")
    returns = []
    for ep in range(args.n_episodes):
        r = run_episode(env, mac, scheme, groups, preprocess, args_ns)
        returns.append(r)
        if (ep + 1) % 10 == 0:
            print(f"  [{ep+1}/{args.n_episodes}] running mean={np.mean(returns):.2f}")

    mean_r = float(np.mean(returns))
    std_r  = float(np.std(returns))
    print(f"\nMean return: {mean_r:.2f} +/- {std_r:.2f}")

    # --- save ---
    result = {
        "algo": args.algo, "sharing": args.sharing,
        "env": args.env, "seed": args.seed,
        "attack": args.attack, "epsilon": args.epsilon,
        "n_episodes": args.n_episodes,
        "mean_return": mean_r, "std_return": std_r,
        "checkpoint_timestep": int(ckpt_path.name),
    }
    out_dir = REPO_ROOT / "results" / args.env / args.algo / args.sharing / f"seed{args.seed}"
    out_path = out_dir / f"attack_{args.attack}_eps{args.epsilon}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
