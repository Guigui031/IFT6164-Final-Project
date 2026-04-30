"""Train ONE (algo, sharing, seed) cell, either from a YAML config or CLI args.

YAML mode (recommended):
  python exp_train.py --config experiments/train/mappo_shared_seed1.yaml

CLI mode (backward-compat):
  python exp_train.py --algo mappo --sharing shared --env mpe_simple_spread --seed 1

Skip-if-done: exits with [SKIP] if a checkpoint already exists for the cell.
While EPyMARL trains, a live progress PNG is refreshed at training_plots/.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

REPO_ROOT    = Path(__file__).resolve().parent
PYTHON       = sys.executable
EPYMARL_DIR  = REPO_ROOT / "epymarl"
EPYMARL_MAIN = EPYMARL_DIR / "src" / "main.py"


# CLI-mode env map. env_args is the literal dict passed to sacred's `env_args`;
# its shape varies per env_config (gymma uses "key", smaclite uses "map_name").
ENV_MAP = {
    "mpe_simple_spread": {
        "env_config":    "gymma",
        "env_args":      {"key": "pz-mpe-simple-spread-v3", "time_limit": 25},
        "default_t_max": 1_050_000,
    },
    "mpe_simple_reference": {
        "env_config":    "gymma",
        "env_args":      {"key": "pz-mpe-simple-reference-v3", "time_limit": 25},
        "default_t_max": 1_050_000,
    },
    "smaclite_2s_vs_1sc": {
        "env_config":    "smaclite",
        "env_args":      {"map_name": "2s_vs_1sc", "time_limit": 150, "use_cpp_rvo2": False},
        "default_t_max": 500_000,
    },
}

# Sacred overrides to neutralise confounds when the _ns variant defaults differ
# from the shared variant's defaults (CLAUDE.md sharing-toggle table).
EQUALISATION_OVERRIDES = {
    ("iql",   "independent"): {"use_rnn": "True", "epsilon_anneal_time": "200000"},
    ("ippo",  "independent"): {"use_rnn": "True"},
    ("mappo", "independent"): {"use_rnn": "True"},
}


# ---------------------------------------------------------------------------
# Helpers (duplicated from scripts/run_sweep.py — refactor into shared module
# if a third caller appears)
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _sacred_env_subdir(env_config: str, env_args: dict) -> str:
    if env_config == "gymma":
        return env_args["key"]
    if env_config in ("smaclite", "sc2", "sc2v2"):
        return env_args["map_name"]
    raise ValueError(f"Unknown env config: {env_config}")


def _find_checkpoint(env_name: str, algo: str, sharing: str, seed: int) -> Path | None:
    models_root = REPO_ROOT / "results" / env_name / algo / sharing / f"seed{seed}" / "models"
    valid = [p.parent for p in models_root.glob("**/agent.th") if p.parent.name.isdigit()]
    return max(valid, key=lambda p: int(p.name)) if valid else None


def _save_team_plot_png(metrics_dict: dict, path: Path, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, ylabel in [
        (axes[0], "test_return_mean", "Mean return"),
        (axes[1], "pg_loss",          "Actor (PG) loss"),
        (axes[2], "critic_loss",      "Critic loss"),
    ]:
        if key in metrics_dict:
            ax.plot(metrics_dict[key]["steps"], metrics_dict[key]["values"])
        ax.set_xlabel("Timesteps")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _poll_and_plot_team(proc, sacred_config: str, sacred_env_subdir: str,
                        sharing: str, seed: int,
                        plot_path: Path, interval_sec: int):
    sacred_base = EPYMARL_DIR / "results" / "sacred" / sacred_config / sacred_env_subdir
    # Only numbered run dirs — sacred also creates a _sources subdir we must skip.
    known = (
        {d.name for d in sacred_base.glob("*/") if d.is_dir() and d.name.isdigit()}
        if sacred_base.exists() else set()
    )
    metrics_path: Path | None = None
    title = f"{sacred_config} {sharing} seed{seed}"

    def _refresh():
        if metrics_path and metrics_path.exists():
            try:
                _save_team_plot_png(
                    json.loads(metrics_path.read_text()), plot_path, title
                )
            except Exception:
                pass

    while proc.poll() is None:
        time.sleep(interval_sec)
        if metrics_path is None and sacred_base.exists():
            new = [
                d for d in sacred_base.glob("*/")
                if d.is_dir() and d.name not in known and d.name.isdigit()
            ]
            if new:
                metrics_path = max(new, key=lambda d: int(d.name)) / "metrics.json"
        _refresh()

    _refresh()


def _get_epymarl_config(algo: str, sharing: str) -> str:
    if algo == "maddpg":
        if sharing == "shared":
            raise ValueError(
                "MADDPG has no shared variant — see extended abstract §3. "
                "Use --sharing independent."
            )
        return "maddpg_ns"
    return algo if sharing == "shared" else f"{algo}_ns"


# ---------------------------------------------------------------------------
# Param resolution: YAML and CLI both flow into the same canonical dict
# ---------------------------------------------------------------------------

def _params_from_yaml(yaml_path: Path) -> dict:
    cfg = _load_yaml(yaml_path)
    overrides = dict(cfg.get("overrides", {}))
    return {
        "env_name":          cfg["name"],
        "env_config":        cfg["env"]["config"],
        "env_args":          cfg["env"]["args"],
        "sacred_config":     cfg["algo"]["sacred_config"],
        "sharing":           cfg["algo"]["sharing"],
        "algo":              cfg["algo"].get("name", "mappo"),
        "seed":              cfg["seed"],
        "t_max":             cfg["t_max"],
        "overrides":         overrides,
        "plot_interval_sec": cfg.get("plot_interval_sec", 60),
    }


def _params_from_cli(args) -> dict:
    env_info      = ENV_MAP[args.env]
    sacred_config = _get_epymarl_config(args.algo, args.sharing)
    overrides = dict(EQUALISATION_OVERRIDES.get((args.algo, args.sharing), {}))
    overrides["save_model"]          = "True"
    overrides["save_model_interval"] = str(args.save_model_interval)
    if args.algo == "mappo" and args.sharing == "shared":
        overrides["obs_agent_id"] = "True"
    env_args = dict(env_info["env_args"])
    if args.time_limit is not None:
        env_args["time_limit"] = args.time_limit
    return {
        "env_name":          args.env,
        "env_config":        env_info["env_config"],
        "env_args":          env_args,
        "sacred_config":     sacred_config,
        "sharing":           args.sharing,
        "algo":              args.algo,
        "seed":              args.seed,
        "t_max":             args.t_max or env_info["default_t_max"],
        "overrides":         overrides,
        "plot_interval_sec": 60,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train one EPyMARL (algo x env x sharing x seed) cell",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to training YAML (alternative to CLI args)")
    parser.add_argument("--algo",    choices=["iql", "ippo", "mappo", "qmix", "vdn", "maddpg"])
    parser.add_argument("--sharing", choices=["shared", "independent"])
    parser.add_argument("--env",     choices=list(ENV_MAP))
    parser.add_argument("--seed",    type=int)
    parser.add_argument("--t_max",   type=int, default=None,
                        help="Override t_max (default: env's default_t_max)")
    parser.add_argument("--time_limit", type=int, default=None,
                        help="Override episode time_limit")
    parser.add_argument("--save_model_interval", type=int, default=500_000)
    args = parser.parse_args()

    if args.config is not None:
        params = _params_from_yaml(args.config)
    else:
        if not (args.algo and args.sharing and args.env and args.seed is not None):
            parser.error("CLI mode requires --algo, --sharing, --env, --seed")
        params = _params_from_cli(args)

    label = f"{params['algo']}/{params['sharing']}/seed{params['seed']}"
    if _find_checkpoint(params["env_name"], params["algo"],
                        params["sharing"], params["seed"]):
        print(f"[SKIP] {label} -- checkpoint already exists")
        return

    out_dir = (
        REPO_ROOT / "results" / params["env_name"]
        / params["algo"] / params["sharing"] / f"seed{params['seed']}"
    )
    plot_path = (
        REPO_ROOT / "training_plots"
        / f"{params['env_name']}_{params['sharing']}_seed{params['seed']}.png"
    )
    print(f"[RUN]  {label}")

    sacred_args = (
        [f"env_args.{k}={v}" for k, v in params["env_args"].items()]
        + [f"t_max={params['t_max']}", f"seed={params['seed']}",
           f"local_results_path={out_dir.as_posix()}"]
        + [f"{k}={v}" for k, v in params["overrides"].items()]
    )
    cmd = [
        PYTHON, str(EPYMARL_MAIN),
        f"--config={params['sacred_config']}",
        f"--env-config={params['env_config']}",
        "with",
    ] + sacred_args
    print(f"    $ {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, cwd=str(EPYMARL_DIR))
    _poll_and_plot_team(
        proc, params["sacred_config"],
        _sacred_env_subdir(params["env_config"], params["env_args"]),
        params["sharing"], params["seed"],
        plot_path, params["plot_interval_sec"],
    )
    proc.wait()
    if proc.returncode != 0:
        sys.exit(proc.returncode)

    ckpt = _find_checkpoint(params["env_name"], params["algo"],
                            params["sharing"], params["seed"])
    print(f"\nDone. Checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
