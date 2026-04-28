"""Orchestrate a full MARL sweep from a YAML experiment config.

A sweep YAML (under experiments/sweeps/) defines:
  - env: which environment + EPyMARL env-config + env_args
  - train: t_max + a list of (sharing, sacred_config, overrides) variants
  - sdor: SDor adversary training params
  - attack: list of attacks, list of epsilons, n_episodes
  - seeds: list of seeds to run
  - plot: live-PNG refresh intervals

Skip-if-done: every step checks whether its output file already exists before
launching a subprocess. Interrupt and re-run at any time; completed work is
never redone.

Usage:
  python scripts/run_sweep.py --config experiments/sweeps/mpe_simple_spread.yaml --phase all
  python scripts/run_sweep.py --config experiments/sweeps/mpe_simple_spread.yaml --phase attack --seeds 1
"""
import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON    = sys.executable
EPYMARL   = REPO_ROOT / "epymarl"


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_experiment(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Experiment YAML not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _sacred_env_subdir(env_cfg: dict) -> str:
    """Return the directory name Sacred uses to namespace runs by env."""
    if env_cfg["config"] == "gymma":
        return env_cfg["args"]["key"]
    if env_cfg["config"] in ("smaclite", "sc2", "sc2v2"):
        return env_cfg["args"]["map_name"]
    raise ValueError(f"Unknown env config: {env_cfg['config']}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd, **kwargs):
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def find_checkpoint(env_name: str, sharing: str, seed: int) -> Path | None:
    models_root = REPO_ROOT / "results" / env_name / "mappo" / sharing / f"seed{seed}" / "models"
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
    """Poll Sacred metrics.json while EPyMARL runs and refresh the PNG."""
    sacred_base = EPYMARL / "results" / "sacred" / sacred_config / sacred_env_subdir
    known = (
        {d.name for d in sacred_base.glob("*/") if d.is_dir()}
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
                pass  # file may be partially written between Sacred flushes

    while proc.poll() is None:
        time.sleep(interval_sec)
        if metrics_path is None and sacred_base.exists():
            new = [
                d for d in sacred_base.glob("*/")
                if d.is_dir() and d.name not in known
            ]
            if new:
                metrics_path = max(new, key=lambda d: int(d.name)) / "metrics.json"
        _refresh()

    _refresh()  # final update after the process exits


# ---------------------------------------------------------------------------
# Phase: train
# ---------------------------------------------------------------------------

def phase_train(exp: dict, seeds: list[int]):
    print("\n=== PHASE: train ===")
    env_name          = exp["name"]
    env_cfg           = exp["env"]
    t_max             = exp["train"]["t_max"]
    plot_interval_sec = exp.get("plot", {}).get("team_interval_sec", 60)
    sacred_env_subdir = _sacred_env_subdir(env_cfg)

    for variant in exp["train"]["variants"]:
        sacred_config = variant["sacred_config"]
        sharing       = variant["sharing"]
        overrides     = variant["overrides"]
        for seed in seeds:
            label = f"mappo/{sharing}/seed{seed}"
            if find_checkpoint(env_name, sharing, seed):
                print(f"[SKIP] {label}")
                continue
            out_dir   = REPO_ROOT / "results" / env_name / "mappo" / sharing / f"seed{seed}"
            plot_path = REPO_ROOT / "training_plots" / f"{env_name}_{sharing}_seed{seed}.png"
            print(f"[RUN]  {label}")
            sacred_args = (
                [f"env_args.{k}={v}" for k, v in env_cfg["args"].items()]
                + [f"t_max={t_max}", f"seed={seed}",
                   f"local_results_path={out_dir.as_posix()}"]
                + [f"{k}={v}" for k, v in overrides.items()]
            )
            cmd = [
                PYTHON, "src/main.py",
                f"--config={sacred_config}",
                f"--env-config={env_cfg['config']}",
                "with",
            ] + sacred_args
            print(f"    $ {' '.join(str(c) for c in cmd)}")
            proc = subprocess.Popen([str(c) for c in cmd], cwd=str(EPYMARL))
            _poll_and_plot_team(
                proc, sacred_config, sacred_env_subdir, sharing, seed,
                plot_path, plot_interval_sec,
            )
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)


# ---------------------------------------------------------------------------
# Phase: sdor
# ---------------------------------------------------------------------------

def phase_sdor(exp: dict, seeds: list[int]):
    print("\n=== PHASE: sdor ===")
    env_name         = exp["name"]
    sdor_cfg         = exp["sdor"]
    sdor_epsilon     = sdor_cfg["epsilon"]
    n_train_episodes = sdor_cfg["n_train_episodes"]
    plot_interval    = sdor_cfg.get("plot_interval", 500)

    for seed in seeds:
        label   = f"sdor/seed{seed}"
        sdor_pt = REPO_ROOT / "results" / "sdor" / env_name / f"seed{seed}" / "sdor.pt"
        if sdor_pt.exists():
            print(f"[SKIP] {label}")
            continue
        if not find_checkpoint(env_name, "shared", seed):
            print(f"[WAIT] {label} -- shared checkpoint missing; run --phase train first")
            continue
        print(f"[RUN]  {label}")
        _run([
            PYTHON, REPO_ROOT / "exp_sdor_train.py",
            "--env",              env_name,
            "--seed",             str(seed),
            "--epsilon",          str(sdor_epsilon),
            "--n_train_episodes", str(n_train_episodes),
            "--plot_interval",    str(plot_interval),
        ])


# ---------------------------------------------------------------------------
# Phase: attack
# ---------------------------------------------------------------------------

def phase_attack(exp: dict, seeds: list[int]):
    print("\n=== PHASE: attack ===")
    env_name = exp["name"]
    atk_cfg  = exp["attack"]
    attacks  = atk_cfg["attacks"]
    epsilons = atk_cfg["epsilons"]
    n_eps    = atk_cfg["n_episodes"]

    for sharing, seed in product(("shared", "independent"), seeds):
        if not find_checkpoint(env_name, sharing, seed):
            print(f"[WAIT] {sharing}/seed{seed} -- checkpoint missing; run --phase train first")
            continue

        seed_dir  = REPO_ROOT / "results" / env_name / "mappo" / sharing / f"seed{seed}"
        sdor_ckpt = REPO_ROOT / "results" / "sdor" / env_name / f"seed{seed}"

        for attack in attacks:
            eps_list = [0.0] if attack == "no_attack" else [e for e in epsilons if e > 0]
            for eps in eps_list:
                out_json = seed_dir / f"attack_{attack}_eps{eps}.json"
                label    = f"{attack} eps={eps} {sharing}/seed{seed}"

                if out_json.exists():
                    print(f"[SKIP] {label}")
                    continue

                if attack == "sdor_stor" and not (sdor_ckpt / "sdor.pt").exists():
                    print(f"[WAIT] {label} -- SDor checkpoint missing; run --phase sdor first")
                    continue

                print(f"[RUN]  {label}")
                cmd = [
                    PYTHON, REPO_ROOT / "exp_attack.py",
                    "--algo",       "mappo",
                    "--sharing",    sharing,
                    "--env",        env_name,
                    "--seed",       str(seed),
                    "--attack",     attack,
                    "--epsilon",    str(eps),
                    "--n_episodes", str(n_eps),
                ]
                if attack == "sdor_stor":
                    cmd += ["--sdor_ckpt", str(sdor_ckpt)]
                _run(cmd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a sweep defined by a YAML config under experiments/sweeps/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to sweep YAML")
    parser.add_argument("--phase", default="all",
                        choices=["all", "train", "sdor", "attack"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Optional override of the YAML seeds list")
    args = parser.parse_args()

    exp   = load_experiment(args.config)
    seeds = args.seeds if args.seeds is not None else exp["seeds"]

    print(f"Loaded experiment: {exp['name']} (seeds={seeds})")

    if args.phase in ("all", "train"):
        phase_train(exp, seeds)
    if args.phase in ("all", "sdor"):
        phase_sdor(exp, seeds)
    if args.phase in ("all", "attack"):
        phase_attack(exp, seeds)

    print("\nDone.")


if __name__ == "__main__":
    main()
