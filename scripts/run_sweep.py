"""Orchestrate the full MAPPO sweep: train teams -> train SDor -> eval attacks.

Phases:
  train   -- train MAPPO shared + independent for all requested seeds
  sdor    -- train one SDor per seed (against the shared protagonist)
  attack  -- run all attacks x all epsilons x all seeds x both sharings
  all     -- run all three phases in order

Skip-if-done: each step checks whether its output file already exists before
launching a subprocess. Interrupt and re-run at any time; completed work is
never redone.

Usage examples:
  python scripts/run_sweep.py --phase all --seeds 1 2 3 --epsilons 0.05 0.1 0.2
  python scripts/run_sweep.py --phase train --seeds 2 3
  python scripts/run_sweep.py --phase attack --seeds 1 2 3 --epsilons 0.05 0.1 0.2
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

REPO_ROOT  = Path(__file__).resolve().parent.parent
PYTHON     = sys.executable
EPYMARL    = REPO_ROOT / "epymarl"

ENV        = "mpe_simple_spread"
ENV_KEY    = "pz-mpe-simple-spread-v3"
TIME_LIMIT = 25
T_MAX      = 1_000_000
N_EVAL_EPS = 1000

ATTACKS = ["no_attack", "random_noise", "fgsm", "sdor_stor"]

# EPyMARL config per sharing variant, plus Sacred overrides to neutralise confounds
# (use_rnn=True equalises RNN usage; see CLAUDE.md sharing-toggle section)
TRAIN_CONFIGS = [
    ("mappo",    "shared",      {"use_rnn": "True", "obs_agent_id": "True"}),
    ("mappo_ns", "independent", {"use_rnn": "True"}),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd, **kwargs):
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def find_checkpoint(env: str, sharing: str, seed: int) -> Path | None:
    models_root = REPO_ROOT / "results" / env / "mappo" / sharing / f"seed{seed}" / "models"
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


def _poll_and_plot_team(proc, cfg_name: str, sharing: str, seed: int,
                        plot_path: Path, interval_sec: int):
    """Poll Sacred metrics.json while EPyMARL runs and refresh the PNG."""
    sacred_base = EPYMARL / "results" / "sacred" / cfg_name / ENV_KEY
    known = (
        {d.name for d in sacred_base.glob("*/") if d.is_dir()}
        if sacred_base.exists() else set()
    )
    metrics_path: Path | None = None
    title = f"MAPPO {sharing} seed{seed}"

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

def phase_train(seeds: list[int], plot_interval_sec: int):
    print("\n=== PHASE: train ===")
    for cfg_name, sharing, overrides in TRAIN_CONFIGS:
        for seed in seeds:
            label = f"mappo/{sharing}/seed{seed}"
            if find_checkpoint(ENV, sharing, seed):
                print(f"[SKIP] {label}")
                continue
            out_dir = REPO_ROOT / "results" / ENV / "mappo" / sharing / f"seed{seed}"
            plot_path = REPO_ROOT / "training_plots" / f"mappo_{sharing}_seed{seed}.png"
            print(f"[RUN]  {label}")
            sacred_args = [
                f"env_args.time_limit={TIME_LIMIT}",
                f"env_args.key={ENV_KEY}",
                f"t_max={T_MAX}",
                f"seed={seed}",
                f"local_results_path={out_dir.as_posix()}",
            ] + [f"{k}={v}" for k, v in overrides.items()]
            cmd = [
                PYTHON, "src/main.py",
                f"--config={cfg_name}", "--env-config=gymma",
                "with",
            ] + sacred_args
            print(f"    $ {' '.join(str(c) for c in cmd)}")
            proc = subprocess.Popen([str(c) for c in cmd], cwd=str(EPYMARL))
            _poll_and_plot_team(
                proc, cfg_name, sharing, seed, plot_path, plot_interval_sec
            )
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)


# ---------------------------------------------------------------------------
# Phase: sdor
# ---------------------------------------------------------------------------

def phase_sdor(seeds: list[int], sdor_epsilon: float, n_train_episodes: int,
               plot_interval: int):
    print("\n=== PHASE: sdor ===")
    for seed in seeds:
        label     = f"sdor/seed{seed}"
        sdor_pt   = REPO_ROOT / "results" / "sdor" / ENV / f"seed{seed}" / "sdor.pt"
        if sdor_pt.exists():
            print(f"[SKIP] {label}")
            continue
        if not find_checkpoint(ENV, "shared", seed):
            print(f"[WAIT] {label} -- shared checkpoint missing; run --phase train first")
            continue
        print(f"[RUN]  {label}")
        _run([
            PYTHON, REPO_ROOT / "exp_sdor_train.py",
            "--env", ENV,
            "--seed", str(seed),
            "--epsilon", str(sdor_epsilon),
            "--n_train_episodes", str(n_train_episodes),
            "--plot_interval", str(plot_interval),
        ])


# ---------------------------------------------------------------------------
# Phase: attack
# ---------------------------------------------------------------------------

def phase_attack(seeds: list[int], epsilons: list[float]):
    print("\n=== PHASE: attack ===")
    for sharing, seed in product(("shared", "independent"), seeds):
        if not find_checkpoint(ENV, sharing, seed):
            print(f"[WAIT] {sharing}/seed{seed} -- checkpoint missing; run --phase train first")
            continue

        seed_dir = REPO_ROOT / "results" / ENV / "mappo" / sharing / f"seed{seed}"
        sdor_ckpt = REPO_ROOT / "results" / "sdor" / ENV / f"seed{seed}"

        for attack in ATTACKS:
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
                    "--env",        ENV,
                    "--seed",       str(seed),
                    "--attack",     attack,
                    "--epsilon",    str(eps),
                    "--n_episodes", str(N_EVAL_EPS),
                ]
                if attack == "sdor_stor":
                    cmd += ["--sdor_ckpt", str(sdor_ckpt)]
                _run(cmd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate MAPPO sweep: train -> sdor -> attack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase", default="all",
                        choices=["all", "train", "sdor", "attack"],
                        help="Which phase(s) to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3],
                        help="Seeds to process")
    parser.add_argument("--epsilons", nargs="+", type=float,
                        default=[0.0, 0.05, 0.1, 0.2],
                        help="Epsilon values for the attack sweep (0.0 = clean baseline)")
    parser.add_argument("--sdor_epsilon", type=float, default=0.1,
                        help="Perturbation budget used when training SDor")
    parser.add_argument("--n_sdor_episodes", type=int, default=7_000,
                        help="Training episodes for each SDor run")
    parser.add_argument("--plot_interval_sec", type=int, default=60,
                        help="Seconds between team-training plot refreshes")
    parser.add_argument("--plot_interval", type=int, default=500,
                        help="Episodes between SDor plot refreshes")
    args = parser.parse_args()

    if args.phase in ("all", "train"):
        phase_train(args.seeds, args.plot_interval_sec)
    if args.phase in ("all", "sdor"):
        phase_sdor(args.seeds, args.sdor_epsilon, args.n_sdor_episodes,
                   args.plot_interval)
    if args.phase in ("all", "attack"):
        phase_attack(args.seeds, args.epsilons)

    print("\nDone.")


if __name__ == "__main__":
    main()
