"""Thin wrapper around EPyMARL's src/main.py for one training cell.

Usage (from repo root, with venv activated):

    python exp_train.py --algo iql --sharing shared --seed 1
    python exp_train.py --algo qmix --sharing independent --seed 3 --t_max 1000000

Responsibilities:
  * Map (algo, sharing) to the correct EPyMARL --config= value.
  * Auto-inject confound-equalization overrides so the shared-vs-independent
    ablation is clean (see CLAUDE.md -> "Sharing toggle").
  * Normalize output paths to results/<env>/<algo>/<sharing>/seed<N>/.
  * Stream stdout+stderr into that dir so the sweep is restartable.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
EPYMARL_DIR = REPO_ROOT / "epymarl"
VENV_PY = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

ALGOS = ["iql", "ippo", "mappo", "qmix", "vdn"]  # discrete-action MPE algos; MADDPG deferred
SHARING = ["shared", "independent"]
ENVS = {
    "mpe_simple_spread": {
        "env_config": "gymma",
        "env_args": {
            "time_limit": 25,
            "key": "pz-mpe-simple-spread-v3",
        },
    },
}

# Per-algorithm overrides needed to neutralize non-sharing differences between
# EPyMARL's default <algo>.yaml and <algo>_ns.yaml. See CLAUDE.md.
# These are applied ONLY to the `independent` variant.
NS_OVERRIDES = {
    "iql":   {"use_rnn": True, "epsilon_anneal_time": 200000},
    "ippo":  {"use_rnn": True},
    "mappo": {"use_rnn": True},
    "qmix":  {},  # already clean (shared and ns both use use_rnn=False)
    "vdn":   {},  # already clean
}


def build_command(algo: str, sharing: str, seed: int, env: str,
                  t_max: int | None, extra: list[str]) -> tuple[list[str], Path]:
    env_spec = ENVS[env]
    epymarl_config = algo if sharing == "shared" else f"{algo}_ns"

    out_dir = REPO_ROOT / "results" / env / algo / sharing / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_label = f"{algo}_{sharing}_{env}_seed{seed}"
    cmd = [
        str(VENV_PY),
        "src/main.py",
        f"--config={epymarl_config}",
        f"--env-config={env_spec['env_config']}",
        "with",
        f"seed={seed}",
        f"label={unique_label}",
        "save_model=True",
        "use_tensorboard=True",
        # Redirect EPyMARL's result dir to ours so checkpoints, TB logs, and
        # sacred runs land under our results/ tree instead of epymarl/results/.
        f"local_results_path={(out_dir / 'epymarl').as_posix()}",
    ]
    # subprocess.run takes a list of args directly (no shell), so no quoting
    # is needed — sacred parses bare tokens like `env_args.key=pz-mpe-simple-spread-v3`.
    for k, v in env_spec["env_args"].items():
        cmd.append(f"env_args.{k}={v}")

    if t_max is not None:
        cmd.append(f"t_max={t_max}")

    # NS overrides: equalize architecture / hyperparameters between shared and
    # independent so the only thing that differs is parameter sharing.
    if sharing == "independent":
        for k, v in NS_OVERRIDES.get(algo, {}).items():
            cmd.append(f"{k}={v}")

    cmd.extend(extra)
    return cmd, out_dir


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", required=True, choices=ALGOS)
    p.add_argument("--sharing", required=True, choices=SHARING)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--env", default="mpe_simple_spread", choices=list(ENVS))
    p.add_argument("--t_max", type=int, default=None,
                   help="Override t_max (total env steps). Default uses EPyMARL's env config value.")
    p.add_argument("--dry_run", action="store_true", help="Print the command and exit without running.")
    p.add_argument("extra", nargs="*",
                   help="Extra sacred overrides passed through as-is (e.g. 'batch_size_run=4').")
    args = p.parse_args()

    if not VENV_PY.exists():
        sys.exit(f"venv python not found at {VENV_PY}. Run setup per README.md.")

    cmd, out_dir = build_command(args.algo, args.sharing, args.seed,
                                  args.env, args.t_max, args.extra)

    print(f"[exp_train] out_dir = {out_dir}")
    print(f"[exp_train] cmd = {' '.join(cmd)}")

    if args.dry_run:
        return 0

    log_path = out_dir / "train.log"
    print(f"[exp_train] streaming stdout+stderr to {log_path}")

    # cwd must be epymarl/ because main.py uses relative imports and file paths.
    with open(log_path, "wb") as log:
        proc = subprocess.run(cmd, cwd=EPYMARL_DIR, stdout=log, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        print(f"[exp_train] FAILED with exit code {proc.returncode}. Tail of log:")
        with open(log_path) as f:
            print("".join(f.readlines()[-30:]))
    else:
        print(f"[exp_train] OK. Tail of log:")
        with open(log_path) as f:
            print("".join(f.readlines()[-6:]))

    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
