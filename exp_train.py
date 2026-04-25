import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
EPYMARL_DIR = REPO_ROOT / "epymarl"
EPYMARL_MAIN = EPYMARL_DIR / "src" / "main.py"

# Sacred overrides needed to equalize _ns configs against their shared counterparts.
# The _ns YAML files change more than just sharing (e.g. use_rnn: False), so we
# re-inject the shared defaults here to keep the ablation clean.
EQUALISATION_OVERRIDES = {
    ("iql",   "independent"): ["use_rnn=True", "epsilon_anneal_time=200000"],
    ("ippo",  "independent"): ["use_rnn=True"],
    ("mappo", "independent"): ["use_rnn=True"],
    # qmix, vdn: _ns defaults are already equivalent to shared — no override needed
    # maddpg: independent-only; used as-is
}

ENV_MAP = {
    "mpe_simple_spread": {
        "key": "pz-mpe-simple-spread-v3",
        "env_config": "gymma",
        "default_time_limit": 25,
        "default_t_max": 2050000,
    },
}


def get_epymarl_config(algo: str, sharing: str) -> str:
    if algo == "maddpg":
        if sharing == "shared":
            raise ValueError(
                "MADDPG has no shared variant — see extended abstract §3. "
                "Use --sharing independent."
            )
        return "maddpg_ns"
    return algo if sharing == "shared" else f"{algo}_ns"


def find_python() -> str:
    candidates = [
        REPO_ROOT / "venv"  / "Scripts" / "python.exe",
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
        REPO_ROOT / "venv"  / "bin"     / "python",
        REPO_ROOT / ".venv" / "bin"     / "python",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    print(f"WARNING: no venv python found under {REPO_ROOT}; falling back to {sys.executable}")
    return sys.executable


def main():
    parser = argparse.ArgumentParser(description="Train one EPyMARL (algo × env × sharing × seed) cell")
    parser.add_argument("--algo", required=True,
                        choices=["iql", "ippo", "mappo", "qmix", "vdn", "maddpg"])
    parser.add_argument("--sharing", required=True, choices=["shared", "independent"])
    parser.add_argument("--env", required=True, choices=list(ENV_MAP))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--t_max", type=int, default=None,
                        help="Override t_max (default: env's default_t_max)")
    parser.add_argument("--time_limit", type=int, default=None,
                        help="Override episode time_limit (default: 25 for mpe_simple_spread)")
    parser.add_argument("--save_model_interval", type=int, default=500000)
    args = parser.parse_args()

    env_info = ENV_MAP[args.env]
    config   = get_epymarl_config(args.algo, args.sharing)
    t_max       = args.t_max      or env_info["default_t_max"]
    time_limit  = args.time_limit or env_info["default_time_limit"]

    results_dir = (
        REPO_ROOT / "results" / args.env / args.algo / args.sharing / f"seed{args.seed}"
    )
    overrides = EQUALISATION_OVERRIDES.get((args.algo, args.sharing), [])

    cmd = [
        find_python(),
        str(EPYMARL_MAIN),
        f"--config={config}",
        f"--env-config={env_info['env_config']}",
        "with",
        f"env_args.key={env_info['key']}",
        f"env_args.time_limit={time_limit}",
        f"t_max={t_max}",
        f"seed={args.seed}",
        f"local_results_path={results_dir.as_posix()}",
        "save_model=True",
        f"save_model_interval={args.save_model_interval}",
    ] + overrides

    print("=" * 72)
    print("EPyMARL command:")
    print("  " + " ".join(cmd))
    print(f"  cwd:         {EPYMARL_DIR}")
    print(f"  checkpoints: {results_dir}/models/")
    print(f"  Sacred JSON: epymarl/results/sacred/{config}/*/")
    print("=" * 72)

    result = subprocess.run(cmd, cwd=str(EPYMARL_DIR))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
