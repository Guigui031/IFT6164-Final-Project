import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT  = Path(__file__).resolve().parent
SACRED_DIR = REPO_ROOT / "epymarl" / "results" / "sacred"

MAC_TO_SHARING = {
    "basic_mac":       "shared",
    "non_shared_mac":  "independent",
    "maddpg_controller": "independent",
}

REWARD_METRIC = "test_return_mean"

LOSS_METRICS = {
    "mappo":  ["pg_loss", "critic_loss"],
    "ippo":   ["pg_loss", "critic_loss"],
    "iql":    ["loss"],
    "qmix":   ["loss"],
    "vdn":    ["loss"],
    "maddpg": ["loss"],
}

COLORS = {"shared": "#2196F3", "independent": "#FF5722"}


def load_runs(algo: str, env_filter: str | None):
    """Return dict[sharing -> list[metrics_dict]] from Sacred JSON files."""
    target_names = {algo, f"{algo}_ns"}
    runs = defaultdict(list)

    for metrics_path in SACRED_DIR.glob("**/metrics.json"):
        config_path = metrics_path.parent / "config.json"
        if not config_path.exists():
            continue
        try:
            config  = json.loads(config_path.read_text())
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue

        if config.get("name", "") not in target_names:
            continue
        env_key = config.get("env_args", {}).get("key", "")
        if env_filter and env_filter.replace("_", "-") not in env_key:
            continue

        sharing = MAC_TO_SHARING.get(config.get("mac", ""), "unknown")
        runs[sharing].append(metrics)

    return runs


def aggregate(runs_list: list, metric: str):
    """Compute mean ± std across seeds for one metric. Returns (steps, means, stds) or Nones."""
    series = []
    for run in runs_list:
        if metric not in run:
            continue
        series.append((np.array(run[metric]["steps"]), np.array(run[metric]["values"])))

    if not series:
        return None, None, None

    max_len = max(len(s) for _, s in series)
    all_steps, all_vals = [], []
    for steps, vals in series:
        if len(steps) < max_len:
            pad = max_len - len(steps)
            steps = np.concatenate([steps, np.full(pad, np.nan)])
            vals  = np.concatenate([vals,  np.full(pad, np.nan)])
        all_steps.append(steps)
        all_vals.append(vals)

    steps = np.nanmean(np.stack(all_steps), axis=0)
    vals  = np.stack(all_vals)
    return steps, np.nanmean(vals, axis=0), np.nanstd(vals, axis=0)


def _smooth(arr: np.ndarray, window: int | None) -> np.ndarray:
    if not window or window <= 1:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def plot_metric(ax, runs: dict, metric: str, title: str, smoothing: int | None) -> bool:
    found = False
    for sharing in ("shared", "independent"):
        if sharing not in runs:
            continue
        steps, means, stds = aggregate(runs[sharing], metric)
        if steps is None:
            continue
        found = True
        color = COLORS[sharing]
        n = len(steps) - (smoothing or 1) + 1 if smoothing else len(steps)
        s = steps[:n]
        m = _smooth(means, smoothing)
        d = _smooth(stds,  smoothing)
        label = f"{sharing} (n={len(runs[sharing])})"
        ax.plot(s, m, label=label, color=color)
        ax.fill_between(s, m - d, m + d, alpha=0.2, color=color)

    ax.set_title(title)
    ax.set_xlabel("Timesteps")
    if found:
        ax.legend()
    return found


def main():
    parser = argparse.ArgumentParser(
        description="Plot reward + loss curves for shared vs independent from Sacred logs"
    )
    parser.add_argument("--algo", required=True,
                        choices=["iql", "ippo", "mappo", "qmix", "vdn", "maddpg"])
    parser.add_argument("--env", default=None,
                        help="Filter runs by env key substring (e.g. simple_spread)")
    parser.add_argument("--out", default="figures",
                        help="Directory to write PDF plots")
    parser.add_argument("--smoothing", type=int, default=None,
                        help="Rolling average window size")
    args = parser.parse_args()

    runs = load_runs(args.algo, args.env)
    if not runs:
        print(f"No Sacred runs found for algo='{args.algo}' under {SACRED_DIR}")
        print("Run exp_train.py first, then try again.")
        sys.exit(1)

    for sharing, run_list in runs.items():
        print(f"  {sharing}: {len(run_list)} run(s) found")

    loss_metrics = LOSS_METRICS.get(args.algo, ["loss"])
    ncols = 1 + len(loss_metrics)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    env_tag = args.env or "all"
    plot_metric(axes[0], runs, REWARD_METRIC,
                f"{args.algo} — test return ({env_tag})", args.smoothing)
    axes[0].set_ylabel("Mean episode return")

    for i, metric in enumerate(loss_metrics):
        plot_metric(axes[1 + i], runs, metric,
                    f"{args.algo} — {metric}", args.smoothing)
        axes[1 + i].set_ylabel(metric)

    plt.tight_layout()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.algo}_{env_tag}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
