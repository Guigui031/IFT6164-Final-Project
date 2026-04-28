import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT  = Path(__file__).resolve().parent
SACRED_DIR = REPO_ROOT / "epymarl" / "results" / "sacred"
RESULTS_DIR = REPO_ROOT / "results"

MAC_TO_SHARING = {
    "basic_mac":        "shared",
    "non_shared_mac":   "independent",
    "maddpg_controller":"independent",
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

ATTACK_ORDER  = ["no_attack", "random_noise", "fgsm", "sdor_stor"]


def _apply_style():
    plt.rcParams.update({
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "figure.dpi":        150,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })
ATTACK_LABELS = {
    "no_attack":   "Clean",
    "random_noise": "Random noise",
    "fgsm":        "FGSM",
    "sdor_stor":   "SDor+STor",
}


# ---------------------------------------------------------------------------
# Training-curve mode helpers
# ---------------------------------------------------------------------------

def load_train_runs(algo: str, env_filter: str | None, seed: int | None,
                    min_steps: int = 5, latest: bool = False):
    """Return dict[sharing -> list[metrics_dict]] from Sacred JSON files.

    latest=True: keep only the single longest run per sharing variant
                 (by number of recorded test points, ties broken by Sacred run ID).
    """
    target_names = {algo, f"{algo}_ns"}
    # When latest=True, track (n_points, sacred_id, metrics) per sharing
    candidates = defaultdict(list)

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
        if seed is not None and config.get("seed") != seed:
            continue

        n_points = len(metrics.get(REWARD_METRIC, {}).get("steps", []))
        if n_points < min_steps:
            continue

        sharing   = MAC_TO_SHARING.get(config.get("mac", ""), "unknown")
        sacred_id = int(metrics_path.parent.name)
        candidates[sharing].append((n_points, sacred_id, metrics))

    if latest:
        return {
            sharing: [max(runs, key=lambda x: (x[0], x[1]))[2]]
            for sharing, runs in candidates.items()
        }

    return {sharing: [m for _, _, m in runs] for sharing, runs in candidates.items()}


def aggregate(runs_list: list, metric: str):
    """Mean ± std across runs for one metric. Returns (steps, means, stds) or Nones."""
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


def plot_train_metric(ax, runs: dict, metric: str, title: str, smoothing: int | None) -> bool:
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
        n_runs = len(runs[sharing])
        label = sharing if n_runs == 1 else f"{sharing} (n={n_runs})"
        ax.plot(s, m, label=label, color=color)
        if n_runs > 1:
            ax.fill_between(s, m - d, m + d, alpha=0.2, color=color)

    ax.set_title(title)
    ax.set_xlabel("Timesteps")
    if found:
        ax.legend()
    return found


# ---------------------------------------------------------------------------
# Attack-comparison mode helpers
# ---------------------------------------------------------------------------

def load_attack_results(algo: str, env: str, attack: str, seed: int | None):
    """Return dict[sharing -> dict[epsilon -> list[mean_return]]]."""
    results = defaultdict(lambda: defaultdict(list))

    pattern = f"attack_{attack}_eps*.json"
    for json_path in RESULTS_DIR.glob(f"{env}/{algo}/*/*/{pattern}"):
        # path structure: results/<env>/<algo>/<sharing>/seed<N>/attack_*.json
        parts = json_path.parts
        sharing = parts[-3]          # e.g. "shared"
        seed_dir = parts[-2]         # e.g. "seed1"
        run_seed = int(seed_dir.replace("seed", ""))

        if seed is not None and run_seed != seed:
            continue

        data = json.loads(json_path.read_text())
        eps = data["epsilon"]
        results[sharing][eps].append(data["mean_return"])

    return results


def plot_attack_comparison(ax, results: dict, title: str):
    """Plot mean return vs epsilon, one line per sharing variant.

    epsilon=0 is drawn as a dashed horizontal reference line (clean baseline)
    rather than a curve point, so the x-axis only covers attacked conditions.
    """
    found = False
    for sharing in ("shared", "independent"):
        if sharing not in results:
            continue
        color = COLORS[sharing]
        eps_data = results[sharing]
        n = len(next(iter(eps_data.values())))

        baseline_vals = eps_data.get(0.0, [])
        attack_data   = {e: v for e, v in eps_data.items() if e > 0}

        if baseline_vals:
            baseline_mean = float(np.mean(baseline_vals))
            ax.axhline(baseline_mean, color=color, linestyle="--", alpha=0.6,
                       label=f"{sharing} clean ({baseline_mean:.1f})")

        if attack_data:
            epsilons = sorted(attack_data)
            means = [np.mean(attack_data[e]) for e in epsilons]
            stds  = [np.std(attack_data[e])  for e in epsilons]
            ax.plot(epsilons, means, marker="o",
                    label=f"{sharing} attacked (n={n})", color=color)
            ax.fill_between(epsilons,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            alpha=0.2, color=color)

            # annotate % drop at largest epsilon relative to clean baseline
            if baseline_vals and baseline_mean != 0:
                drop_pct = 100 * (baseline_mean - means[-1]) / abs(baseline_mean)
                ax.annotate(f"Δ{drop_pct:.1f}%",
                            xy=(epsilons[-1], means[-1]),
                            xytext=(4, 0), textcoords="offset points",
                            fontsize=8, color=color, va="center")

        found = True

    ax.set_title(title)
    ax.set_xlabel("Epsilon (noise magnitude)")
    ax.set_ylabel("Mean episode return")
    if found:
        ax.legend(fontsize=8)
    return found


# ---------------------------------------------------------------------------
# Attack-bar mode helpers
# ---------------------------------------------------------------------------

def load_attack_bar_results(algo: str, env: str, epsilon: float, seed: int | None):
    """Return dict[attack -> dict[sharing -> (mean, std)]].

    no_attack is always loaded at eps=0.0; all other attacks are loaded at epsilon.
    """
    results = defaultdict(dict)

    for json_path in RESULTS_DIR.glob(f"{env}/{algo}/*/*/attack_*.json"):
        parts    = json_path.parts
        sharing  = parts[-3]
        run_seed = int(parts[-2].replace("seed", ""))

        if seed is not None and run_seed != seed:
            continue

        data      = json.loads(json_path.read_text())
        attack    = data["attack"]
        file_eps  = data["epsilon"]

        if attack == "no_attack" and file_eps == 0.0:
            results[attack][sharing] = (data["mean_return"], data["std_return"])
        elif attack != "no_attack" and file_eps == epsilon:
            results[attack][sharing] = (data["mean_return"], data["std_return"])

    return results


def plot_attack_bar(ax, results: dict, title: str):
    attacks = [a for a in ATTACK_ORDER if a in results]
    if not attacks:
        return False

    x     = np.arange(len(attacks))
    width = 0.35

    for i, sharing in enumerate(("shared", "independent")):
        means, stds = [], []
        for attack in attacks:
            if sharing in results[attack]:
                m, s = results[attack][sharing]
                means.append(m)
                stds.append(s)
            else:
                means.append(np.nan)
                stds.append(0.0)

        offset = (i - 0.5) * width
        bar_rects = ax.bar(x + offset, means, width, label=sharing,
                           color=COLORS[sharing], alpha=0.8,
                           yerr=stds, capsize=4, error_kw={"alpha": 0.5})
        for rect, mv in zip(bar_rects, means):
            if not np.isnan(mv):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        mv - 1.0, f"{mv:.1f}",
                        ha="center", va="top", fontsize=8)

    all_vals = [v for atk in results.values() for (v, _) in atk.values()
                if not np.isnan(v)]
    if all_vals:
        ax.set_ylim(min(all_vals) * 1.15, 5)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_LABELS.get(a, a) for a in attacks])
    ax.set_ylabel("Mean episode return")
    ax.set_title(title)
    ax.legend()
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _apply_style()
    parser = argparse.ArgumentParser(description="Plot training curves or attack comparison")
    parser.add_argument("--algo",     required=True,
                        choices=["iql", "ippo", "mappo", "qmix", "vdn", "maddpg"])
    parser.add_argument("--env",      default=None,
                        help="Env filter (e.g. simple_spread)")
    parser.add_argument("--seed",     type=int, default=None,
                        help="Plot only this seed (default: aggregate all seeds)")
    parser.add_argument("--mode",     default="train",
                        choices=["train", "attack", "attack_compare", "sdor_train"],
                        help="'train': Sacred curves. 'attack': return vs epsilon. "
                             "'attack_compare': bar chart across all attacks at fixed epsilon. "
                             "'sdor_train': SDor offline training curves.")
    parser.add_argument("--attack",   default="random_noise",
                        help="Attack name for --mode attack (default: random_noise)")
    parser.add_argument("--epsilon",  type=float, default=0.1,
                        help="Epsilon for --mode attack_compare (default: 0.1)")
    parser.add_argument("--out",      default="figures")
    parser.add_argument("--smoothing", type=int, default=None,
                        help="Rolling average window (train mode only)")
    parser.add_argument("--min_steps", type=int, default=5,
                        help="Ignore Sacred runs with fewer than N recorded test points "
                             "(filters smoke tests; default 5)")
    parser.add_argument("--latest", action="store_true",
                        help="Plot only the longest run per sharing variant instead of aggregating")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_tag  = f"_seed{args.seed}" if args.seed is not None else ""
    env_tag   = args.env or "all"

    if args.mode == "train":
        runs = load_train_runs(args.algo, args.env, args.seed,
                               min_steps=args.min_steps, latest=args.latest)
        if not runs:
            print(f"No Sacred runs found for algo='{args.algo}' under {SACRED_DIR}")
            sys.exit(1)
        for sharing, run_list in runs.items():
            print(f"  {sharing}: {len(run_list)} run(s)")

        loss_metrics = LOSS_METRICS.get(args.algo, ["loss"])
        ncols = 1 + len(loss_metrics)
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
        if ncols == 1:
            axes = [axes]

        plot_train_metric(axes[0], runs, REWARD_METRIC,
                          f"{args.algo} — test return ({env_tag}{seed_tag})", args.smoothing)
        axes[0].set_ylabel("Mean episode return")
        for i, metric in enumerate(loss_metrics):
            plot_train_metric(axes[1 + i], runs, metric,
                              f"{args.algo} — {metric}", args.smoothing)
            axes[1 + i].set_ylabel(metric)

        out_path = out_dir / f"{args.algo}_{env_tag}{seed_tag}_train.pdf"

    elif args.mode == "attack_compare":
        if args.env is None:
            print("--env is required for attack_compare mode")
            sys.exit(1)
        results = load_attack_bar_results(args.algo, args.env, args.epsilon, args.seed)
        if not results:
            print(f"No attack results found under {RESULTS_DIR}. Run exp_attack.py first.")
            sys.exit(1)
        for attack, sharing_data in results.items():
            for sharing, (m, s) in sharing_data.items():
                print(f"  {attack:15s} {sharing:11s}  mean={m:.2f}  std={s:.2f}")

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_attack_bar(ax, results,
                        f"{args.algo} attacks at ε={args.epsilon} ({env_tag}{seed_tag})")
        out_path = out_dir / f"{args.algo}_{env_tag}{seed_tag}_attack_compare_eps{args.epsilon}.pdf"

    elif args.mode == "attack":
        if args.env is None:
            print("--env is required for attack mode")
            sys.exit(1)
        results = load_attack_results(args.algo, args.env, args.attack, args.seed)
        if not results:
            print(f"No attack results found. Run exp_attack.py first.")
            sys.exit(1)
        for sharing, eps_data in results.items():
            print(f"  {sharing}: {sorted(eps_data.keys())}")

        fig, ax = plt.subplots(figsize=(7, 4))
        plot_attack_comparison(ax, results,
                               f"{args.algo} — {args.attack} ({env_tag}{seed_tag})")
        out_path = out_dir / f"{args.algo}_{env_tag}{seed_tag}_{args.attack}.pdf"

    else:  # sdor_train mode
        if args.env is None or args.seed is None:
            print("--env and --seed are required for sdor_train mode")
            sys.exit(1)
        metrics_path = (
            RESULTS_DIR / "sdor" / args.env / f"seed{args.seed}" / "training_metrics.json"
        )
        if not metrics_path.exists():
            print(f"No SDor training metrics found at {metrics_path}")
            sys.exit(1)
        m = json.loads(metrics_path.read_text())
        print(f"Loaded {len(m['episodes'])} episodes of SDor training metrics.")

        episodes = np.array(m["episodes"])
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        def _plot_curve(ax, y_raw, ylabel, title):
            y = np.array(y_raw, dtype=float)
            valid = ~np.isnan(y)
            y_plot = _smooth(y[valid], args.smoothing)
            x_plot = episodes[valid][:len(y_plot)]
            ax.plot(x_plot, y_plot, color="#2196F3")
            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)
            ax.set_title(title)

        _plot_curve(axes[0], m["ep_return"],   "Team reward",   f"SDor — protagonist reward ({env_tag}{seed_tag})")
        _plot_curve(axes[1], m["critic_loss"], "Critic loss",   f"SDor — critic loss")
        _plot_curve(axes[2], m["actor_loss"],  "Actor loss",    f"SDor — actor loss")

        out_path = out_dir / f"sdor_{env_tag}{seed_tag}_training.pdf"

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
