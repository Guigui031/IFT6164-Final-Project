"""Aggregate attack-sweep metrics across cells/seeds into a table + plots.

Walks results/<env>/<algo>/<sharing>/seed<N>/attack_<attack>_eps<eps>.json
and produces:
    - figures/attack_curves.png   (return vs epsilon, per algo/sharing/attack)
    - figures/attack_drop_bar.png (% return drop at fixed epsilon, per algo)
    - figures/results_table.tex   (LaTeX main-results table for the report)
    - figures/aggregate.json      (machine-readable summary)

Means are over SEEDS; errors are standard error of the mean across seeds
(not over episodes — each metrics.json already averages episodes).

Usage (from repo root, venv active):
    python exp_aggregate.py
    python exp_aggregate.py --env mpe_simple_spread --focus_epsilon 0.25
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = REPO_ROOT / "figures"

ALGOS = ["iql", "ippo", "mappo", "qmix", "vdn"]
SHARINGS = ["shared", "independent"]
ATTACKS = ["no_attack", "random_noise", "fgsm", "sdor_stor"]

# matches attack_<name>_eps<value>.json; <name> may itself contain underscores
ATTACK_FILE_RE = re.compile(r"^attack_(?P<atk>.+)_eps(?P<eps>[0-9.]+)\.json$")


def scan(env: str):
    """Return nested dict: records[algo][sharing][attack][epsilon] = list of (seed, mean).

    "no_attack" runs are stored as attack='no_attack' epsilon=0.0 for downstream plotting.
    """
    records = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    env_dir = RESULTS_ROOT / env
    if not env_dir.exists():
        raise FileNotFoundError(f"No env dir {env_dir}")
    for algo in ALGOS:
        for sharing in SHARINGS:
            cell_root = env_dir / algo / sharing
            if not cell_root.exists():
                continue
            for seed_dir in sorted(cell_root.glob("seed*")):
                seed_m = re.match(r"^seed(\d+)$", seed_dir.name)
                if not seed_m:
                    continue
                seed = int(seed_m.group(1))
                for json_path in sorted(seed_dir.glob("attack_*.json")):
                    m = ATTACK_FILE_RE.match(json_path.name)
                    if not m:
                        continue
                    attack = m.group("atk")
                    eps = float(m.group("eps"))
                    with open(json_path) as f:
                        metrics = json.load(f)
                    mean = float(metrics["mean_return"])
                    records[algo][sharing][attack][eps].append((seed, mean))
    return records


def mean_sem(values):
    vals = [v for _, v in values]
    if not vals:
        return (math.nan, math.nan, 0)
    n = len(vals)
    m = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return (m, sem, n)


def render_results_table(records, focus_epsilon: float, out_path: Path):
    """LaTeX table: one row per algo/sharing pair, cols = clean / attacks at focus_epsilon.

    Emits a booktabs table the final report can \\input."""
    lines = []
    lines.append(r"\begin{tabular}{llrrr}")
    lines.append(r"\toprule")
    lines.append(r"Algorithm & Params & Clean & Random ($\epsilon="
                 + f"{focus_epsilon:g}" + r"$) & FGSM ($\epsilon="
                 + f"{focus_epsilon:g}" + r"$) \\")
    lines.append(r"\midrule")
    for algo in ALGOS:
        for sharing in SHARINGS:
            if sharing not in records.get(algo, {}):
                continue
            row = [algo.upper(), sharing]
            # Clean (attack='none' at any eps, take minimum eps)
            clean = records[algo][sharing].get("no_attack", {})
            clean_vals = clean[sorted(clean.keys())[0]] if clean else []
            m, sem, n = mean_sem(clean_vals)
            row.append(f"${m:+.1f}\\pm{sem:.1f}$" if not math.isnan(m) else "--")
            # Random at focus_epsilon
            rand = records[algo][sharing].get("random_noise", {}).get(focus_epsilon, [])
            m, sem, _ = mean_sem(rand)
            row.append(f"${m:+.1f}\\pm{sem:.1f}$" if not math.isnan(m) else "--")
            # FGSM at focus_epsilon
            fgsm = records[algo][sharing].get("fgsm", {}).get(focus_epsilon, [])
            m, sem, _ = mean_sem(fgsm)
            row.append(f"${m:+.1f}\\pm{sem:.1f}$" if not math.isnan(m) else "--")
            lines.append(" & ".join(row) + r" \\")
        if algo != ALGOS[-1]:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines))
    print(f"[aggregate] wrote {out_path}")


def render_attack_curves(records, out_path: Path):
    """Plot: for each algo/sharing, return vs epsilon under each attack."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_algos = sum(1 for a in ALGOS if a in records)
    fig, axes = plt.subplots(n_algos, 1, figsize=(7, 2.5 * n_algos), sharex=True)
    if n_algos == 1:
        axes = [axes]
    ai = 0
    for algo in ALGOS:
        if algo not in records:
            continue
        ax = axes[ai]
        ai += 1
        for sharing in SHARINGS:
            if sharing not in records[algo]:
                continue
            for attack in ATTACKS:
                data = records[algo][sharing].get(attack, {})
                if not data:
                    continue
                eps_vals = sorted(data.keys())
                means = [mean_sem(data[e])[0] for e in eps_vals]
                sems = [mean_sem(data[e])[1] for e in eps_vals]
                style = "-" if sharing == "shared" else "--"
                label = f"{sharing}/{attack}"
                ax.errorbar(eps_vals, means, yerr=sems, fmt=style,
                            marker="o", markersize=4, capsize=3, label=label)
        ax.set_title(f"{algo.upper()}")
        ax.set_ylabel("test return mean")
        ax.grid(alpha=0.3)
        if ai == 1:
            ax.legend(fontsize=8, ncol=3, loc="upper right")
    axes[-1].set_xlabel(r"perturbation $\epsilon$ (L$_\infty$)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[aggregate] wrote {out_path}")


def render_drop_bars(records, focus_epsilon: float, out_path: Path):
    """Bar chart: % return drop at focus_epsilon, grouped by algo, split by sharing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []  # (algo, sharing, fgsm_drop, random_drop)
    for algo in ALGOS:
        if algo not in records:
            continue
        for sharing in SHARINGS:
            if sharing not in records[algo]:
                continue
            clean = records[algo][sharing].get("no_attack", {})
            if not clean:
                continue
            clean_mean = mean_sem(clean[sorted(clean.keys())[0]])[0]
            fgsm_mean = mean_sem(records[algo][sharing].get("fgsm", {}).get(focus_epsilon, []))[0]
            rand_mean = mean_sem(records[algo][sharing].get("random_noise", {}).get(focus_epsilon, []))[0]
            rows.append((algo, sharing, clean_mean, fgsm_mean, rand_mean))

    labels = [f"{a}\n{s[:3]}" for (a, s, *_) in rows]
    fgsm_drops = [c - f for (_, _, c, f, _) in rows]
    rand_drops = [c - r for (_, _, c, _, r) in rows]
    x = np.arange(len(rows))
    w = 0.4

    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(rows)), 4))
    ax.bar(x - w / 2, rand_drops, w, label="random")
    ax.bar(x + w / 2, fgsm_drops, w, label="FGSM")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f"return drop vs clean (ε={focus_epsilon:g})")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[aggregate] wrote {out_path}")


def scan_transfer(env: str, epsilon: float):
    """Return dict[algo][sharing][seed] = dict(matrix, baseline_mean, n_agents).

    Looks for <cell>/transfer_eps<epsilon>.json for every cell.
    """
    out = defaultdict(lambda: defaultdict(dict))
    env_dir = RESULTS_ROOT / env
    fname = f"transfer_eps{epsilon}.json"
    for algo in ALGOS:
        for sharing in SHARINGS:
            cell_root = env_dir / algo / sharing
            if not cell_root.exists():
                continue
            for seed_dir in sorted(cell_root.glob("seed*")):
                m = re.match(r"^seed(\d+)$", seed_dir.name)
                if not m:
                    continue
                seed = int(m.group(1))
                matrix_path = seed_dir / fname
                if not matrix_path.exists():
                    continue
                with open(matrix_path) as f:
                    data = json.load(f)
                out[algo][sharing][seed] = {
                    "matrix": np.asarray(data["matrix"]),
                    "baseline_mean": float(data["baseline_mean"]),
                    "n_agents": int(data["n_agents"]),
                }
    return out


def render_transfer_table(transfer_records, epsilon: float, out_path: Path):
    """LaTeX table: per (algo, sharing) mean diagonal drop, off-diagonal drop, and ratio.

    Transfer ratio > 1 indicates a perturbation crafted for agent i damages the team
    more when applied to another agent than when applied to agent i itself — a sign
    that team value depends non-trivially on which agent is perturbed (mixer effect).
    """
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Algorithm & Params & Clean & Diag.\ drop & Off-diag.\ drop & Transfer ratio \\",
        r"\midrule",
    ]
    for algo in ALGOS:
        for sharing in SHARINGS:
            if sharing not in transfer_records.get(algo, {}):
                continue
            cells = transfer_records[algo][sharing]
            diag_drops, off_drops, clean_vals = [], [], []
            for seed, d in cells.items():
                base = d["baseline_mean"]
                clean_vals.append(base)
                N = d["n_agents"]
                m = d["matrix"]
                for s in range(N):
                    for t in range(N):
                        drop = base - float(m[s, t])
                        (diag_drops if s == t else off_drops).append(drop)
            clean_m = float(np.mean(clean_vals))
            diag_m = float(np.mean(diag_drops)) if diag_drops else float("nan")
            off_m = float(np.mean(off_drops)) if off_drops else float("nan")
            ratio = off_m / diag_m if diag_m > 0 else float("nan")
            lines.append(
                f"{algo.upper()} & {sharing} & "
                f"${clean_m:+.2f}$ & ${diag_m:.2f}$ & ${off_m:.2f}$ & ${ratio:.2f}$ \\\\"
            )
        if algo != ALGOS[-1]:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines))
    print(f"[aggregate] wrote {out_path}")


def render_transfer_heatmaps(transfer_records, epsilon: float, out_path: Path):
    """Grid of heatmaps: one row per algo, two columns (shared | independent).

    Each cell shows the seed-averaged N x N drop matrix (baseline - mean_return).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algos_present = [a for a in ALGOS if a in transfer_records]
    n_rows = len(algos_present)
    if n_rows == 0:
        return
    fig, axes = plt.subplots(n_rows, 2, figsize=(6.5, 2.3 * n_rows), squeeze=False)

    # Shared colour scale across all panels — helps visual comparison.
    all_drops = []
    for algo in algos_present:
        for sharing in SHARINGS:
            for d in transfer_records[algo].get(sharing, {}).values():
                base = d["baseline_mean"]
                all_drops.append(base - d["matrix"])
    vmin = float(np.min([x.min() for x in all_drops]))
    vmax = float(np.max([x.max() for x in all_drops]))

    for i, algo in enumerate(algos_present):
        for j, sharing in enumerate(SHARINGS):
            ax = axes[i, j]
            cells = transfer_records[algo].get(sharing, {})
            if not cells:
                ax.set_visible(False)
                continue
            mats = []
            bases = []
            for d in cells.values():
                bases.append(d["baseline_mean"])
                mats.append(d["matrix"])
            mean_mat = np.mean(np.stack(mats), axis=0)
            base = float(np.mean(bases))
            drop_mat = base - mean_mat  # higher = worse for protagonist
            im = ax.imshow(drop_mat, vmin=vmin, vmax=vmax, cmap="inferno")
            ax.set_title(f"{algo.upper()} / {sharing}", fontsize=9)
            N = drop_mat.shape[0]
            ax.set_xticks(range(N))
            ax.set_yticks(range(N))
            ax.set_xlabel("target agent")
            ax.set_ylabel("source agent")
            for s in range(N):
                for t in range(N):
                    ax.text(t, s, f"{drop_mat[s, t]:.1f}", ha="center", va="center",
                            color="white" if drop_mat[s, t] > (vmin + vmax) / 2 else "black",
                            fontsize=7)
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.8])
    fig.colorbar(im, cax=cbar_ax, label="return drop vs clean")
    fig.suptitle(f"Transfer matrices at $\\epsilon={epsilon}$", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.88, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[aggregate] wrote {out_path}")


def dump_json(records, out_path: Path):
    out = {}
    for algo, by_sh in records.items():
        out[algo] = {}
        for sharing, by_atk in by_sh.items():
            out[algo][sharing] = {}
            for attack, by_eps in by_atk.items():
                out[algo][sharing][attack] = {}
                for eps, pairs in by_eps.items():
                    m, sem, n = mean_sem(pairs)
                    out[algo][sharing][attack][str(eps)] = {
                        "mean": m, "sem": sem, "n_seeds": n,
                        "per_seed": [{"seed": s, "mean": v} for (s, v) in pairs],
                    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[aggregate] wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="mpe_simple_spread")
    p.add_argument("--focus_epsilon", type=float, default=0.25,
                   help="Epsilon used for the main results table + bar chart.")
    p.add_argument("--out", type=Path, default=FIGURES_ROOT,
                   help="Directory for generated tables and figures.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    records = scan(args.env)
    if not records:
        print(f"[aggregate] no records found for env={args.env}")
        return 1

    n_records = 0
    for algo, by_sh in records.items():
        for sharing, by_atk in by_sh.items():
            for attack, by_eps in by_atk.items():
                for eps, pairs in by_eps.items():
                    n_records += len(pairs)
    print(f"[aggregate] found {n_records} per-seed data points across "
          f"{len(records)} algos")

    dump_json(records, args.out / "aggregate.json")
    render_results_table(records, args.focus_epsilon,
                         args.out / "results_table.tex")
    render_attack_curves(records, args.out / "attack_curves.png")
    render_drop_bars(records, args.focus_epsilon,
                     args.out / "attack_drop_bar.png")

    # Transfer analysis (only runs if exp_transfer.py was run at focus_epsilon).
    transfer_records = scan_transfer(args.env, args.focus_epsilon)
    n_transfer_cells = sum(len(sh_map)
                           for algo in transfer_records.values()
                           for sh_map in algo.values())
    if n_transfer_cells > 0:
        print(f"[aggregate] found transfer data for {n_transfer_cells} cells "
              f"at epsilon={args.focus_epsilon}")
        render_transfer_table(transfer_records, args.focus_epsilon,
                              args.out / "transfer_table.tex")
        render_transfer_heatmaps(transfer_records, args.focus_epsilon,
                                 args.out / "transfer_heatmaps.png")
    else:
        print(f"[aggregate] no transfer data at epsilon={args.focus_epsilon}; "
              "run exp_transfer.py to generate it.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
