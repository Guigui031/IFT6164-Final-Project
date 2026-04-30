"""Plot MPE simple_spread training curves (test_return_mean) per algo, shared vs independent.

Writes one PNG per algorithm under figures/mpe_simple_spread_<algo>.png so the
report can include each panel independently and add its own caption.
"""
import json
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENV_SUBDIR = "pz-mpe-simple-spread-v3"
SEEDS = (1, 2, 3)

ALGOS = [
    ("iql",   "IQL"),
    ("ippo",  "IPPO"),
    ("mappo", "MAPPO"),
    ("qmix",  "QMIX"),
    ("vdn",   "VDN"),
]

VARIANTS = [
    ("shared",      "",    "tab:blue",   "-"),
    ("independent", "_ns", "tab:orange", "-"),
]


def load_seed_curves(cfg, expected_local_suffix, seed):
    """Return (steps, values) for the matching sacred run, or (None, None) if missing."""
    pattern = REPO / "epymarl" / "results" / "sacred" / cfg / ENV_SUBDIR / "*" / "config.json"
    for cfg_path in sorted(glob.glob(str(pattern))):
        with open(cfg_path) as f:
            cdata = json.load(f)
        local = cdata.get("local_results_path", "").replace("\\", "/")
        if cdata.get("seed") == seed and local.endswith(expected_local_suffix):
            with open(Path(cfg_path).parent / "metrics.json") as f:
                m = json.load(f)
            r = m.get("test_return_mean", {})
            if r:
                return np.asarray(r["steps"]), np.asarray(r["values"])
    return None, None


def aggregate(curves):
    """Align curves on a common step grid (intersection of steps) and return mean/SEM."""
    if not curves:
        return None, None, None
    common = curves[0][0]
    for s, _ in curves[1:]:
        common = np.intersect1d(common, s)
    if common.size == 0:
        return None, None, None
    stacked = []
    for s, v in curves:
        idx = np.searchsorted(s, common)
        stacked.append(v[idx])
    arr = np.stack(stacked, axis=0)
    return common, arr.mean(axis=0), arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])


# First pass: load every curve so we can compute shared axis limits across all
# 5 plots (avoids per-plot autoscaling that makes panels visually incomparable).
plot_data = {}  # algo -> list of (sharing, color, ls, steps, mean, sem)
for algo, _ in ALGOS:
    series = []
    for sharing, suffix, color, ls in VARIANTS:
        cfg = f"{algo}{suffix}"
        expected = f"mpe_simple_spread/{algo}/{sharing}/seed"
        curves = []
        for seed in SEEDS:
            s, v = load_seed_curves(cfg, f"{expected}{seed}", seed)
            if s is not None:
                curves.append((s, v))
        steps, mean, sem = aggregate(curves)
        if steps is None:
            continue
        series.append((sharing, color, ls, steps, mean, sem))
    plot_data[algo] = series

# Compute global axis limits with a small margin.
all_lo, all_hi, all_xmax = [], [], []
for series in plot_data.values():
    for _, _, _, steps, mean, sem in series:
        all_lo.append(float((mean - sem).min()))
        all_hi.append(float((mean + sem).max()))
        all_xmax.append(float(steps.max()))
y_lo, y_hi = min(all_lo), max(all_hi)
y_pad = 0.05 * (y_hi - y_lo)
y_lim = (y_lo - y_pad, y_hi + y_pad)
x_lim = (0.0, max(all_xmax))

# Second pass: render each algo with the shared limits.
for algo, _ in ALGOS:
    series = plot_data.get(algo, [])
    if not series:
        continue
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    for sharing, color, ls, steps, mean, sem in series:
        ax.plot(steps, mean, color=color, linestyle=ls, lw=1.5, label=sharing)
        ax.fill_between(steps, mean - sem, mean + sem, color=color, alpha=0.18)
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel("Mean test return")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path = OUT_DIR / f"mpe_simple_spread_{algo}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved -> {out_path}")
