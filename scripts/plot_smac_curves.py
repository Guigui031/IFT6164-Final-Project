"""Plot SMAClite training curves (test_return_mean + win rate) for each run."""
import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures" / "smaclite_2s_vs_1sc_training.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

CELLS = [
    ("qmix",    "shared",      "qmix",     "tab:blue",   "-"),
    ("qmix_ns", "independent", "qmix",     "tab:orange", "-"),
]

fig, ax = plt.subplots(figsize=(7, 4))

for cfg, sharing, algo, color, ls in CELLS:
    cfg_glob = f"epymarl/results/sacred/{cfg}/2s_vs_1sc/*/config.json"
    for cfg_path in sorted(glob.glob(str(REPO / cfg_glob))):
        cfg_data = json.load(open(cfg_path))
        local = cfg_data.get("local_results_path", "").replace("\\", "/")
        expected = f"smaclite_2s_vs_1sc/{algo}/{sharing}/seed99"
        if cfg_data.get("seed") == 99 and local.endswith(expected):
            metrics_path = cfg_path.replace("config.json", "metrics.json")
            m = json.load(open(metrics_path))
            r = m.get("test_return_mean", {})
            ax.plot(r["steps"], r["values"], color=color, linestyle=ls, marker="o", markersize=4, label=f"{algo.upper()} / {sharing}")
            break

ax.set_xlabel("Environment timesteps")
ax.set_ylabel("Mean test return")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"Saved -> {OUT}")
