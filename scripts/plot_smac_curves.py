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
    ("qmix",    "shared",      "qmix",     "C0", "-"),
    ("qmix_ns", "independent", "qmix",     "C1", "--"),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

for cfg, sharing, algo, color, ls in CELLS:
    cfg_glob = f"epymarl/results/sacred/{cfg}/2s_vs_1sc/*/config.json"
    for cfg_path in sorted(glob.glob(str(REPO / cfg_glob))):
        cfg_data = json.load(open(cfg_path))
        local = cfg_data.get("local_results_path", "").replace("\\", "/")
        expected = f"smaclite_2s_vs_1sc/{algo}/{sharing}/seed1"
        if cfg_data.get("seed") == 1 and local.endswith(expected):
            metrics_path = cfg_path.replace("config.json", "metrics.json")
            m = json.load(open(metrics_path))
            r = m.get("test_return_mean", {})
            w = m.get("test_battle_won_mean", {})
            axes[0].plot(r["steps"], r["values"], color=color, linestyle=ls, marker="o", markersize=4, label=f"{algo} / {sharing}")
            axes[1].plot(w["steps"], w["values"], color=color, linestyle=ls, marker="o", markersize=4, label=f"{algo} / {sharing}")
            break

axes[0].set_xlabel("env steps")
axes[0].set_ylabel("test_return_mean")
axes[0].set_title("Test return")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].set_xlabel("env steps")
axes[1].set_ylabel("test_battle_won_mean")
axes[1].set_title("Test win rate")
axes[1].set_ylim(-0.05, 1.05)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.suptitle("SMAClite 2s_vs_1sc — QMIX shared vs independent (seed 1, t_max=500K)")
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"Saved -> {OUT}")
