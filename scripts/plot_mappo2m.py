"""Plot MAPPO/shared MPE simple_spread: 1M production seeds vs 2M control (seed=99)."""
import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures" / "mappo_2m_calibration.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

ENV_SUBDIR = "pz-mpe-simple-spread-v3"
EXPECTED_SUFFIX = "mpe_simple_spread/mappo/shared/seed"

CELLS = [
    (1,  "seed=1 (1M)",        "C0", "-",  1.2),
    (2,  "seed=2 (1M)",        "C1", "-",  1.2),
    (3,  "seed=3 (1M)",        "C2", "-",  1.2),
    (99, "seed=99 (2M control)", "C3", "-", 2.0),
]


def load(seed):
    pattern = REPO / "epymarl" / "results" / "sacred" / "mappo" / ENV_SUBDIR / "*" / "config.json"
    for cfg_path in sorted(glob.glob(str(pattern))):
        with open(cfg_path) as f:
            cdata = json.load(f)
        local = cdata.get("local_results_path", "").replace("\\", "/")
        if cdata.get("seed") == seed and local.endswith(f"{EXPECTED_SUFFIX}{seed}"):
            with open(Path(cfg_path).parent / "metrics.json") as f:
                m = json.load(f)
            r = m.get("test_return_mean", {})
            if r:
                return r["steps"], r["values"]
    return None, None


fig, ax = plt.subplots(figsize=(11, 5))
for seed, label, color, ls, lw in CELLS:
    steps, values = load(seed)
    if steps is None:
        continue
    ax.plot(steps, values, color=color, linestyle=ls, lw=lw, label=label)

ax.set_xlabel("Environment timesteps")
ax.set_ylabel("Mean test return")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"Saved -> {OUT}")
