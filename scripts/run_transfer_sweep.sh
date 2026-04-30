#!/usr/bin/env bash
# Runs the cross-agent transfer-matrix analysis on every trained cell.
#
# Cells: {iql, ippo, mappo, qmix, vdn} x {shared, independent} x SEEDS
# For each cell, exp_transfer.py produces an N x N matrix of mean returns
# at a single epsilon (default 0.25 — matches the main-results-table focus).
#
# Usage (from repo root, bash / git-bash):
#   bash scripts/run_transfer_sweep.sh
#   SEEDS="1 2 3" EPSILON=0.25 N_EPISODES=50 bash scripts/run_transfer_sweep.sh

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
if   [[ -f "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then VENV_PY="$REPO_ROOT/.venv/Scripts/python.exe"
elif [[ -f "$REPO_ROOT/venv/Scripts/python.exe"  ]]; then VENV_PY="$REPO_ROOT/venv/Scripts/python.exe"
elif [[ -f "$REPO_ROOT/.venv/bin/python"         ]]; then VENV_PY="$REPO_ROOT/.venv/bin/python"
elif [[ -f "$REPO_ROOT/venv/bin/python"          ]]; then VENV_PY="$REPO_ROOT/venv/bin/python"
elif command -v python >/dev/null 2>&1;            then VENV_PY="$(command -v python)"
else echo "ERROR: no python interpreter found"; exit 1
fi

ALGOS=(iql ippo mappo qmix vdn)
SHARINGS=(shared independent)
SEEDS="${SEEDS:-1 2 3}"
EPSILON="${EPSILON:-0.25}"
N_EPISODES="${N_EPISODES:-50}"
ENV="${ENV:-mpe_simple_spread}"

count_cells=0
for _ in $SEEDS; do for _ in "${ALGOS[@]}"; do for _ in "${SHARINGS[@]}"; do count_cells=$((count_cells+1)); done; done; done
echo "[transfer-sweep] env=$ENV cells=$count_cells epsilon=$EPSILON n_episodes=$N_EPISODES"

i=0
for seed in $SEEDS; do
  for algo in "${ALGOS[@]}"; do
    for sharing in "${SHARINGS[@]}"; do
      i=$((i+1))
      cell="results/$ENV/$algo/$sharing/seed$seed"
      out="$cell/transfer_eps${EPSILON}.json"
      if [[ -f "$out" ]]; then
        echo "[transfer-sweep $i/$count_cells] SKIP $algo $sharing seed=$seed"
        continue
      fi
      echo "[transfer-sweep $i/$count_cells] RUN  $algo $sharing seed=$seed"
      "$VENV_PY" exp_transfer.py --algo "$algo" --sharing "$sharing" --env "$ENV" --seed "$seed" \
        --epsilon "$EPSILON" --n_episodes "$N_EPISODES" \
        > /dev/null || echo "  FAILED $algo $sharing seed=$seed"
    done
  done
done

echo "[transfer-sweep] done"
