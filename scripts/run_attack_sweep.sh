#!/usr/bin/env bash
# Evaluates every trained cell under every attack at every epsilon.
#
# Cells: {iql, ippo, mappo, qmix, vdn} x {shared, independent} x SEEDS
# Attacks: none (once per cell), random + fgsm across EPSILONS
#
# Usage (from repo root, bash / git-bash):
#   bash scripts/run_attack_sweep.sh
#   SEEDS="1 2 3" EPSILONS="0.0 0.05 0.1 0.25 0.5" bash scripts/run_attack_sweep.sh

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
VENV_PY="$REPO_ROOT/.venv/Scripts/python.exe"

ALGOS=(iql ippo mappo qmix vdn)
SHARINGS=(shared independent)
SEEDS="${SEEDS:-1 2 3}"
EPSILONS="${EPSILONS:-0.05 0.1 0.25 0.5}"
N_EPISODES="${N_EPISODES:-100}"

# Number of runs: (#cells) * (1 for none + 2 attacks * #eps)
count_eps=0; for _ in $EPSILONS; do count_eps=$((count_eps+1)); done
count_cells=0
for _ in $SEEDS; do for _ in "${ALGOS[@]}"; do for _ in "${SHARINGS[@]}"; do count_cells=$((count_cells+1)); done; done; done
total=$(( count_cells * (1 + 2 * count_eps) ))

echo "[attack-sweep] cells=$count_cells epsilons='$EPSILONS' n_episodes=$N_EPISODES total=$total"

i=0
for seed in $SEEDS; do
  for algo in "${ALGOS[@]}"; do
    for sharing in "${SHARINGS[@]}"; do
      ckpt="results/mpe_simple_spread/$algo/$sharing/seed$seed"

      # none (clean eval in same harness)
      i=$((i+1))
      if [[ -f "$ckpt/attacks/none_eps0.0/metrics.json" ]]; then
        echo "[attack-sweep $i/$total] SKIP $algo $sharing seed=$seed none"
      else
        echo "[attack-sweep $i/$total] RUN  $algo $sharing seed=$seed none"
        "$VENV_PY" exp_attack.py --ckpt "$ckpt" --attack none --epsilon 0.0 \
          --n_episodes "$N_EPISODES" --seed "$seed" \
          > /dev/null || { echo "  FAILED"; continue; }
      fi

      for atk in random fgsm; do
        for eps in $EPSILONS; do
          i=$((i+1))
          out="$ckpt/attacks/${atk}_eps${eps}/metrics.json"
          if [[ -f "$out" ]]; then
            echo "[attack-sweep $i/$total] SKIP $algo $sharing seed=$seed $atk eps=$eps"
            continue
          fi
          echo "[attack-sweep $i/$total] RUN  $algo $sharing seed=$seed $atk eps=$eps"
          "$VENV_PY" exp_attack.py --ckpt "$ckpt" --attack "$atk" --epsilon "$eps" \
            --n_episodes "$N_EPISODES" --seed "$seed" \
            > /dev/null || { echo "  FAILED"; continue; }
        done
      done
    done
  done
done

echo "[attack-sweep] done"
