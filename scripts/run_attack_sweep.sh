#!/usr/bin/env bash
# Evaluates every trained cell under every attack at every epsilon.
#
# Cells: {iql, ippo, mappo, qmix, vdn} x {shared, independent} x SEEDS
# Attacks: no_attack (once per cell), random_noise + fgsm across EPSILONS
#
# Usage (from repo root, bash / git-bash):
#   bash scripts/run_attack_sweep.sh
#   SEEDS="1 2 3" EPSILONS="0.05 0.1 0.25 0.5" bash scripts/run_attack_sweep.sh

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
VENV_PY="$REPO_ROOT/.venv/Scripts/python.exe"

ALGOS=(iql ippo mappo qmix vdn)
SHARINGS=(shared independent)
SEEDS="${SEEDS:-1 2 3}"
EPSILONS="${EPSILONS:-0.05 0.1 0.25 0.5}"
N_EPISODES="${N_EPISODES:-100}"
ENV="${ENV:-mpe_simple_spread}"

# Number of runs: (#cells) * (1 for no_attack + 2 attacks * #eps)
count_eps=0; for _ in $EPSILONS; do count_eps=$((count_eps+1)); done
count_cells=0
for _ in $SEEDS; do for _ in "${ALGOS[@]}"; do for _ in "${SHARINGS[@]}"; do count_cells=$((count_cells+1)); done; done; done
total=$(( count_cells * (1 + 2 * count_eps) ))

echo "[attack-sweep] env=$ENV cells=$count_cells epsilons='$EPSILONS' n_episodes=$N_EPISODES total=$total"

i=0
for seed in $SEEDS; do
  for algo in "${ALGOS[@]}"; do
    for sharing in "${SHARINGS[@]}"; do
      cell="results/$ENV/$algo/$sharing/seed$seed"

      # no_attack (clean eval in same harness). Do NOT `continue` on failure —
      # the subsequent random/fgsm evals for this cell are independent and
      # should still be attempted.
      i=$((i+1))
      if [[ -f "$cell/attack_no_attack_eps0.0.json" ]]; then
        echo "[attack-sweep $i/$total] SKIP $algo $sharing seed=$seed no_attack"
      else
        echo "[attack-sweep $i/$total] RUN  $algo $sharing seed=$seed no_attack"
        "$VENV_PY" exp_attack.py --algo "$algo" --sharing "$sharing" --env "$ENV" --seed "$seed" \
          --attack no_attack --epsilon 0.0 --n_episodes "$N_EPISODES" \
          > /dev/null || echo "  FAILED (no_attack) — continuing with this cell's attacks anyway"
      fi

      for atk in random_noise fgsm; do
        for eps in $EPSILONS; do
          i=$((i+1))
          out="$cell/attack_${atk}_eps${eps}.json"
          if [[ -f "$out" ]]; then
            echo "[attack-sweep $i/$total] SKIP $algo $sharing seed=$seed $atk eps=$eps"
            continue
          fi
          echo "[attack-sweep $i/$total] RUN  $algo $sharing seed=$seed $atk eps=$eps"
          "$VENV_PY" exp_attack.py --algo "$algo" --sharing "$sharing" --env "$ENV" --seed "$seed" \
            --attack "$atk" --epsilon "$eps" --n_episodes "$N_EPISODES" \
            > /dev/null || { echo "  FAILED"; continue; }
        done
      done
    done
  done
done

echo "[attack-sweep] done"
