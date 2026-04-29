#!/usr/bin/env bash
# Sequentially trains every cell in the clean-env sweep.
#
# Cells: {iql, ippo, mappo, qmix, vdn} x {shared, independent} x SEEDS.
# Skips cells whose checkpoint for t_max already exists, so the sweep is restartable.
#
# Usage (from repo root, in bash / git-bash):
#   bash scripts/run_train_sweep.sh                  # default SEEDS + T_MAX
#   T_MAX=2050000 SEEDS="1 2 3 4 5" bash scripts/run_train_sweep.sh

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
VENV_PY="$REPO_ROOT/.venv/Scripts/python.exe"

ALGOS=(iql ippo mappo qmix vdn)
SHARINGS=(shared independent)
SEEDS="${SEEDS:-1 2 3}"
T_MAX="${T_MAX:-1000000}"

echo "[sweep] t_max=$T_MAX seeds=$SEEDS algos=${ALGOS[*]} sharing=${SHARINGS[*]}"
echo "[sweep] cwd=$REPO_ROOT"
echo

total=0
for _ in $SEEDS; do for _ in "${ALGOS[@]}"; do for _ in "${SHARINGS[@]}"; do total=$((total+1)); done; done; done
i=0

for seed in $SEEDS; do
  for algo in "${ALGOS[@]}"; do
    for sharing in "${SHARINGS[@]}"; do
      i=$((i+1))
      out="results/mpe_simple_spread/$algo/$sharing/seed$seed"
      # Skip if ANY checkpoint exists — exp_train.py's own _find_checkpoint uses
      # the same lenient rule. Note: EPyMARL saves at multiples of
      # save_model_interval, so the final step may not equal T_MAX exactly.
      ckpt_glob="$out/models/*/*/agent.th"
      if compgen -G "$ckpt_glob" > /dev/null; then
        echo "[sweep $i/$total] SKIP $algo $sharing seed=$seed (checkpoint at step $T_MAX already exists)"
        continue
      fi
      echo "[sweep $i/$total] RUN  $algo $sharing seed=$seed t_max=$T_MAX"
      "$VENV_PY" exp_train.py --algo "$algo" --sharing "$sharing" --env mpe_simple_spread --seed "$seed" --t_max "$T_MAX" \
        || { echo "[sweep $i/$total] FAILED $algo $sharing seed=$seed — continuing to next cell"; }
    done
  done
done

echo
echo "[sweep] done"
