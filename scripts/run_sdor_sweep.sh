#!/usr/bin/env bash
# Trains the SDor stochastic adversary for each (algo, seed) cell against the
# shared protagonist (Option B). The same SDor is then used to attack BOTH
# shared and independent variants of that algorithm at evaluation time, since
# STor's per-step FGSM uses the target MAC's gradients.
#
# Cells: ALGOS x SEEDS (sharing is always "shared" -- exp_sdor_train.py
# hardcodes the protagonist sharing variant).
# Skips cells whose sdor.pt already exists. Skips cells whose team checkpoint
# is missing with a [WAIT] message (run run_train_sweep.sh first).
#
# Usage (from repo root, bash / git-bash):
#   bash scripts/run_sdor_sweep.sh
#   ENV=mpe_simple_reference SEEDS="1 2 3" bash scripts/run_sdor_sweep.sh

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

# Only MAPPO is wired for SDor today (the eval pipeline assumes a discrete
# protagonist; extending to other algos is a separate change).
ALGOS=(mappo)
SEEDS="${SEEDS:-1 2 3}"
ENV="${ENV:-mpe_simple_spread}"

count_cells=0
for _ in $SEEDS; do for _ in "${ALGOS[@]}"; do count_cells=$((count_cells+1)); done; done
echo "[sdor-sweep] env=$ENV cells=$count_cells algos=${ALGOS[*]} seeds='$SEEDS'"

i=0
for seed in $SEEDS; do
  for algo in "${ALGOS[@]}"; do
    i=$((i+1))
    out="results/sdor/$ENV/$algo/shared/seed$seed/sdor.pt"
    if [[ -f "$out" ]]; then
      echo "[sdor-sweep $i/$count_cells] SKIP $algo seed=$seed"
      continue
    fi

    # Need the team's shared checkpoint to train against
    team_ckpt_glob="results/$ENV/$algo/shared/seed$seed/models/*/*/agent.th"
    if ! compgen -G "$team_ckpt_glob" > /dev/null; then
      echo "[sdor-sweep $i/$count_cells] WAIT $algo seed=$seed (no shared team checkpoint -- run run_train_sweep.sh first)"
      continue
    fi

    echo "[sdor-sweep $i/$count_cells] RUN  $algo seed=$seed"
    "$VENV_PY" exp_sdor_train.py --algo "$algo" --env "$ENV" --seed "$seed" \
      || echo "  FAILED $algo seed=$seed -- continuing to next cell"
  done
done

echo
echo "[sdor-sweep] done"
