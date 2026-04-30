# IFT6164 Final Project

Course project investigating whether **parameter sharing in cooperative MARL amplifies or dampens adversarial vulnerability**, using EPyMARL on MPE cooperative navigation. See `extended-abstract/extended_abstract.tex` for the full pre-registered research plan.

## What's in this repo

- `extended-abstract/` — pre-registered extended abstract (the authoritative research spec)
- `report/` — 8-page final report (work in progress)
- `papers/` — reference PDFs
- `epymarl/` — vendored fork of [uoe-agents/epymarl](https://github.com/uoe-agents/epymarl) with two small Windows-compatibility patches (see `CLAUDE.md` → "Environment")
- `src/` — our code: attacks, env wrappers, analysis (mostly placeholders; see roadmap in `CLAUDE.md`)
- `configs/` — per-experiment YAML overrides (being filled in as we go)
- `requirements.txt` — frozen pip requirements from a working venv
- `CLAUDE.md` — detailed architecture, experiment matrix, implementation roadmap, and gotchas. **Read this before making any real changes.**

## Prerequisites

- **OS:** Windows 11 (what we developed on; Linux should work but isn't tested).
- **GPU:** NVIDIA GPU with recent driver. We test on an RTX 3060 with driver 591.86. Any CUDA ≥ 12.1 driver is fine.
- **Python:** 3.10 (EPyMARL's deps don't all play well with 3.11+).
- **Package manager:** [`uv`](https://github.com/astral-sh/uv) recommended — installs a pinned Python and resolves deps fast. Plain `pip + venv` works too if you already have Python 3.10 installed.
- **Git** for cloning.

## Setup

### Option 1 — with `uv` (recommended, what we used)

From the repo root:

```bash
# 1. Install Python 3.10 (only if you don't already have one `uv` knows about)
uv python install 3.10

# 2. Create the venv
uv venv --python 3.10 .venv

# 3. Activate it (pick the right one for your shell)
# PowerShell
.venv\Scripts\Activate.ps1
# cmd.exe
.venv\Scripts\activate.bat
# bash / git-bash
source .venv/Scripts/activate

# 4. Install PyTorch with CUDA 12.1 wheels (the custom index is important)
uv pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# 5. Install the rest
uv pip install -r requirements.txt
```

### Option 2 — with pip

Same steps, but use the standard tools:

```bash
python3.10 -m venv .venv
source .venv/Scripts/activate  # or the Windows equivalent
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

### Sanity-check the install

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output (yours will differ on device name):

```
2.2.2+cu121 True NVIDIA GeForce RTX 3060
```

### Verify the parameter-sharing toggle

This is the load-bearing assumption of the whole study — make sure the independent-network variant really has N× the parameters of the shared one:

```bash
python src/utils/verify_sharing.py
```

Expected: ratio = 3.00× for 3 agents (both GRU and MLP backbones).

## Smoke test — run IQL on MPE for a few seconds

From the repo root:

```bash
cd epymarl
python src/main.py --config=iql --env-config=gymma \
  with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3" \
  t_max=2000 test_interval=1000 log_interval=500
```

This runs a 2000-step toy training run (~7 seconds). You should see TD loss around 1.0, epsilon close to 1.0, and test returns around −130 (not converged — the point is that the loop runs end-to-end without crashing).

A full 2M-step IQL training on the RTX 3060 takes **~90 minutes** and converges to `test_return_mean ≈ −34` on MPE simple_spread with 3 agents.

## Running experiments

The full pipeline (train → attack → transfer-matrix → aggregate) goes through four wrapper scripts at the repo root and three matching sweep scripts under `scripts/`. The wrappers handle config selection, the `_ns` ↔ `_shared` equalisation overrides (see `CLAUDE.md` → "Sharing toggle"), and the per-cell `results/<env>/<algo>/<sharing>/seed<n>/` output layout.

All commands below are run from the repo root with the venv activated.

### 1. Train a single cell

```bash
python exp_train.py --algo iql --sharing shared --env mpe_simple_spread --seed 1 --t_max 1000000
```

Arguments:
- `--algo` ∈ {`iql`, `ippo`, `mappo`, `qmix`, `vdn`}
- `--sharing` ∈ {`shared`, `independent`} — `independent` automatically applies the `_ns` config plus the equalisation overrides
- `--env` ∈ {`mpe_simple_spread`, `smaclite_2s_vs_1sc`} (entries in `ENV_MAP` at the top of `exp_train.py`; add a new entry to support a new env)
- `--seed` integer
- `--t_max` total environment steps (defaults to the env's `default_t_max`)

The wrapper is restartable — it skips cells whose checkpoint already exists. A live-updating training plot is saved to `figures/<env>/<algo>_<sharing>_seed<n>.png` and the model + sacred metadata land under `results/<env>/<algo>/<sharing>/seed<n>/`.

### 2. Train every cell in the matrix

```bash
# default: 5 algos × 2 sharings × 3 seeds = 30 cells, t_max=1M, on mpe_simple_spread
bash scripts/run_train_sweep.sh

# override knobs
T_MAX=2050000 SEEDS="1 2 3 4 5" bash scripts/run_train_sweep.sh
```

Sequential, restartable, ~30–60 min per MPE cell on the RTX 3060.

### 3. Evaluate a checkpoint under attack

```bash
# clean baseline
python exp_attack.py --algo iql --sharing shared --env mpe_simple_spread --seed 1 \
  --attack no_attack --epsilon 0.0 --n_episodes 100

# random noise / FGSM
python exp_attack.py --algo iql --sharing shared --env mpe_simple_spread --seed 1 \
  --attack fgsm --epsilon 0.1 --n_episodes 100
```

`--attack` ∈ {`no_attack`, `random_noise`, `fgsm`, `sdor_stor`}. Output: `results/<env>/<algo>/<sharing>/seed<n>/attack_<atk>_eps<e>.json`.

### 4. Attack sweep (all cells × {random, FGSM} × ε grid)

```bash
bash scripts/run_attack_sweep.sh
SEEDS="1 2 3" EPSILONS="0.05 0.1 0.25 0.5" N_EPISODES=100 ENV=mpe_simple_spread \
  bash scripts/run_attack_sweep.sh
```

Default ε grid is {0.05, 0.1, 0.25, 0.5}; default `N_EPISODES=100`. `ENV` defaults to `mpe_simple_spread`.

### 5. Cross-agent transfer matrix (single cell + sweep)

```bash
# single cell — produces an N×N matrix of mean returns
python exp_transfer.py --algo iql --sharing shared --env mpe_simple_spread --seed 1 \
  --epsilon 0.25 --n_episodes 50

# sweep over all cells
bash scripts/run_transfer_sweep.sh
SEEDS="1 2 3" EPSILON=0.25 N_EPISODES=50 ENV=mpe_simple_spread \
  bash scripts/run_transfer_sweep.sh
```

Output: `results/<env>/<algo>/<sharing>/seed<n>/transfer_eps<e>.json`. The transfer ratio (off-diagonal / diagonal drop) is computed downstream by `exp_aggregate.py`.

### 6. SDOR-stor learned adversary (optional, expensive)

```bash
# train the SAC-based adversary against a fixed protagonist checkpoint (Option B)
python exp_sdor_train.py --algo mappo --sharing shared --env mpe_simple_spread --seed 1
```

Trained adversary → `results/sdor/<env>/<algo>/<sharing>/seed<n>/sdor.pt`. Then evaluate any victim with `exp_attack.py --attack sdor_stor`.

### 7. Aggregate everything into report tables/figures

```bash
python exp_aggregate.py
```

Reads every `attack_*.json` and `transfer_*.json` under `results/`, writes the LaTeX tables + summary figures into `figures/`. Re-run after any new sweep.

### 8. Run the whole pipeline end-to-end (overnight)

All four sweep scripts and the aggregator chained together with `&&`, so each stage only runs if the previous succeeded. The `ENV` env var picks the environment — defaults to `mpe_simple_spread`, override to run on a different one.

**Run from `bash` / `git-bash`** (PowerShell does not understand `&&` chaining or `export VAR=value`):

```bash
# default env (simple_spread)
{ bash scripts/run_train_sweep.sh \
  && bash scripts/run_sdor_sweep.sh \
  && bash scripts/run_attack_sweep.sh \
  && bash scripts/run_transfer_sweep.sh \
  && python exp_aggregate.py ; } 2>&1 | tee sweep_default.log

# different env (e.g. simple_reference)
export ENV=mpe_simple_reference

{ bash scripts/run_train_sweep.sh \
  && bash scripts/run_sdor_sweep.sh \
  && bash scripts/run_attack_sweep.sh \
  && bash scripts/run_transfer_sweep.sh \
  && python exp_aggregate.py ; } 2>&1 | tee sweep_${ENV}.log
```

`tee` keeps a full transcript at `sweep_<env>.log`. Every individual stage is restartable, so if you Ctrl+C and re-run the chain, completed cells are skipped.

Wall-clock estimate on an RTX 3060: **~20–35 hours** for all five algorithms × two sharings × three seeds (training dominates; SDor + attack + transfer + aggregate add a few hours). If your shell may close while you're away, prefix with `nohup` and append `&` so the job survives a disconnect:

```bash
nohup bash -c 'export ENV=mpe_simple_reference && \
  bash scripts/run_train_sweep.sh && \
  bash scripts/run_sdor_sweep.sh && \
  bash scripts/run_attack_sweep.sh && \
  bash scripts/run_transfer_sweep.sh && \
  python exp_aggregate.py' > sweep_simple_reference.log 2>&1 &
```

### Outputs at a glance

```
results/<env>/<algo>/<sharing>/seed<n>/
├── models/<run_token>/<step>/agent.th     # EPyMARL checkpoint (every save_model_interval)
├── sacred/<run_id>/{config.json, metrics.json, cout.txt}
├── tb_logs/<run_token>/                   # tensorboard scalars
├── attack_<atk>_eps<e>.json               # one file per (attack, ε) — mean return + per-episode
└── transfer_eps<e>.json                   # N×N transfer matrix
figures/                                    # aggregated plots + LaTeX tables
```

`tensorboard --logdir results/` to browse training curves.

## Optional: SMAClite (second-environment validation)

[SMAClite](https://github.com/uoe-agents/smaclite) is a lightweight pure-Python reimplementation of SMAC scenarios — same interface, same maps, no StarCraft II install required. We use it as a stretch-goal cross-environment validation; the main MPE results don't depend on it.

### Install

```bash
uv pip install --python .venv/Scripts/python.exe "git+https://github.com/uoe-agents/smaclite.git"
```

This pulls in `smaclite==0.0.1` plus `scikit-learn`, `joblib`, `rtree`, `threadpoolctl` (RVO2 collision-avoidance dependencies).

### Verify the install

```bash
python -c "import smaclite, gymnasium as gym; print(sorted(s for s in gym.envs.registry if 'smac' in s))"
```

You should see ~13 registered scenarios under the `smaclite/` namespace, e.g. `smaclite/2s_vs_1sc-v0`, `smaclite/2s3z-v0`, `smaclite/3s5z-v0`, `smaclite/MMM-v0`.

### Smoke test (2 minutes)

```bash
cd epymarl
python src/main.py --config=qmix --env-config=smaclite \
  with env_args.map_name=2s_vs_1sc \
  t_max=5000 test_interval=100000 log_interval=2500 \
  save_model=False use_tensorboard=False
```

### Train + evaluate via our wrappers

`exp_train.py` / `exp_attack.py` / `exp_transfer.py` all have `smaclite_2s_vs_1sc` in their `ENV_MAP`. Drive the sweeps the same way as for MPE, just pass `ENV=`:

```bash
python exp_train.py --algo qmix --sharing shared --env smaclite_2s_vs_1sc --seed 1 \
  --t_max 1000000 --save_model_interval 100000

ENV=smaclite_2s_vs_1sc bash scripts/run_train_sweep.sh
ENV=smaclite_2s_vs_1sc bash scripts/run_attack_sweep.sh
ENV=smaclite_2s_vs_1sc bash scripts/run_transfer_sweep.sh
```

Adding more maps takes one `ENV_MAP` entry per new scenario in each of the three `exp_*.py` files — see the `smaclite_2s_vs_1sc` entry in `exp_train.py` for the shape (`env_config: smaclite`, `env_args: {map_name, time_limit, use_cpp_rvo2}`).

### Performance

On the RTX 3060 with the default numpy RVO2 backend:

- `2s_vs_1sc` (2 agents, smallest): **~33 env steps/sec** → t_max=500K ≈ 4 hours per cell
- `2s3z` (5 agents): **~5.6 env steps/sec** → t_max=500K ≈ 25 hours per cell

For comparison, MPE `simple_spread` runs at ~167 steps/sec. SMAClite is CPU-bound by the RVO2 collision-avoidance solver. The C++ RVO2 backend (`use_cpp_rvo2: True`) is reportedly 5–10× faster but requires Visual Studio Build Tools to compile on Windows — not currently used.

## Notes for collaborators

- Results files (`epymarl/results/`, `results/`, `*.pth`) are **git-ignored** on purpose — we reproduce them from configs + seeds, not from stored artifacts.
- The EPyMARL fork has one Windows-compat patch:
  - `epymarl/src/envs/__init__.py` — makes the `smaclite` import optional (so MPE-only runs don't require the SMAClite package even though we now install it for the SMAC stretch experiments).
  Two earlier patches that redirected sacred output and tb_logs into per-cell directories were reverted to match the upstream-default layout that the kamen pipeline assumes.
  If you rebase onto upstream EPyMARL, re-apply the optional-import patch.
- **When running the shared-vs-independent ablation**, EPyMARL's default `_ns` configs also change `use_rnn` (and for IQL, `epsilon_anneal_time`) — these are confounds that must be equalized via sacred overrides. See `CLAUDE.md` → "Sharing toggle" for the exact per-algorithm override strategy.
- Deadline is tight (9 days from 2026-04-21). See `CLAUDE.md` → "Implementation roadmap" for the day-by-day plan. Scope is **MPE only, 6 algorithms, 2 sharing modes, 3 attacks**; SMAC is a stretch goal, not budgeted for.

## Contact

- Guillaume Genois — guillaume.genois@umontreal.ca
- Kamen Damov — kamen.damov@umontreal.ca
