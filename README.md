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

## Running a full baseline (reproduce the sanity training)

```bash
cd epymarl
python src/main.py --config=iql --env-config=gymma \
  with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3" \
  save_model=True use_tensorboard=True \
  label="baseline_iql_shared_seed1" seed=1
```

Outputs land in `epymarl/results/`:
- `models/<run_token>/` — a `.pt` checkpoint every 50k steps
- `tb_logs/<run_token>/` — tensorboard scalars
- `sacred/<run_id>/` — sacred's run metadata (config, stdout, scalars)

View the curves with:

```bash
tensorboard --logdir epymarl/results/tb_logs
```

## Notes for collaborators

- Results files (`epymarl/results/`, `results/`, `*.pth`) are **git-ignored** on purpose — we reproduce them from configs + seeds, not from stored artifacts.
- The EPyMARL fork has two necessary patches for Windows/modern-dep compat:
  - `epymarl/src/envs/__init__.py` — makes the `smaclite` import optional (we don't install SMAClite).
  - `epymarl/src/run.py` — strips colons from the timestamp used in log directory names (invalid on Windows filesystems).
  If you rebase onto upstream EPyMARL, re-apply these.
- **When running the shared-vs-independent ablation**, EPyMARL's default `_ns` configs also change `use_rnn` (and for IQL, `epsilon_anneal_time`) — these are confounds that must be equalized via sacred overrides. See `CLAUDE.md` → "Sharing toggle" for the exact per-algorithm override strategy.
- Deadline is tight (9 days from 2026-04-21). See `CLAUDE.md` → "Implementation roadmap" for the day-by-day plan. Scope is **MPE only, 6 algorithms, 2 sharing modes, 3 attacks**; SMAC is a stretch goal, not budgeted for.

## Contact

- Guillaume Genois — guillaume.genois@umontreal.ca
- Kamen Damov — kamen.damov@umontreal.ca
