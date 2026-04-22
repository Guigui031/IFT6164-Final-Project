# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state of the repo

The repo is **writeup-first, code-not-yet-written**. Today it contains:

- `extended-abstract/extended_abstract.tex` — pre-registered hypothesis and experiment plan (authoritative for the research question; see "Scoped experiment matrix" below for what we actually run given compute cuts).
- `extended-abstract/references.bib`, `extended-abstract/format.sty`, `extended-abstract/Examples of extended abstracts/` — LaTeX support material.
- `report/report.tex` — skeleton of the 8-page final report (sections are `TODO` stubs).
- `papers/` — four reference PDFs (Robust MARL, MADDPG, MAPPO, Robust Adversarial RL).

There is **no Python package, no configs, no training code, no results directory yet**. Anything in the "Planned structure" section below is a to-be-created target, not something already on disk — check `ls` before assuming a path exists.

**Deadline: 9 days from project start.** See the roadmap section — this drives every scope decision on this page.

## Research question (ground truth)

From `extended-abstract/extended_abstract.tex`: **does parameter sharing in cooperative MARL amplify or dampen adversarial vulnerability?** Prior robust-MARL work uses parameter sharing by default and never isolates it as a variable — this project does.

### Scoped experiment matrix (what we actually run)

The extended abstract names 6 algorithms × 2 sharing × 2 environments × 3 attacks. The realized compute budget (see below) forces cuts but keeps the full algorithm set. The pre-registered question is unchanged; any remaining cuts are disclosed in the final report.

- **Algorithms (6):** IQL, IPPO, MAPPO, QMIX, VDN, MADDPG — all from EPyMARL.
  - IQL, IPPO, MAPPO, QMIX, VDN each run in **shared** and **independent** variants (5 × 2 = 10 algo-sharing cells).
  - **MADDPG runs only as independent** — the extended abstract treats it as the naturally-non-shared reference architecture. Do not build a "shared MADDPG" variant. (+1 cell.)
  - Total: **11 algo-sharing cells**.
- **Environment (1 + 1 stretch):** **MPE cooperative navigation** is primary, using EPyMARL's bundled MPE fork.
  - Discrete actions for IQL / QMIX / VDN / IPPO / MAPPO.
  - Continuous actions for MADDPG — same scenario (`simple_spread`-equivalent), action-space toggle only.
  - **SMAC `3m`** is a stretch goal; the pipeline must stay env-agnostic so a SMAC run is "add a config," not "rewrite." No time budgeted for SMAC debugging.
- **Attacks (2 + 1 stretch):** random noise + FGSM are required. Stochastic learned adversary (Zhou et al. 2025) is a stretch; cut it if it's not working by end of day 4.
- **Seeds:** 5 per cell where possible, 3 minimum.

Total required: **11 cells × 5 seeds ≈ 55 clean training runs**, plus attack evaluations on each. Fits on one 3060 if MPE runs stay under ~30–60 min/seed; time the first IQL baseline to confirm before launching the sweep.

Metrics to report: **performance drop under attack**, **cross-agent perturbation transfer** (does a perturbation crafted for agent $i$ fool agent $j$? — direct test of whether shared weights create a shared vulnerability), and **adversary sample efficiency** (only meaningful if the learned adversary lands).

### Compute budget

Single RTX 3060 on Windows 11. **No Mila access** — the abstract's "Mila cluster + personal RTX 3060" line is out of date. Everything is local, which is why the matrix above is cut so aggressively. Wall-clock is the binding constraint, not GPU memory.

## Planned directory structure

Adapted from `ddidacus/cola-experiments` (flat `exp_*.py` entry points + YAML configs + per-run `results/` folders), with EPyMARL as the underlying training framework:

```
IFT6164-Final-Project/
├── extended-abstract/     # existing — pre-registered plan (do not drift from this)
├── report/                # existing — final 8-page report
├── papers/                # existing — reference PDFs
├── src/                   # to create — project code (our wrappers, attacks, analysis)
│   ├── attacks/           # random, FGSM, stochastic adversary implementations
│   ├── wrappers/          # observation-perturbation env wrapper for EPyMARL
│   ├── analysis/          # transfer-matrix, sample-efficiency, plotting utilities
│   └── utils/             # seeding, logging, I/O
├── epymarl/               # to add — vendored or submoduled EPyMARL (source of the 6 algos)
├── configs/               # to create — one YAML per (algo, env, sharing) cell + attack specs
│   ├── algs/              # iql.yaml, ippo.yaml, mappo.yaml, maddpg.yaml, qmix.yaml, vdn.yaml
│   ├── envs/              # smac_3m.yaml, smac_8m.yaml, mpe_simple_spread.yaml
│   ├── sharing/           # shared.yaml, independent.yaml (overrides)
│   └── attacks/           # random.yaml, fgsm.yaml, stochastic_adv.yaml
├── exp_train.py           # to create — train one (algo, env, sharing) cell, save checkpoint
├── exp_attack.py          # to create — load checkpoint, run the 3 attacks, dump metrics
├── exp_transfer.py        # to create — cross-agent perturbation transfer matrix
├── exp_adv_efficiency.py  # to create — measure adversary sample efficiency
├── exp_aggregate.py       # to create — sweep over runs, produce report tables/figures
├── scripts/               # to create
│   ├── run_train_sweep.sh       # SLURM batch for all (algo × env × sharing) cells
│   └── run_attack_sweep.sh      # SLURM batch for all attacks on all trained checkpoints
├── results/               # to create — results/<env>/<algo>/<sharing>/<seed>/ (config, weights, metrics, plots)
├── requirements.txt       # to create — pin torch, sacred, pyyaml, smac, pettingzoo, numpy, matplotlib
└── README.md              # to create
```

Key deviations from `cola-experiments` that are deliberate for this project:

- **EPyMARL is a hard dependency.** Vendor it under `epymarl/` as a fork (not a submodule) — we will patch the eval loop to inject observation perturbations. The project's own code stays thin: attack wrappers, the learned-adversary training loop, and analysis. Resist rewriting trainers.
- **A results path must include `<sharing>`.** The whole point of the study is the shared-vs-independent axis, so it must be a first-class directory component, not a config field buried inside a single run dir.
- **Attacks are evaluation-time only for the base experiments.** The protagonist is trained on clean observations first; attacks are applied at eval. Adversarial *training* (ATLA-style, Zhou-2025-style) is an optional extension, not the main experiment — don't let it accidentally become the main experiment.
- **Env-agnostic pipeline even though we only run MPE.** SMAC may come back as a stretch goal, so `exp_train.py`, attack wrappers, and analysis scripts must read env-specific details (obs shape, action space, agent count) from config, not hardcode MPE. A hardcoded MPE shortcut that saves a day now costs two days to undo later.

## Implementation roadmap (9-day schedule)

The report is due 9 days from project start (counting day 1 as setup day). Days 7–9 are report writing, so code + experiments collapse into 6 days. Everything below has a hard wall-clock budget.

- **Day 1 — Setup.** Scaffold the repo, create Python venv (CUDA-enabled torch for the 3060), vendor EPyMARL as a fork, and get one IQL-on-MPE-simple-spread baseline to produce a checkpoint with a rising reward curve. If the pipeline isn't green by end of day 1, everything downstream slips.
- **Day 2 — Sharing toggle + training sweep.** Verify EPyMARL's sharing flag produces genuinely independent networks (compare parameter counts; the "independent" variant must have ~N× the trainable params of the shared one). Write `exp_train.py` wrapper, launch the 11-cell × 5-seed ≈ 55-run clean training sweep. Let it run overnight; if total wall-clock looks like it will exceed ~16 hours, drop to 3 seeds for less-critical cells.
- **Day 3 — Attacks.** Implement random + FGSM observation wrappers. Start the stochastic adversary if time permits.
- **Day 4 — Attack eval + finish stochastic adversary (or cut).** `exp_attack.py` over all clean checkpoints × all attacks × a handful of ε values. Firm cut decision on the stochastic adversary by end of day.
- **Day 5 — Analysis.** Transfer matrix + adversary efficiency. These are the scripts that actually answer the research question — don't skimp.
- **Day 6 — Aggregation + figures.** `exp_aggregate.py` produces tables and plots. Re-run any missing seeds discovered during aggregation — this is the last day that's safe to do so.
- **Days 7–9 — Report.** Fill `report/report.tex`. Intro + related work port from the extended abstract. Discussion must compare against the pre-registered hypothesis honestly.

Day 1 blocks everything. If the sanity run on MPE is still failing at end of day 2, the project is in trouble and the scope needs to cut further (first candidates to drop: VDN, then the learned adversary, then one of MAPPO/QMIX).

## Environment (as installed)

- **Python:** 3.10.20, installed via `uv python install 3.10`.
- **Venv:** `.venv/` at repo root, created with `uv venv --python 3.10 .venv`. Activate via `.venv\Scripts\activate` (PowerShell) or source the appropriate script in bash.
- **GPU:** RTX 3060, driver 591.86 (CUDA 13.1 supported). Using PyTorch `2.2.2+cu121` — stable combo with the 3060. Upgrade paths require a matching CUDA-wheel torch build.
- **Torch/NumPy:** `torch==2.2.2+cu121` was compiled against NumPy 1.x, so NumPy is pinned to `numpy<2` (currently `1.26.4`). Don't upgrade either in isolation.
- **EPyMARL:** vendored at `epymarl/` from upstream `uoe-agents/epymarl` main. We patched `epymarl/src/envs/__init__.py` to make the `smaclite` import optional — upstream imports it unconditionally, which crashes without the package installed. If you rebase onto upstream, re-apply that patch.
- **MPE backend:** PettingZoo `1.25.0` (`pettingzoo[mpe]`). EPyMARL addresses MPE envs via keys like `pz-mpe-simple-spread-v3` through its `gymma` wrapper. PettingZoo prints a `DeprecationWarning: pettingzoo.mpe has been moved to mpe2` — ignore for now, revisit only if something breaks.
- **Dependency snapshot:** `requirements.txt` at repo root, produced by `uv pip freeze`. Re-create a working env with `uv venv --python 3.10 .venv && uv pip install -r requirements.txt` then install torch with `uv pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121` (the CUDA index is not encoded in the freeze).
- **Not installed:** SMAC, SMACv2, smaclite, PySC2, VMAS, LBF, RWARE. Keep it that way unless SMAC comes back into scope.

### Gotchas

- `sacred` imports `pkg_resources`; newer setuptools drops it. We pinned `setuptools<81` so sacred still works.
- EPyMARL's PettingZoo wrapper emits `UserWarning: reward returned by step() must be a float... actual type: <class 'list'>` — harmless but load-bearing if we ever add reward logging that assumes a scalar.
- Sacred's file observer is chatty — for the sweep we'll want to suppress or redirect it; the default dumps a per-run tree inside `epymarl/results/`.

### Smoke-test command (known-good, ~7s)

```bash
cd epymarl && \
  ../.venv/Scripts/python.exe src/main.py \
  --config=iql --env-config=gymma \
  with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3" \
  t_max=2000 test_interval=1000 log_interval=500
```

Expect TD loss ≈ 1.0, epsilon ≈ 1.0, test_return_mean around −130 (not converged, just proof the loop is live). A full default run is `t_max=2050000` ≈ 2 hours/seed on this machine.

## Commands

### LaTeX (the only commands that work today)

```bash
# From extended-abstract/ or report/:
pdflatex extended_abstract.tex
bibtex   extended_abstract
pdflatex extended_abstract.tex
pdflatex extended_abstract.tex
```

Use `latexmk -pdf extended_abstract.tex` if latexmk is available — it handles the bibtex passes automatically.

### Planned training / eval (once `src/` exists)

None of these exist yet — they're the target CLI surface:

```bash
# Train one cell (algo × env × sharing × seed)
python exp_train.py --algo iql --env smac_3m --sharing shared --seed 1

# Evaluate a checkpoint under all three attacks
python exp_attack.py --ckpt results/smac_3m/iql/shared/seed1/ --attacks random,fgsm,stochastic

# Cross-agent transfer matrix for a checkpoint
python exp_transfer.py --ckpt results/smac_3m/iql/shared/seed1/

# Aggregate everything under results/ into report tables
python exp_aggregate.py --out figures/
```

When implementing these, keep the CLI consistent across the four `exp_*.py` scripts (same `--ckpt`, `--seed`, `--out` conventions) — downstream sweep scripts depend on it.

## Sharing toggle — how to run a clean ablation

EPyMARL ships matched algo configs `<algo>.yaml` (shared) and `<algo>_ns.yaml` (independent). The non-shared variant:

- swaps `agent: rnn` → `rnn_ns` (a `ModuleList` of N independent `RNNAgent` instances)
- swaps `mac: basic_mac` → `non_shared_mac`
- sets `obs_agent_id: False` (unneeded when agents have their own weights)

Verified: for N=3 agents, independent nets have exactly 3× the params of shared — toggle is real. See `src/utils/verify_sharing.py`.

**But the `_ns` defaults also change architecture beyond sharing.** These extra differences are confounds that must be neutralized via sacred overrides when running the ablation:

| Algorithm | Shared default | `_ns` default | Fix (override on NS side) |
|---|---|---|---|
| IQL | `use_rnn: True`, `epsilon_anneal_time: 200000` | `use_rnn: False`, `epsilon_anneal_time: 50000` | `use_rnn=True epsilon_anneal_time=200000` |
| IPPO | `use_rnn: True` | `use_rnn: False` | `use_rnn=True` |
| MAPPO | `use_rnn: True` | `use_rnn: False` | `use_rnn=True` |
| QMIX | `use_rnn: False` | `use_rnn: False` | clean, no override |
| VDN | `use_rnn: False` | `use_rnn: False` | clean, no override |
| MADDPG | N/A (not run as shared) | `use_rnn: False` | use `maddpg_ns` as-is |

The `exp_train.py` wrapper (task 7) should inject these overrides automatically based on the `<algo, sharing>` pair, not require the user to remember them. If you introduce a matched `configs/algs/<algo>_shared.yaml` + `<algo>_indep.yaml` pair later, bake the equalization in there instead.

## Things to watch out for

- **Don't confuse "IPPO has independent critics" with "IPPO has independent actors."** The parameter-sharing axis in this study is about actor weights (and value nets for value-based methods). IPPO's independence refers to not having a centralized critic; its actors can still share parameters in the shared variant. Be explicit about what's being toggled.
- **EPyMARL on Windows may have rough edges.** EPyMARL's upstream is tested mostly on Linux. Expect friction with `sacred`'s file-observer paths, `multiprocessing` fork vs. spawn defaults, and line endings. If something is blocking for more than ~2 hours, switch to WSL2 rather than patching — the time budget can't absorb deep Windows debugging.
- **Seeds matter more than usual here.** Cross-agent transfer and adversary efficiency are noisy; a 3-seed average can easily flip sign on small differences. Budget ≥5 seeds for the comparisons the paper will lean on.
- **The extended abstract is pre-registered; the scope cut is disclosed, not hidden.** Results that contradict the hypothesis should be reported honestly in the Discussion section — don't silently retune the question to match the data. The scope reduction (6→4 algorithms, 2→1 environment, 3→2 required attacks) belongs in the methodology section of the final report with the "why": no Mila access, 9-day budget.
- **Two MPE action-space configs coexist.** Discrete MPE for IQL / QMIX / VDN / IPPO / MAPPO; continuous MPE for MADDPG. Same scenario, different action space. Keep this to one config-file toggle, not two cloned env configs that can drift.
- **MADDPG is independent-only.** The extended abstract treats MADDPG as the naturally-non-shared reference architecture. There is no "shared MADDPG" cell. If someone asks to add one, re-read §3 of the extended abstract.
- **Python 3.10.** EPyMARL's upstream pins are old and don't play well with 3.11+. Stick with 3.10 in the venv.
