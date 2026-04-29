- Create environmnent file with all the dependencies.
- Run baseline MAPPO with no perturbation: with and without param sharing.
- Devise basic attacks on observations: Random noise and FGSM.
- Plot the results

Hypothesis:
If parameters across agents are shared, perturbation in the env will crash the whole system -> Coop Multi Agent will get less rewards, than independently trained models.

The logic is that a well devised attack can lead to catastrophic results in shared setting. You need to devise a single attack to crash the system, vs devising a distinct attack per agent.

We perturb the observations only, so we can use a centralised critic in any case. The ablation study is done on the shared vs non-shared parameters.


## Tomorrow morning (after overnight seed=1 sweep)

Overnight job: `SEEDS="1" T_MAX=1050000 bash scripts/run_train_sweep.sh` (started 2026-04-28 ~22:50, ETA ~6am, ~7h, 10 cells).

### 1. Verify overnight training succeeded
```bash
# expect 10 cells with checkpoints near step 1000025
ls -d results/mpe_simple_spread/*/*/seed1/models/*/1000025 | wc -l   # should print 10
# any train.log with errors?
grep -l -E "Error|Traceback|FAILED" results/mpe_simple_spread/*/*/seed1/train.log
```

### 2. Run seed=1 attack evaluations (~45 min)
```bash
SEEDS="1" EPSILONS="0.05 0.1 0.25 0.5" bash scripts/run_attack_sweep.sh
```
Produces `results/<env>/<algo>/<sharing>/seed1/attack_<name>_eps<eps>.json` for each of 10 cells × {no_attack, random_noise, fgsm} × 4 epsilons = ~90 evals.

### 3. Run seed=1 transfer matrices (~7 min)
```bash
SEEDS="1" EPSILON=0.25 N_EPISODES=50 bash scripts/run_transfer_sweep.sh
```
Produces `results/<env>/<algo>/<sharing>/seed1/transfer_eps0.25.json` for 10 cells.

### 4. (Optional, ~2.6 h) SDOR training for seeds {1, 2, 3}
SDOR (Option B) trains against the **shared MAPPO** protagonist, then transfers to all other cells via STor's per-target gradient.
```bash
for s in 1 2 3; do
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe exp_sdor_train.py \
    --algo mappo --env mpe_simple_spread --seed $s \
    --n_train_episodes 10000
done
```
Each takes ~52 min. Output: `results/sdor/mpe_simple_spread/mappo/shared/seed<N>/sdor.pt`.

### 5. (Optional, ~45 min) SDOR-stor attack eval across all cells
Once SDORs exist, evaluate them as the `sdor_stor` attack at multiple epsilons:
```bash
# TODO: run_attack_sweep.sh does NOT yet include sdor_stor — needs an inner-loop
# addition that passes --attack sdor_stor --sdor_ckpt <path> for each cell.
# Quick version: write a small loop that iterates cells × epsilons and calls
# exp_attack.py --attack sdor_stor --sdor_ckpt results/sdor/.../seed<N>/sdor.pt
```

### 6. Re-aggregate (~10 sec)
```bash
.venv/Scripts/python.exe exp_aggregate.py --focus_epsilon 0.25
```
Refreshes `figures/{results_table.tex, transfer_table.tex, attack_curves.png, attack_drop_bar.png, transfer_heatmaps.png, aggregate.json}` with n=3 seed data.

### 7. Report writing (binding constraint, days 7-9)
- Port intro + related work + methods from `extended-abstract/extended_abstract.tex` into `report/report.tex`.
- `\input{../figures/results_table.tex}` and `\input{../figures/transfer_table.tex}` in the Results section.
- Discussion: lead with the cleanest contrast (QMIX TR shared 0.34 vs independent 0.09; MAPPO TR shared 0.89 vs independent 0.42). Caveat n=3 seeds; single environment; SDOR-stor as stretch.

### 8. (Optional housekeeping)
- Delete `kamen/mappo-baseline` branch on origin (still carries the trailer in its history): `git push origin --delete kamen/mappo-baseline`.
- Delete `backup-before-rewrite` local branch and `refs/original/*` once confident in the rewrite.
- Update poster placeholders in `poster/horz/poster.tex` and copy `figures/transfer_heatmaps.png` to `poster/img/`.

### Things explicitly skipped (low ROI given deadline)
- More ε values for transfer matrix (only 0.25 — focus_epsilon is enough).
- Re-running migrated `random_noise` / `no_attack` (deltas were within RNG noise).
- MADDPG (continuous-action MPE; needs `gymma.py` patch — several hours of risk for one extra row).
- A second environment (SMAC) — pipeline is env-agnostic but no time.

## Nice-to-have run options (only if time permits after the main report)

Sorted roughly by `value / effort`. None of these are required for the report; they would strengthen specific claims or open follow-up directions.

### Quick wins (under 1 hour)
- **Tighter eval CI**: bump `N_EPISODES=100 → 200` for the attack sweep at the focus epsilon (0.25). Halves SEM at the cost of ~30 min for 60 evals (3 seeds × 10 cells × 2 attacks). Helps the headline table only.
- **Transfer at additional ε** (0.05, 0.1, 0.5): adds a transfer-ratio-vs-ε plot for the discussion. ~15 min per ε with seeds {2,3} only; ~22 min with all 3 seeds. `SEEDS="1 2 3" EPSILON=<e> N_EPISODES=50 bash scripts/run_transfer_sweep.sh`.
- **Random noise sanity-rerun**: even though deltas were ≤1 pt, re-running gives a fully consistent set on disk. ~50 min for 80 evals (3 seeds × 10 cells × 4 ε but skip-if-done already covers most).

### Medium effort (1-3 hours)
- **5 seeds total** (add seeds 4, 5): brings SEM well above CLAUDE.md's "3 minimum". ~14 hours of overnight training (2 × 7 h), then ~1 h evals. Big credibility boost; real wall-clock cost.
- **SDOR-stor across multiple ε**: currently SDOR is trained at one ε. Re-training at ε ∈ {0.05, 0.25, 0.5} answers "does SDOR generalize across budgets?" (~52 min × 3 ε × 3 seeds = ~7 h, but only one seed needed for a calibration plot).
- **SDOR Option B+ (algo-specialist SDORs)**: train SDOR against shared-IQL and shared-QMIX in addition to shared-MAPPO. Compares whether a "specialist" SDOR (matched to protagonist) outperforms a "generalist" via STor transfer. Each adds ~52 min × N_seeds.

### Larger experiments (>3 hours, >1 day)
- **SDOR Option A (per-cell SDORs)**: train one SDOR per (algo, sharing, seed) cell, eval directly without STor transfer. ~26 h for 3 seeds. Answers whether STor's transfer is the source of any drop in SDOR efficacy across cells. Mostly an ablation; not a headline result.
- **MADDPG cell**: patch `epymarl/src/envs/gymma.py` to handle Box action spaces, then train MADDPG (continuous-action MPE simple_spread). ~2-4 h debugging + ~2 h training. Adds the 11th cell from the original CLAUDE.md matrix.
- **SMAC `3m` cell**: pipeline is env-agnostic — add SMACv1 dependencies (`pip install smac`), write `configs/envs/smac_3m.yaml`, train one calibration cell to confirm SMAC StarCraft engine works on Windows. ~4-8 h, probably too much risk this late.
- **Mid-training transfer-vs-step plot**: re-train with `--save_model_interval 100000` to get 10 intermediate checkpoints per cell, then run transfer matrix on each. Gives a "transferability emerges as training proceeds" plot. ~7 h re-train + ~70 min × N_cells × 10 steps for transfer evals.

### Methodological / robustness ablations (low priority; future work)
- **Adversarial training (ATLA-style)**: train protagonists *with* FGSM-perturbed observations during training, compare to clean-trained baselines. Complete experiment matrix doubles in size.
- **Robust MAC architectures**: try Lipschitz-bounded layers or randomized smoothing in the agent network.
- **Other MPE scenarios**: `simple_tag` (predator-prey, mixed cooperative/competitive), `simple_world_comm`, `simple_reference`. Tests whether the parameter-sharing-→-shared-vulnerability story is task-dependent.
- **Higher t_max for value-based**: clean returns for QMIX/VDN (-46) suggest under-convergence vs PPO-family (-50). Train QMIX/VDN to t_max=2M, see if the transfer ratio comparison shifts.
- **Larger N (more agents)**: simple_spread can run with 5-10 agents. Test whether the shared-vs-independent contrast scales with N.

### SMAC / SMAClite extensions (started 2026-04-28: only QMIX shared+indep on `2s_vs_1sc`, t_max=500K, seed=1)

What we shipped: a single (algo, sharing) pair on the smallest smaclite map, one seed, three attacks (no_attack, random_noise, fgsm). Calibration showed `2s_vs_1sc` runs at ~33 steps/sec on the 3060 (vs ~167 for MPE), so larger experiments are 5–30× more expensive than MPE.

- **Second algorithm on the same map** (`VDN` shared+indep on `2s_vs_1sc`): tests whether the QMIX result holds with a simpler mixer. ~8 h training + ~1 h eval.
- **Actor-critic family on SMAC** (`MAPPO` shared+indep on `2s_vs_1sc`): only value-based was tested; adding MAPPO gives the family-vs-family contrast we have on MPE. ~8 h.
- **More SMAC maps**: only `2s_vs_1sc` (2 agents, smallest) was tried. Other smaclite maps available: `2s3z` (5 ag, ~5.6 steps/sec ≈ 30× slower), `3s5z` (8 ag), `MMM` (10 ag, mixed unit types), `MMM2`, `corridor`, `3s_vs_5z`. None tried. Larger maps would test whether the shared-vs-indep contrast scales with N agents.
- **More SMAC seeds**: only seed=1 was trained. Per-cell SMAC training is ~4 h on the smallest map; even 3 seeds (the CLAUDE.md minimum) would be ~24 h for the QMIX pair alone.
- **Longer SMAC training**: t_max=500K is partial-convergence on `2s_vs_1sc`; standard SMAC papers use 1M–2M. ~8–16 h per cell at our throughput.
- **SMAC-tuned ε grid**: we used the same `{0.05, 0.1, 0.25, 0.5}` as MPE for direct comparability, but SMAC observations are normalised differently — a SMAC-native grid like `{0.01, 0.05, 0.1, 0.25}` may better match the literature.
- **SDOR on SMAC**: never attempted. SDOR per-episode cost on SMAC will be much higher than on MPE (env step is ~5× slower); 10K episodes × ~1.5 s/ep ≈ 4 h per seed. Risk: untested code path on SMAC; may need debugging beyond the CLI loosening we did for QMIX.
- **`use_cpp_rvo2: True`**: the calibration ran with the numpy RVO2 collision-avoidance backend (the default). Switching to the C++ port could give 5–10× speed-up but requires Visual Studio Build Tools on Windows — risky compile.
- **Full SMAC (PySC2 + StarCraft II)**: smaclite is a Python clone, not the canonical SMAC. A "real-SMAC" appendix run would require ~30 GB StarCraft II install + ~12–48 h training per cell. Out of scope for the deadline; mention as future work.
- **SMACv2** (`samvelyan/smacv2`): newer randomised scenarios, more challenging benchmark, same StarCraft II requirement. Future work.

### Pipeline / engineering nice-to-haves
- **Make `exp_attack.py` and `exp_transfer.py` ENV_MAP entries match `exp_train.py`**: only `exp_train.py` was extended to handle SMAClite for the overnight run. Before running attack/transfer eval on the SMAC checkpoints, copy the `smaclite_2s_vs_1sc` ENV_MAP entry into the other two scripts (and refactor their `ENV_MAP` to use the same `env_args` dict shape).
- **Refactor `ENV_MAP` into a shared module** (`src/utils/env_map.py`): currently duplicated across 3 entry points. Trivial refactor; eliminates a class of "added env to one file but not the others" bugs.
- **Skip-if-done across attacks**: `scripts/run_attack_sweep.sh` checks file existence per (cell, attack, eps). If the bash entry-point is ever scripted to also call `exp_train.py`, dedupe the existence check.
- **`exp_aggregate.py` env-aware multi-env scan**: aggregator currently scans `results/<env>/...` but the LaTeX tables and figures are written without an env qualifier in the filename. Re-running `exp_aggregate.py --env smaclite_2s_vs_1sc` will *overwrite* the MPE-derived `figures/results_table.tex`. Save outputs under `figures/<env>/...` to keep both.
