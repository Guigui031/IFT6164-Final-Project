"""Quick read-only inspection of SMAClite QMIX training curves from sacred."""
import json
import glob

for sharing in ["shared", "independent"]:
    cfg_subdir = f"qmix{'_ns' if sharing == 'independent' else ''}"
    cfg_glob = f"epymarl/results/sacred/{cfg_subdir}/2s_vs_1sc/*/config.json"
    for cfg_path in sorted(glob.glob(cfg_glob)):
        cfg = json.load(open(cfg_path))
        local = cfg.get("local_results_path", "").replace("\\", "/")
        expected = f"smaclite_2s_vs_1sc/qmix/{sharing}/seed1"
        if cfg.get("seed") == 1 and local.endswith(expected):
            metrics_path = cfg_path.replace("config.json", "metrics.json")
            m = json.load(open(metrics_path))
            r = m.get("test_return_mean", {})
            w = m.get("test_battle_won_mean", {})
            steps = r.get("steps", [])
            vals = r.get("values", [])
            wsteps = w.get("steps", [])
            wvals = w.get("values", [])
            print(f"\n=== qmix/{sharing}/seed1 ===")
            print(f"  test_return_mean: {len(vals)} points, final={vals[-1]:.2f}" if vals else "  no test_return_mean")
            for i in range(0, len(vals), max(1, len(vals)//8)):
                print(f"    step={steps[i]:>8d}  return={vals[i]:.2f}")
            if vals:
                print(f"    step={steps[-1]:>8d}  return={vals[-1]:.2f}  (final)")
            print(f"  test_battle_won_mean: {len(wvals)} points, final={wvals[-1]:.2f}" if wvals else "  no test_battle_won_mean")
            for i in range(0, len(wvals), max(1, len(wvals)//8)):
                print(f"    step={wsteps[i]:>8d}  win_rate={wvals[i]:.2f}")
            if wvals:
                print(f"    step={wsteps[-1]:>8d}  win_rate={wvals[-1]:.2f}  (final)")
            break
