"""One-time migration: old per-cell layout -> new pipeline layout.

Old (with our previous epymarl patches):
  results/<env>/<algo>/<sharing>/seedN/epymarl/models/<token>/<step>/agent.th
  results/<env>/<algo>/<sharing>/seedN/epymarl/sacred/<cfg>/<env_key>/<id>/...
  results/<env>/<algo>/<sharing>/seedN/epymarl/tb_logs/<token>/...
  results/<env>/<algo>/<sharing>/seedN/attacks/<name>_eps<eps>/metrics.json

New (kamen pipeline + reverted patches):
  results/<env>/<algo>/<sharing>/seedN/models/<token>/<step>/agent.th
  epymarl/results/sacred/<cfg>/<env_key>/<id>/...   (global, run_ids re-numbered)
  epymarl/results/tb_logs/<token>/...               (global)
  results/<env>/<algo>/<sharing>/seedN/attack_<name>_eps<eps>.json
"""
import json
import re
import shutil
from pathlib import Path

REPO    = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
EPYRES  = REPO / "epymarl" / "results"

ATTACK_NAME_MAP = {"none": "no_attack", "random": "random_noise", "fgsm": "fgsm"}

(EPYRES / "sacred").mkdir(parents=True, exist_ok=True)
(EPYRES / "tb_logs").mkdir(parents=True, exist_ok=True)


def fix_local_path(local: str) -> str:
    return re.sub(r"[/\\]epymarl/?$", "", local)


def migrate_cell(cell: Path):
    env, algo, sharing, seed_dir = cell.parts[-4], cell.parts[-3], cell.parts[-2], cell.parts[-1]
    seed = int(seed_dir.replace("seed", ""))
    print(f"\n[{cell.relative_to(REPO).as_posix()}]")

    # 1. Move models up
    old_m, new_m = cell / "epymarl" / "models", cell / "models"
    if old_m.exists():
        if new_m.exists():
            for child in list(old_m.iterdir()):
                shutil.move(str(child), str(new_m / child.name))
            old_m.rmdir()
        else:
            shutil.move(str(old_m), str(new_m))
        print("  models/   migrated")

    # 2. Reformat attack JSONs
    old_a = cell / "attacks"
    if old_a.exists():
        n_atk = 0
        for atk_dir in old_a.iterdir():
            m = re.match(r"^(none|random|fgsm)_eps([0-9.]+)$", atk_dir.name)
            if not m:
                continue
            old_name, eps = m.group(1), m.group(2)
            new_name = ATTACK_NAME_MAP[old_name]
            mp = atk_dir / "metrics.json"
            if not mp.exists():
                continue
            data = json.loads(mp.read_text())
            new = {
                "algo": algo, "sharing": sharing, "env": env, "seed": seed,
                "attack": new_name, "epsilon": float(eps),
                "n_episodes": data.get("n_episodes"),
                "mean_return": data.get("return_mean"),
                "std_return":  data.get("return_std"),
                "checkpoint_timestep": data.get("model_step"),
            }
            if "returns" in data:
                new["returns"] = data["returns"]
            (cell / f"attack_{new_name}_eps{eps}.json").write_text(json.dumps(new, indent=2))
            n_atk += 1
        shutil.rmtree(old_a)
        print(f"  attacks/  migrated ({n_atk} files)")

    # 3. Sacred runs -> global, with run_id re-numbering and local_results_path fix
    old_s = cell / "epymarl" / "sacred"
    if old_s.exists():
        n_sac = 0
        for cfg_dir in list(old_s.iterdir()):
            if not cfg_dir.is_dir():
                continue
            for env_key_dir in list(cfg_dir.iterdir()):
                if not env_key_dir.is_dir():
                    continue
                target_root = EPYRES / "sacred" / cfg_dir.name / env_key_dir.name
                target_root.mkdir(parents=True, exist_ok=True)
                existing_ids = [int(d.name) for d in target_root.iterdir()
                                if d.is_dir() and d.name.isdigit()]
                next_id = max(existing_ids, default=0) + 1
                for src in list(env_key_dir.iterdir()):
                    if not src.is_dir():
                        continue
                    if src.name == "_sources":
                        target_sources = target_root / "_sources"
                        target_sources.mkdir(exist_ok=True)
                        for f in list(src.iterdir()):
                            tgt = target_sources / f.name
                            if not tgt.exists():
                                shutil.move(str(f), str(tgt))
                            else:
                                f.unlink()
                        src.rmdir()
                    elif src.name.isdigit():
                        cfg_path = src / "config.json"
                        if cfg_path.exists():
                            cfg = json.loads(cfg_path.read_text())
                            cfg["local_results_path"] = fix_local_path(
                                cfg.get("local_results_path", "")
                            )
                            cfg_path.write_text(json.dumps(cfg, indent=2))
                        shutil.move(str(src), str(target_root / str(next_id)))
                        next_id += 1
                        n_sac += 1
                if env_key_dir.exists() and not any(env_key_dir.iterdir()):
                    env_key_dir.rmdir()
            if cfg_dir.exists() and not any(cfg_dir.iterdir()):
                cfg_dir.rmdir()
        if old_s.exists() and not any(old_s.iterdir()):
            old_s.rmdir()
        print(f"  sacred/   migrated ({n_sac} runs)")

    # 4. tb_logs -> global
    old_t = cell / "epymarl" / "tb_logs"
    if old_t.exists():
        n_tb = 0
        target_tb = EPYRES / "tb_logs"
        for src in list(old_t.iterdir()):
            tgt = target_tb / src.name
            if tgt.exists():
                tgt = target_tb / f"{src.name}__seed{seed}"
            shutil.move(str(src), str(tgt))
            n_tb += 1
        old_t.rmdir()
        print(f"  tb_logs/  migrated ({n_tb} runs)")

    # 5. Clean empty <cell>/epymarl/
    leftover = cell / "epymarl"
    if leftover.exists() and not any(leftover.iterdir()):
        leftover.rmdir()


def main():
    cells = sorted(RESULTS.glob("*/*/*/seed*"))
    cells = [c for c in cells if c.is_dir()]
    print(f"[migrate] {len(cells)} cells found")
    for cell in cells:
        migrate_cell(cell)
    print("\n[migrate] DONE")


if __name__ == "__main__":
    main()
