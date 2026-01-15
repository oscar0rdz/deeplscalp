from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml

# Importa tu runner existente para reutilizar tu lógica interna
# Ajusta estos imports según tu repo (esto es lo único que podrías tener que tocar una vez)
from scripts.run_v71_walkforward import main as wf_main  # tu runner actual


def load_cfg(p: str) -> dict:
    with open(p, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--pair", default=None)
    ap.add_argument("--tf", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-folds", type=int, default=None)
    args, unknown = ap.parse_known_args()

    # Default config
    cfg_path = args.config or ("_kaggle_v71_wf_realistic.yaml" if Path("_kaggle_v71_wf_realistic.yaml").exists() else "_kaggle_v71_wf_tune.yaml")
    if not Path(cfg_path).exists():
        raise SystemExit(f"No existe config: {cfg_path}")

    cfg = load_cfg(cfg_path)
    pair = args.pair or cfg.get("pair", "XRP_USDT")
    tf = args.tf or cfg.get("tf", "5m")
    max_folds = args.max_folds or cfg.get("train", {}).get("max_folds", 3)

    # Ejecuta tu runner original pasándole argumentos "completos"
    # Nota: si tu main() toma sys.argv, esto funciona bien; si no, ajusta.
    os.environ["DEEPLSCALP_CONFIG"] = str(cfg_path)

    # Llamamos tu runner original como CLI (más robusto)
    import subprocess
    import sys
    cmd = [
        sys.executable, "scripts/run_v71_walkforward.py",
        "--config", str(cfg_path),
        "--pair", str(pair),
        "--tf", str(tf),
        "--device", str(args.device),
        "--max-folds", str(max_folds),
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)

    # Validación del summary
    rep = Path("artifacts/reports")
    summ = rep / "walkforward_summary.csv"
    if not summ.exists():
        raise SystemExit("No se generó artifacts/reports/walkforward_summary.csv")
    df = pd.read_csv(summ)
    print("\n[OK] Summary head:")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
