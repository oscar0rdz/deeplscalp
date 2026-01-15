# tools/kaggle_entry.py
import os
import subprocess
import sys
from pathlib import Path


def find_candidate(project_dir: Path) -> Path:
    # Prioridad alta: rutas típicas
    preferred = [
        project_dir / "scripts" / "run_walkforward.py",
        project_dir / "scripts" / "walkforward.py",
        project_dir / "run_walkforward.py",
        project_dir / "walkforward.py",
    ]
    for p in preferred:
        if p.exists():
            return p

    # Buscar por nombre
    for p in project_dir.rglob("*.py"):
        n = p.name.lower()
        if "walkforward" in n and ("run" in n or "wf" in n or "walkforward" in n):
            return p

    # Buscar por huellas de outputs
    for p in project_dir.rglob("*.py"):
        try:
            t = p.read_text(errors="ignore").lower()
        except Exception:
            continue
        if "walkforward_summary" in t or "pred_test.parquet" in t or "artifacts/reports" in t:
            return p

    raise SystemExit("No encontré ningún runner de walkforward en el repo.")

def main():
    project_dir = Path(__file__).resolve().parents[1]  # /project
    data = os.environ.get("DATA_PARQUET", "")
    if not data:
        # fallback: intenta localizar parquet-v1 si está montado
        cand = Path("/kaggle/input/parquet-v1/train_XRP_USDT_5m_v71.parquet")
        if cand.exists():
            data = str(cand)

    if not data:
        raise SystemExit("DATA_PARQUET no definido y no se encontró el parquet por defecto.")

    runner = find_candidate(project_dir)
    print("[KAGGLE_ENTRY] project_dir:", project_dir)
    print("[KAGGLE_ENTRY] data:", data)
    print("[KAGGLE_ENTRY] runner:", runner)

    env = os.environ.copy()
    env["DATA_PARQUET"] = data

    # Intento con argumentos comunes; si falla, corre sin args.
    cmd_variants = [
        [sys.executable, str(runner), "--data", data],
        [sys.executable, str(runner), "--parquet", data],
        [sys.executable, str(runner)],
    ]

    last_err = None
    for cmd in cmd_variants:
        try:
            print("[KAGGLE_ENTRY] running:", " ".join(cmd))
            subprocess.check_call(cmd, cwd=str(project_dir), env=env)
            print("[KAGGLE_ENTRY] OK")
            return 0
        except subprocess.CalledProcessError as e:
            last_err = e
            print("[KAGGLE_ENTRY] failed:", e)

    raise SystemExit(f"No pude ejecutar el runner. Último error: {last_err}")

if __name__ == "__main__":
    raise SystemExit(main())
