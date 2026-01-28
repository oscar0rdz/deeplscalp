from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _find_first(glob_list):
    for p in glob_list:
        if p.exists():
            return p
    return None

def audit_fold(fold_dir: Path, fail: bool = True) -> dict:
    # Ajusta estos nombres si tu repo usa otros.
    candidates = [
        fold_dir / "trades.csv",
        fold_dir / "trades.parquet",
        fold_dir / "sim_trades.csv",
        fold_dir / "executions.csv",
    ]
    trades_path = _find_first(candidates)
    if trades_path is None:
        msg = f"[AUDIT] No encuentro trades en {fold_dir}. Esperaba uno de: {[p.name for p in candidates]}"
        if fail:
            raise SystemExit(msg)
        return {"fold": fold_dir.name, "ok": False, "error": msg}

    if trades_path.suffix == ".parquet":
        df = pd.read_parquet(trades_path)
    else:
        df = pd.read_csv(trades_path)

    cols = set(df.columns)
    has_any_pnl = ("pnl" in cols) or ("ret_net" in cols) or ("ret_raw" in cols)
    
    if not has_any_pnl:
        msg = "[FAIL] trades: falta pnl y no hay ret_net/ret_raw (no se puede evaluar)"
        print(msg)
        if fail:
            return {"fold": fold_dir.name, "ok": False, "error": msg, "exit_code": 2}
        return {"fold": fold_dir.name, "ok": False, "error": msg}

    if "pnl" not in cols:
        msg = "[FAIL] trades: falta columna pnl (aunque exista ret_net/ret_raw)"
        print(msg)
        if fail:
            # Obliga pnl explícito según recomendación
            return {"fold": fold_dir.name, "ok": False, "error": msg, "exit_code": 3}
        
    # fallback para el resto del script si falta pnl
    if "pnl" not in cols:
        if "ret_net" in cols:
            df["pnl"] = df["ret_net"]
        else:
            df["pnl"] = df["ret_raw"]

    required = {"entry_price", "exit_price", "pnl"}
    missing = required - set(df.columns)
    if missing:
        msg = f"[AUDIT] Trades sin columnas {missing} en {trades_path}"
        if fail:
            raise SystemExit(msg)
        return {"fold": fold_dir.name, "ok": False, "error": msg}

    n = len(df)
    if n == 0:
        msg = f"[AUDIT] 0 trades en {fold_dir}"
        if fail:
            raise SystemExit(msg)
        return {"fold": fold_dir.name, "ok": False, "error": msg}

    entry = df["entry_price"].astype(float)
    exitp = df["exit_price"].astype(float)
    pnl = df["pnl"].astype(float)

    same_px = (entry == exitp).mean()
    zero_pnl = (pnl == 0.0).mean()

    # Si tienes qty/size, también lo validamos (opcional)
    zero_qty = None
    if "qty" in df.columns:
        qty = df["qty"].astype(float).abs()
        zero_qty = (qty == 0.0).mean()

    report = {
        "fold": fold_dir.name,
        "n_trades": int(n),
        "same_entry_exit_ratio": float(same_px),
        "zero_pnl_ratio": float(zero_pnl),
        "zero_qty_ratio": None if zero_qty is None else float(zero_qty),
        "pnl_sum": float(pnl.sum()),
        "pnl_mean": float(pnl.mean()),
        "pnl_std": float(pnl.std(ddof=0)),
        "ok": True,
    }

    # Reglas fail-fast: ajusta umbrales si lo necesitas.
    problems = []
    if same_px > 0.80:
        problems.append("entry_price==exit_price demasiado frecuente (bug temporal/ejecución)")
    if zero_pnl > 0.80:
        problems.append("pnl==0.0 demasiado frecuente (bug de PnL/tick-size/fees/qty)")
    if zero_qty is not None and zero_qty > 0.10:
        problems.append("qty==0 demasiado frecuente (bug sizing/step-size)")

    report["problems"] = problems
    if problems and fail:
        raise SystemExit("[AUDIT] FAIL " + fold_dir.name + " :: " + " | ".join(problems))
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="artifacts/reports", help="directorio artifacts/reports")
    ap.add_argument("--fail", action="store_true", help="salir con error si detecta degeneración")
    args = ap.parse_args()

    base = Path(args.reports)
    folds = sorted([p for p in base.glob("fold_*") if p.is_dir()])
    if not folds:
        raise SystemExit(f"[AUDIT] No hay fold_* en {base}")

    out = []
    has_critical_error = False
    for fd in folds:
        res = audit_fold(fd, fail=args.fail)
        out.append(res)
        if res.get("exit_code"):
            has_critical_error = True
            critical_code = res["exit_code"]

    if has_critical_error and args.fail:
        print(f"[AUDIT] FAIL FAST: Critical error detected.")
        sys.exit(critical_code)

    out_path = base / "audit_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[AUDIT] escrito: {out_path}")
    for r in out:
        print(r["fold"], "n=", r["n_trades"], "same_px=", r["same_entry_exit_ratio"], "zero_pnl=", r["zero_pnl_ratio"], "pnl_sum=", r["pnl_sum"])

if __name__ == "__main__":
    main()