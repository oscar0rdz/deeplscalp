# deeplscalp/tools/leak_check_v71.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dir", type=str, required=True, help="artifacts/reports/fold_X")
    ap.add_argument("--pred", type=str, default="pred_test.parquet")
    ap.add_argument("--trades", type=str, default="best_trades.csv")
    args = ap.parse_args()

    fd = Path(args.fold_dir).resolve()
    pred_path = fd/args.pred
    trades_path = fd/args.trades

    if not pred_path.exists():
        raise SystemExit(f"No existe {pred_path}")
    if not trades_path.exists():
        raise SystemExit(f"No existe {trades_path}")

    # Objetivo: detectar “performance que no cae” al romper alineación temporal
    # Nota: aquí no re-simulamos full (depende de tu engine), pero sí detectamos síntomas.
    dp = pd.read_parquet(pred_path)
    # dt = pd.read_csv(trades_path) # Unused for now, but potentially useful

    # Reglas mínimas:
    # - pred debe tener timestamp y alguna señal clave
    cols = dp.columns.tolist()
    tcol = None
    for c in ["ts", "timestamp", "ds", "time"]:
        if c in cols:
            tcol = c; break
    if tcol is None:
        print("[LEAK] WARN: no encuentro columna tiempo en pred_test.parquet. cols=", cols[:30])

    # Si existen probabilidades o score, hacemos un “shift test” simple:
    score_col = None
    for c in ["score", "p_side", "p_long", "p_short"]:
        if c in cols:
            score_col = c; break

    if score_col is None:
        print("[LEAK] INFO: no score column found; no shift test possible.")
        return

    s = pd.to_numeric(dp[score_col], errors="coerce").fillna(0.0).values
    s0 = s.copy()
    s1 = np.roll(s, 1)     # shift +1
    s_1 = np.roll(s, -1)   # shift -1
    rnd = np.random.RandomState(7).permutation(s)

    def corr(a,b):
        a = np.nan_to_num(a); b = np.nan_to_num(b)
        if len(a) < 10: return float("nan")
        return float(np.corrcoef(a,b)[0,1])

    c01 = corr(s0, s1)
    c0m = corr(s0, s_1)
    c0r = corr(s0, rnd)

    out = {
        "score_col": score_col,
        "corr(score, shift+1)": c01,
        "corr(score, shift-1)": c0m,
        "corr(score, random)": c0r,
        "note": "Si corr(score, shift±1) es muy alta o performance no cambia con shifts, sospecha leakage o ejecución lookahead."
    }

    print("[LEAK] report:", out)
    (fd/"leak_check.json").write_text(pd.Series(out).to_json(), encoding="utf-8")
    print("[LEAK] wrote:", fd/"leak_check.json")

if __name__ == "__main__":
    main()
