# deeplscalp/tools/audit_wf.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

# [PATCH] Use unified metrics
from deeplscalp.metrics import compute_metrics_from_trades

def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=str, default="artifacts/reports", help="ruta a artifacts/reports")
    ap.add_argument("--min-trades", type=int, default=200)
    ap.add_argument("--out", type=str, default="walkforward_summary_audited.csv")
    args = ap.parse_args()

    rep = Path(args.reports).resolve()
    if not rep.exists():
        raise SystemExit(f"[audit_wf] No existe reports dir: {rep}")

    folds = sorted([p for p in rep.glob("fold_*") if p.is_dir()])
    if not folds:
        raise SystemExit(f"[audit_wf] No hay fold_* en: {rep}")

    rows = []
    for fd in folds:
        # Prioridad: best_trades.csv -> trades.csv
        tr = fd/"best_trades.csv"
        if not tr.exists():
            tr = fd/"trades.csv"
        
        # Para equity, recalculamos desde trades para ser consistentes,
        # pero podemos chequear si existe best_equity.csv para comparar.
        # El mandato es: audit_wf usa MISMA lógica quemetrics.py.
        
        dtr = load_csv(tr)
        
        # Determine PnL col
        pnl_col = "pnl_net"
        if "pnl_net" not in dtr.columns:
            if "pnl" in dtr.columns: pnl_col = "pnl"
            elif "ret_net" in dtr.columns: pnl_col = "ret_net"
            
        if pnl_col not in dtr.columns:
            print(f"[audit] Fold {fd.name}: No pnl col found in {tr.name}")
            continue

        # Compute metrics
        m = compute_metrics_from_trades(
            dtr, 
            pnl_col=pnl_col, 
            equity_mode="compound", 
            pf_cap=10.0
        )
        
        valid_min_trades = (m["n_trades"] >= args.min_trades)

        rows.append({
            "fold": fd.name.replace("fold_", ""),
            "n_trades": m["n_trades"],
            "winrate": m["winrate"],
            "net": m["net"],
            "equity_final": m["equity_final"],
            "mdd": m["mdd"],
            "profit_factor_raw": m["profit_factor_raw"],
            "profit_factor": m["profit_factor"],
            "gross_profit": m["gross_profit"],
            "gross_loss": m["gross_loss"],
            "pnl_col_used": pnl_col,
            "valid_min_trades": bool(valid_min_trades),
            "flags": json.dumps(m["flags"], ensure_ascii=False),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[audit_wf] No valid folds found. Generating empty audit.")
        df = pd.DataFrame(columns=["fold", "net", "mdd", "profit_factor", "n_trades"])
    else:
        # Reordenar para claridad
        first_cols = ["fold", "net", "mdd", "profit_factor", "profit_factor_raw", "n_trades"]
        cols = first_cols + [c for c in df.columns if c not in first_cols]
        df = df[cols]
    
    out = rep / args.out
    df.to_csv(out, index=False)

    print("\n[audit_wf] == audited summary (unified logic) ==")
    print(df.to_string(index=False))
    print("\n[audit_wf] wrote:", out)

if __name__ == "__main__":
    main()
