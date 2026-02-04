# deeplscalp/tools/audit_wf.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from deeplscalp.reporting.metrics import compute_fold_metrics

def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def normalize_fold(x):
    # soporta "0", 0, 0.0, "0.0"
    try:
        return int(float(x))
    except:
        return x

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
        tr = fd/"best_trades.csv"
        eq = fd/"best_equity.csv"
        if not tr.exists(): tr = fd/"trades.csv"
        if not eq.exists(): eq = fd/"equity.csv"

        dtr = load_csv(tr)
        deq = load_csv(eq)

        m = compute_fold_metrics(dtr, deq if len(deq) else None)

        valid_min_trades = (m.n_trades >= args.min_trades)

        rows.append({
            "fold": fd.name.replace("fold_", ""),
            "n_trades": m.n_trades,
            "winrate": m.winrate,
            "net": m.net,
            "equity_final": m.equity_final,
            "mdd": m.mdd,
            "profit_factor_raw": m.pf_raw,
            "gross_profit": m.gross_profit,
            "gross_loss": m.gross_loss,
            "pnl_col_used": m.pnl_col_used,
            "equity_col_used": (m.equity_col_used or ""),
            "valid_min_trades": bool(valid_min_trades),
            "flags": json.dumps(m.flags, ensure_ascii=False),
        })

    df = pd.DataFrame(rows)
    out = rep / args.out
    df.to_csv(out, index=False)

    print("\n[audit_wf] == audited summary ==")
    print(df.to_string(index=False))
    print("\n[audit_wf] wrote:", out)

if __name__ == "__main__":
    main()
