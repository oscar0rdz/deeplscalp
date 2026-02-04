# deeplscalp/tools/doctor_v71.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=str, default="artifacts/reports")
    ap.add_argument("--min-trades", type=int, default=200)
    args = ap.parse_args()

    rep = Path(args.reports).resolve()
    aud = rep/"walkforward_summary_audited.csv"
    if not aud.exists():
        raise SystemExit(f"[doctor] Falta {aud}. Corre primero audit_wf.")

    df = pd.read_csv(aud)
    problems = []

    for _, r in df.iterrows():
        fold = r["fold"]
        ntr = int(r["n_trades"])
        mdd = float(r["mdd"])
        pf = float(r["profit_factor_raw"]) if str(r["profit_factor_raw"]) != "inf" else 9999.0
        flags = r.get("flags", "")

        if ntr < args.min_trades:
            problems.append((fold, "min_trades", f"n_trades={ntr} < {args.min_trades}"))
        if mdd < 0.002 and ntr >= 80:
            problems.append((fold, "mdd_suspicious", f"mdd={mdd:.4f}"))
        if pf > 50:
            problems.append((fold, "pf_suspicious", f"pf_raw={pf:.2f}"))
        if isinstance(flags, str) and ("fee_suspicious" in flags or "slippage_suspicious" in flags):
            problems.append((fold, "costs_suspicious", flags))

    if problems:
        print("\n[doctor] PROBLEMS DETECTED:")
        for p in problems:
            print(" - fold", p[0], "|", p[1], "|", p[2])
        raise SystemExit("[doctor] FAIL (corrige antes de live)")
    else:
        print("[doctor] OK ✅ (sin banderas rojas evidentes)")

if __name__ == "__main__":
    main()
