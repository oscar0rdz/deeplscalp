import pandas as pd
import json
import numpy as np
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="artifacts/reports", help="Path to reports directory")
    args = parser.parse_args()

    REP = Path(args.reports)
    aud_path = REP / "walkforward_summary_audited.csv"
    
    # Fallback to standard summary if audited not found
    if not aud_path.exists():
        aud_path = REP / "walkforward_summary.csv"
        
    if not aud_path.exists():
        print(f"[FAIL] No summary csv found in {REP}")
        sys.exit(1)

    print(f"[AUDIT] Reading summary from {aud_path}")
    df_a = pd.read_csv(aud_path)

    rows = []
    errors = []

    for _, r in df_a.iterrows():
        try:
            # FIX: Robust fold casting
            fold_val = r["fold"]
            try:
                fold_id = int(round(float(fold_val)))
            except ValueError:
                fold_id = str(fold_val)

            fd = REP / f"fold_{fold_id}"
            
            # Check fold existence
            if not fd.exists():
                # Try simple int
                if isinstance(fold_id, str):
                    try:
                        fd_alt = REP / f"fold_{int(float(fold_id))}"
                        if fd_alt.exists():
                            fd = fd_alt
                            fold_id = int(float(fold_id))
                    except:
                        pass
            
            if not fd.exists():
                errors.append(f"Fold dir not found: {fd}")
                continue

            bm_path = fd / "best_metrics.json"
            if not bm_path.exists():
                errors.append(f"best_metrics.json missing in {fd}")
                # Try looking for just metrics.json
                if (fd/"metrics_test.json").exists():
                     pass # metrics_test is unlikely the one we want matching training summary
                continue

            bm = json.loads(bm_path.read_text())
            
            # Extract safe values
            aud_net = float(r.get("net", np.nan))
            bm_net = float(bm.get("net", np.nan))
            
            aud_mdd = float(r.get("mdd", np.nan))
            bm_mdd = float(bm.get("mdd", np.nan))
            
            aud_pf = float(r.get("profit_factor", np.nan)) 
            # Check raw vs calc
            if "profit_factor_raw" in r:
                 aud_pf_raw = float(r["profit_factor_raw"])
            else:
                 aud_pf_raw = aud_pf
                 
            bm_pf = float(bm.get("profit_factor", np.nan))
            
            aud_ntr = int(float(r.get("n_trades", -1)))
            bm_ntr = int(bm.get("n_trades", -1))

            rows.append({
                "fold": fold_id,
                "aud_net": aud_net,
                "bm_net": bm_net,
                "aud_mdd": aud_mdd,
                "bm_mdd": bm_mdd,
                "aud_pf": aud_pf,
                "bm_pf": bm_pf,
                "aud_ntr": aud_ntr,
                "bm_ntr": bm_ntr,
                "net_diff": aud_net - bm_net,
                "ntr_diff": aud_ntr - bm_ntr
            })

        except Exception as e:
            errors.append(f"Error processing row {r.get('fold', '?')}: {e}")

    if errors:
        print("\n[ERRORS]")
        for e in errors:
            print("  -", e)

    if not rows:
        print("[WARN] No rows processed successfully.")
        return

    cmp = pd.DataFrame(rows)
    # Reorders columns
    cols = ["fold", "aud_ntr", "bm_ntr", "ntr_diff", "aud_net", "bm_net", "net_diff", "aud_mdd", "bm_mdd"]
    # Add PF if space
    cols += ["aud_pf", "bm_pf"]
    
    print("\n[COMPARISON TABLE]")
    print(cmp[cols].to_string(index=False))
    
    # Invariant checks
    mismatch_ntr = cmp[cmp["ntr_diff"] != 0]
    if not mismatch_ntr.empty:
        print(f"\n[FAIL] Mismatch in n_trades count for {len(mismatch_ntr)} folds!")
        print(mismatch_ntr[["fold", "aud_ntr", "bm_ntr"]])
    else:
        print("\n[OK] All n_trades match.")

    # Net diff check
    max_net_diff = cmp["net_diff"].abs().max()
    print(f"\nMax Net Diff: {max_net_diff:.2e}")
    if max_net_diff > 1e-9:
        print("[WARN] Net return mismatch > 1e-9")
    else:
        print("[OK] Net return consistent.")

if __name__ == "__main__":
    main()
