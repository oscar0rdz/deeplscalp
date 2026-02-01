import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import sys

# Ensure project root is in path
PROJECT_DIR = Path(".").resolve()
sys.path.append(str(PROJECT_DIR))

from deeplscalp.backtest.sim_v71 import backtest_from_predictions_v71

ARTIFACTS_DIR = PROJECT_DIR / "artifacts/reports"
FOLD = 0 

def prep_df(p: Path):
    if not p.exists():
        print(f"[ERR] File not found: {p}")
        return None
    
    print(f"[INFO] Reading parquet: {p}")
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print(f"[ERR] Failed to read parquet: {e}")
        return None

    # Handle index/ds
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ds" in df.columns:
            print("[INFO] Converting 'ds' column to DatetimeIndex...")
            t = pd.to_datetime(df["ds"], utc=True, errors="coerce")
            df = df.loc[~t.isna()].copy()
            df.index = pd.DatetimeIndex(t.loc[~t.isna()], name="ds")
            df = df.drop(columns=["ds"])
        else:
            print("[WARN] No 'ds' col and no DatetimeIndex. Sim might fail if not fixed.")
    else:
        print("[INFO] DatetimeIndex detected suitable.")
        
    return df

def run_test():
    print("=== STARTING ROBUST VERIFICATION ===")
    
    # 1. Load data
    # Priority: pred_XRP... (verified exist) -> fold_0/pred_test...
    p1 = PROJECT_DIR / "artifacts/preds/pred_XRP_USDT_v71.parquet"
    p2 = ARTIFACTS_DIR / f"fold_{FOLD}/pred_test.parquet"
    
    pred_path = p1 if p1.exists() else p2
    
    df_base = prep_df(pred_path)
    if df_base is None:
        print("[FAIL] Could not load any dataframe.")
        return

    # Slice for speed (first 25k rows ~ 2 months of 5m data)
    MAX_ROWS = 25000
    if len(df_base) > MAX_ROWS:
        print(f"[INFO] Slicing data from {len(df_base)} to {MAX_ROWS} rows for speed.")
        df_base = df_base.iloc[:MAX_ROWS].copy()
    else:
        print(f"[INFO] Using full data: {len(df_base)} rows.")

    # Base Config
    cfg = {
        "sim": {
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "exec_lag": 0
        },
        "risk": {
            "leverage": 1.0, 
            "risk_fraction": 0.1
        }
    }
    thresholds = {}

    # ---------------------------------------------------------
    # TEST 1: EXEC LAG (Lookahead outcome)
    # ---------------------------------------------------------
    print("\n--- TEST 1: EXEC LAG (Anti-Lookahead) ---")
    
    # Lag=0
    cfg["sim"]["exec_lag"] = 0
    print("Running lag=0...")
    m0, _ = backtest_from_predictions_v71(df_base, cfg, thresholds)
    net0 = m0.get('net', 0)
    pf0 = m0.get('profit_factor', 0)
    print(f"-> Lag=0 | Net: {net0:.4f}, PF: {pf0:.2f}")
    
    # Lag=1
    cfg["sim"]["exec_lag"] = 1
    print("Running lag=1...")
    m1, _ = backtest_from_predictions_v71(df_base, cfg, thresholds)
    net1 = m1.get('net', 0)
    pf1 = m1.get('profit_factor', 0)
    print(f"-> Lag=1 | Net: {net1:.4f}, PF: {pf1:.2f}")
    
    if net1 < net0:
        print(f"[PASS] Performance degraded with lag=1 (Diff: {net1-net0:.4f}). Realism check passed.")
    elif abs(net1 - net0) < 1e-9:
        print("[WARN] No change with lag=1. Suspicious if strategy tracks price momentum.")
    else:
        print(f"[WARN] Performance IMPROVED with lag? ({net1:.4f} > {net0:.4f}). Unusual.")


    # ---------------------------------------------------------
    # TEST 2: FEES (Cost application)
    # ---------------------------------------------------------
    print("\n--- TEST 2: FEE SENSITIVITY ---")
    cfg["sim"]["exec_lag"] = 1 # Keep sane default
    
    # Fee=0
    cfg["sim"]["fee_bps"] = 0.0
    print("Running fee=0...")
    m_fee0, d_fee0 = backtest_from_predictions_v71(df_base, cfg, thresholds)
    net_fee0 = m_fee0.get('net', 0)
    
    # Fee=10
    cfg["sim"]["fee_bps"] = 10.0
    print("Running fee=10...")
    m_fee10, d_fee10 = backtest_from_predictions_v71(df_base, cfg, thresholds)
    net_fee10 = m_fee10.get('net', 0)
    
    print(f"-> Fee=0  | Net: {net_fee0:.4f}")
    print(f"-> Fee=10 | Net: {net_fee10:.4f}")
    
    # Check robustness in diag
    if "pnl_net" in d_fee10["trades_df"].columns:
        print("[CHECK] 'pnl_net' column confirmed in trades dataframe.")
    else:
        print("[FAIL] 'pnl_net' MISSING in trades dataframe.")

    if net_fee10 < net_fee0:
        print(f"[PASS] Net decreased with fees (Diff: {net_fee10-net_fee0:.4f}).")
    else:
        print(f"[FAIL] Net did NOT decrease with fees. Logic broken.")


    # ---------------------------------------------------------
    # TEST 3: PLACEBO (Shuffle)
    # ---------------------------------------------------------
    print("\n--- TEST 3: PLACEBO (Shuffle) ---")
    # Use fees=5, lag=1
    cfg["sim"]["fee_bps"] = 5.0
    cfg["sim"]["exec_lag"] = 1
    
    df_shuf = df_base.copy()
    # Shuffle prediction columns
    pred_cols = [c for c in df_shuf.columns if any(x in c for x in ["pred", "p_", "qL", "qS"])]
    
    if not pred_cols:
        print("[WARN] No prediction columns found to shuffle!")
    else:
        print(f"Shuffling {len(pred_cols)} columns...")
        # Shuffle values within columns independently to break correlations
        for c in pred_cols:
            df_shuf[c] = np.random.permutation(df_shuf[c].values)
            
        print("Running shuffled...")
        m_shuf, _ = backtest_from_predictions_v71(df_shuf, cfg, thresholds)
        net_sh = m_shuf.get('net', 0)
        pf_sh = m_shuf.get('profit_factor', 0)
        
        print(f"-> Shuffle | Net: {net_sh:.4f}, PF: {pf_sh:.2f}")
        
        if abs(net_sh) < 0.2 and pf_sh < 1.3:
            print("[PASS] Placebo collapsed. Signal dependency confirmed.")
        elif pf_sh >= 1.3:
            print(f"[FAIL] Placebo still profitable (PF={pf_sh:.2f}). Market bias or bug?")
        else:
            print("[PASS] Placebo looks random/unprofitable.")

    print("\n=== VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    run_test()
