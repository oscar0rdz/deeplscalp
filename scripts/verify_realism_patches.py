import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure project root is in path
PROJECT_DIR = Path(".").resolve()
sys.path.append(str(PROJECT_DIR))

from deeplscalp.backtest.sim_v71 import backtest_from_predictions_v71

def run_verification():
    print("=== VERIFYING REALISM PATCHES (1-5) ===")
    
    # 1. Load data
    p1 = PROJECT_DIR / "artifacts/preds/pred_XRP_USDT_v71.parquet"
    # Fallbak
    if not p1.exists():
        # Try to find any parquet in artifacts/preds
        preds = list((PROJECT_DIR / "artifacts/preds").glob("*.parquet"))
        if preds:
            p1 = preds[0]
        else:
            print("[FAIL] No prediction parquet found in artifacts/preds.")
            return

    print(f"[INFO] Using data: {p1}")
    df = pd.read_parquet(p1)
    
    # Ensure datetime index
    if "ds" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df["ds"] = pd.to_datetime(df["ds"], utc=True)
        df = df.set_index("ds").sort_index()

    # Slice for speed
    df = df.iloc[:10000].copy()
    
    # 2. Configure for Test
    # Force min_hold_bars = 2 to see if it works
    # Force costs
    cfg = {
        "sim": {
            "fee_bps": 5.0,
            "slippage_bps": 2.0,
            "spread_bps": 1.0,
            "exec_lag": 0, # Should be forced to +1 by code anyway if logic is correct? 
                           # Actually code says: entry_i_proposed = i + 1 + int(latency_bars)
                           # So even with lag=0, entry is at i+1.
            "equity_mode": "compound",
        },
        "risk": {"leverage": 1.0, "risk_fraction": 0.1}
    }
    
    thresholds = {
        "min_hold_bars": 2  # [PARCHE 3]
    }
    
    print("[INFO] Running Backtest...")
    met, diag = backtest_from_predictions_v71(df, cfg, thresholds)
    
    trades = diag.get("trades_df")
    if trades is None or trades.empty:
        print("[WARN] No trades generated. Cannot verify.")
        return

    print(f"[INFO] Generated {len(trades)} trades.")
    
    # --- VERIFY PATCH 1 & 3: Timestamps & Min Hold ---
    # Check Duration
    # trades["ts_exit"] - trades["ts_entry"]
    durations = trades["ts_exit"] - trades["ts_entry"]
    zero_dur = (durations <= pd.Timedelta(0)).sum()
    
    print(f"[CHECK] Zero Duration Trades: {zero_dur}")
    if zero_dur > 0:
        print("[FAIL] Found trades with 0 duration!")
    else:
        print("[PASS] No zero duration trades.")
        
    # Check Min Hold
    # i_exit - i_entry >= min_hold_bars
    if "i_entry" in trades.columns and "i_exit" in trades.columns:
        hold_bars = trades["i_exit"] - trades["i_entry"]
        min_h =  hold_bars.min()
        print(f"[CHECK] Min Holding Bars Observed: {min_h} (Target >= 2)")
        if min_h < 2:
            print(f"[FAIL] Found trades with holding < 2 bars! (Min: {min_h})")
            # Dump bad trades
            bad = trades[hold_bars < 2]
            print(bad[["i_entry", "i_exit", "reason_exit"]].head())
        else:
            print("[PASS] Min hold enforcement works.")
    else:
         print("[WARN] i_entry/i_exit columns missing from trades info.")

    # --- VERIFY PATCH 5: Realistic Costs ---
    # Net should be < Raw
    # fee=5, slip=2, spread=1 => total cost ~ 16bps per trade roundtrip approx
    
    # Pick a random trade
    t = trades.iloc[0]
    diff = t["ret_raw"] - t["ret_net"]
    print(f"[CHECK] Trade 0 Cost impact: {diff*10000:.2f} bps (Raw: {t['ret_raw']*10000:.2f} bps, Net: {t['ret_net']*10000:.2f} bps)")
    
    if diff <= 0:
        print("[FAIL] Net return >= Raw return! Costs not applied?")
    else:
        print("[PASS] Costs appear to be deducted.")

    # --- VERIFY PATCH 2: Entry Index ---
    # Entry timestamp should be > Signal timestamp
    # We don't have signal timestamp easily here, but we know i_entry should be > "signal_i"
    # In sim loop: signal at 'i', entry at 'entry_i'
    # We expect entry_i >= i + 1.
    # The recorded 'i_entry' IS the execution index.
    # We can't strictly verify "signal vs entry" without internal logs, 
    # but 'min_hold_bars' verification partially covers index logic integrity.

    # --- VERIFY PATCH 7: Avg Hold Metric ---
    avg_hold = diag.get("avg_hold_bars")
    print(f"[CHECK] Metric avg_hold_bars in diag: {avg_hold}")
    if avg_hold is None:
        print("[FAIL] avg_hold_bars missing from diag!")
    else:
        print("[PASS] avg_hold_bars present.")

    print("\n=== VERIFICATION FINISHED ===")

if __name__ == "__main__":
    run_verification()
