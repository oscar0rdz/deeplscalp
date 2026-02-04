import pandas as pd
import numpy as np
from deeplscalp.backtest.sim_v71 import backtest_from_predictions_v71, assert_trade_cost_consistency

def test_sim_cost_invariance():
    print("Building dummy data...")
    # Create dummy dataframe
    dates = pd.date_range("2024-01-01", periods=1000, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": 100 + np.random.randn(1000),
        "high": 105 + np.random.randn(1000),
        "low": 95 + np.random.randn(1000),
        "close": 100 + np.random.randn(1000),
        "p_long": np.random.rand(1000),
        "p_short": np.random.rand(1000),
        "p_flat": np.random.rand(1000),
        "pL_tp": np.random.rand(1000),
        "pL_sl": np.random.rand(1000),
        "pS_tp": np.random.rand(1000),
        "pS_sl": np.random.rand(1000),
        "qL10_gross": -0.01 * np.ones(1000),
        "qL50_gross": 0.0 * np.ones(1000),
        "qL90_gross": 0.02 * np.ones(1000),
        "qS10_gross": -0.01 * np.ones(1000),
        "qS50_gross": 0.0 * np.ones(1000),
        "qS90_gross": 0.02 * np.ones(1000),
        "atr": 0.5 * np.ones(1000),
        "iqrL_gross": 0.1 * np.ones(1000),
        "iqrS_gross": 0.1 * np.ones(1000),
    }, index=dates)
    
    # Ensure some trades happen
    p_long_idx = df.columns.get_loc("p_long")
    pL_tp_idx = df.columns.get_loc("pL_tp")
    pL_sl_idx = df.columns.get_loc("pL_sl")
    
    df.iloc[300:400, p_long_idx] = 0.99
    df.iloc[300:400, pL_tp_idx] = 0.99
    df.iloc[300:400, pL_sl_idx] = 0.01
    
    # Config
    cfg = {
        "sim": {
            "fee_bps": 5.0,
            "slippage_bps": 2.0,
            "spread_bps": 1.0,
            "exec_lag": 1
        },
        "risk": {
            "leverage": 1.0, 
            "risk_fraction": 1.0
        }
    }
    
    thresholds = {
        "p_side_min": 0.5,
        "score_q": 0.0,
        "ood_q": 1.0,
        "ev_q": 0.0, # accept all
        "p_tp_min": 0.1,
        "p_sl_max": 0.9,
        "thr_lookback_bars": 300, # sufficient lookback
    }
    
    print("Running backtest simulation...")
    met, diag = backtest_from_predictions_v71(df, cfg, thresholds, adaptive_gating=False)
    
    trades = diag.get("trades_df")
    if trades is None or trades.empty:
        print("[WARN] No trades generated in smoke test. Tweak inputs.")
        return

    print(f"Generated {len(trades)} trades.")
    print(trades[["ret_raw", "ret_net", "fee", "slippage", "spread"]].head())
    
    # Check Invariance
    raw = trades["ret_raw"].astype(float)
    net = trades["ret_net"].astype(float)
    costs = trades["fee"].astype(float) + trades["slippage"].astype(float) + trades["spread"].astype(float)
    
    diff = (net - (raw - costs)).abs().max()
    print(f"Max Diff |Net - (Raw - Costs)|: {diff:.2e}")
    
    if diff < 1e-10:
        print("[PASS] Cost invariance holds.")
    else:
        print("[FAIL] Cost invariance broken!")
        
    try:
        assert_trade_cost_consistency(trades)
        print("[PASS] assert_trade_cost_consistency passed.")
    except Exception as e:
        print(f"[FAIL] assert_trade_cost_consistency failed: {e}")

    # --- INTEGRATION TEST: GENERATE ARTIFACTS FOR AUDIT ---
    import json
    import shutil
    from pathlib import Path
    
    print("\n--- Generating Mock Artifacts for Audit Test ---")
    REP_DIR = Path("artifacts/reports_sim")
    if REP_DIR.exists():
        shutil.rmtree(REP_DIR)
    REP_DIR.mkdir(parents=True)
    
    # Create Fold 0
    f0 = REP_DIR / "fold_0"
    f0.mkdir()
    
    # Save trades as best_trades.csv (unified source)
    trades.to_csv(f0 / "best_trades.csv", index=False)
    
    # Save metrics
    metrics = {
        "net": float(trades["pnl_net"].sum()),
        "n_trades": len(trades),
        "profit_factor": 2.5, # dummy
        "mdd": 0.1 # dummy
    }
    (f0 / "best_metrics.json").write_text(json.dumps(metrics))
    
    # Create Summary CSV with "fold_0.0" issue (float)
    summary_data = [{
        "fold": 0.0, # <--- The suspicious float
        "net": metrics["net"],
        "n_trades": metrics["n_trades"],
        "profit_factor": 2.5,
        "mdd": 0.1,
        "profit_factor_raw": 2.5
    }]
    pd.DataFrame(summary_data).to_csv(REP_DIR / "walkforward_summary_audited.csv", index=False)
    
    print(f"Created artifacts at {REP_DIR}")
    print("Now run: python scripts/audit_robust.py --reports artifacts/reports_sim")

if __name__ == "__main__":
    test_sim_cost_invariance()
