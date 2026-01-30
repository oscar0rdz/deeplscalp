import pandas as pd
import numpy as np
from deeplscalp.backtest.sim_v71 import backtest_from_predictions_v71

def smoke_test():
    print("Starting smoke test for sim_v71 fix...")
    
    # Create dummy data
    n = 1000
    dates = pd.date_range("2023-01-01", periods=n, freq="5min")
    df = pd.DataFrame({
        "open": np.random.rand(n) + 10,
        "high": np.random.rand(n) + 11,
        "low": np.random.rand(n) + 9,
        "close": np.random.rand(n) + 10,
        "p_long": np.random.rand(n),
        "p_short": np.random.rand(n),
        "p_flat": np.random.rand(n),
        "pL_tp": np.random.rand(n),
        "pL_sl": np.random.rand(n),
        "pS_tp": np.random.rand(n),
        "pS_sl": np.random.rand(n),
        "qL10_gross": -np.random.rand(n) * 0.01,
        "qL50_gross": np.random.rand(n) * 0.01,
        "qL90_gross": np.random.rand(n) * 0.02,
        "qS10_gross": -np.random.rand(n) * 0.01,
        "qS50_gross": np.random.rand(n) * 0.01,
        "qS90_gross": np.random.rand(n) * 0.02,
        "atr": np.random.rand(n) * 0.1,
    }, index=dates)
    
    cfg = {
        "risk": {
            "leverage": 3.0,
            "risk_fraction": 0.25,
            "fee_rate": 0.0004,
            "slippage": 0.0002
        }
    }
    
    thresholds = {
        "p_side_min": 0.1, # force some trades
        "score_q": 0.1,
        "ood_q": 0.9,
        "ev_q": 0.1,
        "use_topk": False,
        "thr_lookback_bars": 300
    }
    
    try:
        metrics, diag = backtest_from_predictions_v71(df, cfg, thresholds)
        print("Backtest successful!")
        print(f"Metrics: {metrics}")
        if "profit_factor" in metrics:
            print(f"Profit Factor: {metrics['profit_factor']}")
        else:
            print("ERROR: profit_factor missing from metrics")
            exit(1)
    except NameError as e:
        print(f"FAILED: NameError detected: {e}")
        exit(1)
    except Exception as e:
        print(f"FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    smoke_test()
