
import sys
import os
import pandas as pd
import numpy as np
import torch

# Add root to path
sys.path.append(os.getcwd())

from deeplscalp.backtest.sim_v71 import backtest_from_predictions_v71
from deeplscalp.modeling.train_v71 import train_model_v71
from deeplscalp.metrics import compute_metrics_from_trades
from deeplscalp.gating import apply_gating, GatingConfig

def test_gating_import():
    print("[TEST] Gating import...")
    cfg = GatingConfig(
        p_side_min=0.5,
        score_q=0.9,
        topk_frac=0.1,
        ev_buffer=0.0001,
        q_width_mult=2.0
    )
    print("  GatingConfig ok:", cfg)
    
def test_sim_v71():
    print("[TEST] Sim v71 basic run...")
    # Create dummy prediction df
    N = 100
    dates = pd.date_range("2024-01-01", periods=N, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "ds": dates,
        "open": 100 + np.random.randn(N).cumsum(),
        "high": 100 + np.random.randn(N).cumsum() + 1,
        "low": 100 + np.random.randn(N).cumsum() - 1,
        "close": 100 + np.random.randn(N).cumsum(),
        "p_long": np.random.rand(N),
        "p_short": np.random.rand(N),
        "p_flat": np.random.rand(N),
        "score": np.random.rand(N),
        "atr": np.ones(N) * 1.0,
        "atr_14": np.ones(N) * 1.0,
        "vol_scale": np.ones(N),
    })
    
    # Needs specific columns for v71 sim
    # thresholds
    thresholds = {
        "p_side_min": 0.5,
        "p_tp_min": 0.1,
        "p_sl_max": 0.9,
    }
    cfg = {"objective": {"pf_cap": 10.0}}
    
    # We need to mock other columns used in sim loop (evL, evS etc) which are usually computed in sim
    # Wait, sim computes them from preds. Preds need:
    # "yL_gross" etc? No, those are labels.
    # Sim uses 'pL_tp', 'pL_sl', etc.
    # Let's add them
    df["pL_tp"] = np.random.rand(N)
    df["pL_sl"] = np.random.rand(N)
    df["pS_tp"] = np.random.rand(N)
    df["pS_sl"] = np.random.rand(N)
    df["pL_gross"] = np.random.randn(N) # Not used directly?
    
    # sim_v71 expects 'pL_gross' or 'yL_gross' to compute EV?
    # L600: evL_gross = pred_df["pL_gross"] ...
    # Wait, usually these come from predict_v71 output.
    df["pL_gross"] = 0.01
    df["pS_gross"] = 0.01
    
    # iqr
    df["qL90_gross"] = 0.02
    df["qL10_gross"] = -0.01
    df["qS90_gross"] = 0.02
    df["qS10_gross"] = -0.01
    
    # Regime
    df["p_reg_range"] = 0.5
    df["p_reg_spike"] = 0.0
    df["p_evt_breakout"] = 0.0
    df["p_evt_rebound"] = 0.0
    
    # Run
    try:
        metrics, diag = backtest_from_predictions_v71(df, cfg, thresholds, adaptive_gating=True)
        print("  Sim run successful.")
        print("  Metrics keys:", metrics.keys())
        print("  Metrics:", metrics)
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

def test_dynamic_weights():
    print("[TEST] Dynamic weights in train_v71...")
    # Mock data
    # Just check if function exists and works
    from deeplscalp.modeling.train_v71 import dynamic_class_weights
    y = torch.tensor([0, 0, 1, 1, 1, 2])
    w = dynamic_class_weights(y, n_classes=3)
    print(f"  Weights for counts [2, 3, 1]: {w}")
    # expected: inv prop to counts. 
    # counts=[2,3,1]. inv=[0.5, 0.33, 1.0]. mean ~0.61.
    # w = [0.81, 0.54, 1.63] approx.
    assert w[2] > w[0] > w[1] # class 2 (count 1) should have highest weight
    print("  Assertion logic passed.")

if __name__ == "__main__":
    test_gating_import()
    test_dynamic_weights()
    test_sim_v71()
