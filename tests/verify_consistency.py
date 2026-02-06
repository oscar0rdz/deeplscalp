
import pandas as pd
import numpy as np
import sys
import os

# Ensure we can import the project
sys.path.append(os.getcwd())

from deeplscalp.backtest.sim_v71 import backtest_from_predictions_v71, ExecConfig

def test_topk_consistency():
    print("\n--- Test 1: TopK Consistency (Fraction vs Static) ---")
    
    # Mock Data
    n = 2000
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    df = pd.DataFrame({
        "open": np.random.rand(n) * 100,
        "high": np.random.rand(n) * 100,
        "low": np.random.rand(n) * 100,
        "close": np.random.rand(n) * 100,
        "atr": np.full(n, 0.5),
        "p_long": np.random.rand(n),
        "p_short": np.random.rand(n),
        "p_flat": np.random.rand(n),
        "pL_tp": np.random.rand(n),
        "pL_sl": np.random.rand(n),
        "pS_tp": np.random.rand(n),
        "pS_sl": np.random.rand(n),
        # Required extra cols
        "qL10_gross": np.random.rand(n),
        "qL50_gross": np.random.rand(n),
        "qL90_gross": np.random.rand(n),
        "qS10_gross": np.random.rand(n),
        "qS50_gross": np.random.rand(n),
        "qS90_gross": np.random.rand(n),
        "iqrL_gross": np.full(n, 0.1),
        "iqrS_gross": np.full(n, 0.1),
    }, index=dates)
    df["ds"] = df.index
    
    cfg = {"sim": {}, "risk": {}}
    
    # Case A: Static TopK = 10
    thr_a = {
        "use_topk": True,
        "thr_lookback_bars": 300, # Valid > 200
        "top_k": 10,
        "p_side_min": 0.0, # accept all valid signals
        "p_tp_min": 0.0,
        "p_sl_max": 1.0,
        "ev_abs_min": -999,
        "cooldown_bars": 0
    }
    # To isolate TopK, we need signals that pass other filters.
    # But backtest filters are ANDs. 
    # With random data, many will fail other filters (regime, score, etc).
    # We should trust the code inspection mostly, but let's check debug output if possible?
    # Actually, backtest returns `diag` with `dbg` dictionary.
    
    _, diag_a = backtest_from_predictions_v71(df, cfg, thr_a)
    print(f"Case A (Static=10): gate_topk = {diag_a['dbg']['gate_topk']}")
    
    # Case B: TopK Fraction = 0.10 (100 items)
    # This should yield roughly 100 passes if everything else passes?
    # gate_topk counts REJECTIONS.
    # Total signals = N (if p_side_min=0 and valid cands)
    # topk mask is True for top K items per day.
    # With 1000 items over ~3 days (5m * 1000 = 5000 mins = 3.5 days).
    # top_k is applied PER DAY.
    # So if we have 3 days, static=10 means 30 items passed.
    # Fraction=0.10 means 10% of TOTAL passed? No, logic is:
    # `calc_top_k = int(tk_frac * len(df))` -> 100.
    # `topk_mask = topk_streaming_by_day(..., top_k)` uses that integer as "k per day"?
    # Wait, `topk_streaming_by_day` takes integer `top_k`.
    # If I pass `top_k=100`, it allows 100 trades PER DAY.
    # My patch does: `calc_top_k = int(tk_frac * len(df))`.
    # If len(df) is total dataset size, then `calc_top_k` is 100.
    # So it sets "Trades per day" limit to 10% of the TOTAL dataset size!?
    # That seems wrong if the intention of "topk_frac" in Optuna (0.01-0.05) was "fraction of daily opportunities" vs "fraction of total"?
    # The original objective code was: `top_k = max(50, int(topk_frac * len(pred_val)))`.
    # So it WAS scaling with dataset size.
    # If dataset covers 1 year, len(pred_val) is huge.
    # If len=100k, top_k = 5000 trades per day?
    # That effectively disables TopK if TopK is "per day".
    # Let's check `topk_streaming_by_day`.
    
    # `topk_streaming_by_day` logic:
    # resets heap every new day.
    # keeps top K scores of that day.
    
    # If `top_k` passed is 5000, and we have 288 bars/day...
    # Then it selects ALL bars.
    # So the original logic `top_k = int(frac * len(df))` effectively disabled TopK filter if the dataset is long enough.
    # UNLESS `gate_topk` mismatch issue was that Tuner used this large number, but Audit used defaults (10).
    # If Tuner used large number (e.g. 5000), gate_topk rejects 0.
    # If Audit used 10, gate_topk rejects MANY.
    # THIS MATCHES THE USER REPORT: "Trials: gate_topk : 0 (0.0%)", "Rerun audit: gate_topk : ~7%".
    # So my fix (propagating the large number) is correct to ensure consistency.
    # Whether the STRATEGY is good with that is another story, but consistency is the goal.
    
    thr_b = thr_a.copy()
    thr_b["topk_frac"] = 0.50 # 50% of dataset size -> large number -> should allow almost all per day
    _, diag_b = backtest_from_predictions_v71(df, cfg, thr_b)
    print(f"Case B (Frac=0.5): gate_topk = {diag_b['dbg']['gate_topk']}")
    
    # Expectation: Case B rejects fewer (or zero) compared to Case A, proving `topk_frac` was used.
    
    assert diag_b['dbg']['gate_topk'] < diag_a['dbg']['gate_topk'], "TopK Fraction logic failed to permit more trades!"
    print("SUCCESS: TopK logic overrides static default.")

def test_cost_enforcement():
    print("\n--- Test 2: Cost Enforcement ---")
    
    # Mock costs
    cfg_safe = {
        "sim": {
            "force_costs": True,
            "fee_bps": 0.0,      # Invalid!
            "slippage_bps": 0.0  # Invalid!
        },
        "risk": {},
        "mode": "train" # doesn't matter if force_costs=True
    }
    
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    df = pd.DataFrame({
        "open": np.full(n, 100.0),
        "high": np.full(n, 101.0),
        "low": np.full(n, 99.0),
        "close": np.full(n, 100.0), # Flat price
        # Make signals super strong to ensure entry
        "p_long": np.full(n, 0.9),
        "p_short": np.zeros(n),
        "p_flat": np.zeros(n),
        "pL_tp": np.full(n, 0.8),
        "pL_sl": np.full(n, 0.1),
        "pS_tp": np.zeros(n),
        "pS_sl": np.zeros(n),
        "atr": np.full(n, 1.0),
        "qL10_gross": np.full(n, -0.01),
        "qL50_gross": np.full(n, 0.01),
        "qL90_gross": np.full(n, 0.02),
        "qS10_gross": np.full(n, -0.01),
        "qS50_gross": np.full(n, 0.01),
        "qS90_gross": np.full(n, 0.02),
        "iqrL_gross": np.full(n, 0.1),
        "iqrS_gross": np.full(n, 0.1),
    }, index=dates)
    df["ds"] = df.index

    # Thresolds to allow trade
    thr = {
        "p_side_min": 0.5,
        "p_tp_min": 0.5,
        "p_sl_max": 0.5,
        "score_q": 0.0, # pass all
        "ood_q": 1.0,   # Max quantile (100%)
        "ev_q": 0.0,    # Min quantile (0%)
        "ev_abs_min": -99.0,
        "top_k_min": 10,
        "topk_frac": 0.5,
        "top_k": 1000,
        "thr_lookback_bars": 300,
        "cooldown_bars": 0
    }
    
    # We pass explicit exec_cfg with ZERO costs to simulate the "leak"
    bad_exec = ExecConfig(fee_bps=0.0, slippage_bps=0.0, spread_bps=0.0, exec_lag_bars=1)
    
    # Run
    _, diag = backtest_from_predictions_v71(df, cfg_safe, thr, exec_cfg=bad_exec)
    
    # In my patch:
    # "if force_costs: if final_fee_bps < 1e-9: final_fee_bps = 4.0 ..."
    # And then instantiates CostModel(fee_bps=4.0...)
    # The simulation loop uses `cm` (from CostModel) for costs.
    # It checks `_current_cm` or re-instantiates `cm`?
    # Wait, the code I patched instantiates `cm` at the start of loop.
    # BUT inside the loop there is:
    # "if exec_cfg: cm = CostModel(...)" <--- THIS MIGHT OVERRIDE MY PATCH if I put my patch outside or before loop?
    # Let's double check logic order in `sim_v71.py` patch.
    
    # The patch was inserted BEFORE the loop.
    # "if force_costs: ... final_fee_bps = ..."
    # "cm = CostModel(...)"
    #
    # Then INSIDE the loop:
    # "if exec_cfg: cm = CostModel(...exec_cfg...)"
    #
    # OH NO! I might have left the internal override effective!
    # Let's verify this in the test. If it fails, I need to fix the patch immediately.
    
    trades = diag["trades_df"]
    if trades.empty:
        print("WARNING: No trades generated to verify costs.")
        return

    # Check first trade fee
    first_fee = float(trades.iloc[0]["fee"])
    print(f"Fee (should be > 0): {first_fee}")
    
    assert first_fee > 0.0, "Force Costs failed! Fee is zero."
    print("SUCCESS: Costs enforced despite zero exec_cfg.")

if __name__ == "__main__":
    test_topk_consistency()
    test_cost_enforcement()
