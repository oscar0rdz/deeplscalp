import pandas as pd
from pathlib import Path
from deeplscalp.utils.config import load_yaml
from deeplscalp.data.dataset_builder import build_dataset
from deeplscalp.modeling.train import train_quantile_model, predict_quantiles
from deeplscalp.optimize.optuna_search import _gate_audit
from deeplscalp.backtest.sim import backtest_from_predictions

cfg = load_yaml("configs/full/full_pipeline_event_v2_debug.yaml")

df, feature_cols = build_dataset(cfg)
print(f"Dataset: {len(df)} rows")

# Quick folds
from evaluation.run_full_pipeline_event_v2 import make_folds
folds = make_folds(df, cfg)
fold = folds[0]

# Quick train
model, scaler = train_quantile_model(
    train_df=fold.train.iloc[:1000],  # very quick
    val_df=fold.val.iloc[:500],
    feature_cols=feature_cols,
    cfg=cfg,
    device="cpu",
    fold_id=0,
    out_dir=Path("."),
)

# Predict on val
val_pred = predict_quantiles(model, scaler, fold.val.iloc[:500], feature_cols, cfg, "cpu")

# Debug thresholds
thresholds = {
    "score_min": 0.0,
    "ood_thr": float(val_pred["iqr"].max()),
    "p_tp_min": 0.0,
    "p_sl_max": 1.0,
    "ev_min": -1.0,
    "top_k": 0,
}

audit = _gate_audit(val_pred, thresholds)
print(f"Gate audit: {audit}")

# Backtest
metrics, diag = backtest_from_predictions(val_pred, cfg, thresholds)
print(f"Backtest: n_trades={metrics.get('n_trades', 0)}, pf={metrics.get('profit_factor', 0):.3f}")
