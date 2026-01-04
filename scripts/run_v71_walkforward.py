# scripts/run_v71_walkforward.py
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
import torch
import optuna

from deeplscalp.modeling.train_v71 import train_model_v71, predict_v71
from deeplscalp.modeling.calibration_v71 import fit_temperature_multiclass, apply_temperature_multiclass
from deeplscalp.backtest.sim_v71 import backtest_from_predictions_v71


def _norm_pair(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _align_y_by_pred_index(val_df, pred_df, col: str):
    """
    Alinea y (val_df[col]) exactamente a pred_df.index (datetime utc).
    val_df debe tener columna 'ds' o índice datetime.
    """
    import pandas as pd
    import numpy as np

    v = val_df.copy()
    if "ds" in v.columns:
        v["ds"] = pd.to_datetime(v["ds"], utc=True)
        v = v.set_index("ds", drop=False)
    else:
        v.index = pd.to_datetime(v.index, utc=True)

    idx = pred_df.index
    y = v.loc[idx, col].astype("int64").values
    return np.asarray(y, dtype=np.int64)


def _make_purged_splits(df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seq_len = int(cfg["features"]["seq_len"])
    base_h = int(cfg.get("labels", {}).get("base_horizon", 12))
    max_hold = int(cfg.get("risk", {}).get("max_hold_bars", 16))
    embargo = int(cfg.get("split", {}).get("embargo_bars", max(base_h, max_hold) + 2))

    train_bars = int(cfg.get("split", {}).get("train_bars", 180000))
    val_bars = int(cfg.get("split", {}).get("val_bars", 50000))
    test_bars = int(cfg.get("split", {}).get("test_bars", 80000))

    n = len(df)
    train_end = min(train_bars, n)
    val_end = min(train_end + val_bars, n)
    test_end = min(val_end + test_bars, n)

    train_end_p = max(0, train_end - embargo)
    val_end_p = max(train_end, val_end - embargo)

    train = df.iloc[:train_end_p].copy()
    val_start_ctx = max(0, train_end - (seq_len - 1))
    val = df.iloc[val_start_ctx:val_end_p].copy()
    test_start_ctx = max(0, val_end - (seq_len - 1))
    test = df.iloc[test_start_ctx:test_end].copy()

    return train, val, test


def make_folds(df: pd.DataFrame, cfg: dict):
    w = cfg["walkforward"]
    train_days = int(w["train_days"])
    val_days = int(w["val_days"])
    test_days = int(w["test_days"])
    step_days = int(w["step_days"])
    max_folds = int(cfg["data"]["max_folds"])

    ds_min = df["ds"].min()
    ds_max = df["ds"].max()

    start = ds_min + pd.Timedelta(days=30)
    folds = []
    fid = 0
    while True:
        train_start = start
        train_end = train_start + pd.Timedelta(days=train_days)
        val_end = train_end + pd.Timedelta(days=val_days)
        test_end = val_end + pd.Timedelta(days=test_days)
        if test_end > ds_max:
            break
        folds.append({
            "fold_id": fid,
            "train_start": train_start,
            "train_end": train_end,
            "val_end": val_end,
            "test_end": test_end
        })
        fid += 1
        if fid >= max_folds:
            break
        start = start + pd.Timedelta(days=step_days)

    return folds


def _pick(met: dict, keys: list[str], default=0.0):
    for k in keys:
        if k in met:
            return met[k]
    return default


def objective_factory(cfg, pred_val):
    obj_cfg = cfg["tuning"].get("objective", {})

    pf_cap = float(obj_cfg.get("pf_cap", 3.0))
    mdd_target = float(obj_cfg.get("mdd_target", 0.15))
    mdd_penalty = float(obj_cfg.get("mdd_penalty", 2.0))
    tr_target = int(obj_cfg.get("trades_target", 60))
    tr_penalty = float(obj_cfg.get("trades_penalty", 1.0))

    def obj(trial: optuna.Trial):
        # Hyperparams
        p_side_min = trial.suggest_float("p_side_min", 0.55, 0.80)
        score_q = trial.suggest_float("score_q", 0.80, 0.99)
        q_width_max = trial.suggest_float("q_width_max", 0.02, 0.20)
        ev_buffer = trial.suggest_float("ev_buffer", -0.0003, 0.0010)
        topk = trial.suggest_int("top_k", 1500, 12000)
        atr_min = trial.suggest_float("atr_min", 0.0, 0.0005)
        rv_min = trial.suggest_float("rv_min", 0.0, 0.0002)
        rv_max = trial.suggest_float("rv_max", 0.0003, 0.0030)

        thresholds = {
            "p_side_min": p_side_min,
            "score_q": score_q,
            "q_width_max": q_width_max,
            "ev_abs_min": ev_buffer,
            "top_k": topk,
            "use_topk": True,
            "thr_lookback_bars": 4000,
            "ood_q": 0.90,
            "ev_q": 0.70,
            "p_tp_min": 0.20,
            "p_sl_max": 0.45,
            "ev_buffer_mult": 1.0,
            "q_width_mult": 2.5,
            "cooldown_bars": 1,
            "atr_min": atr_min,
            "rv_min": rv_min,
            "rv_max": rv_max,
        }

        met, _ = backtest_from_predictions_v71(pred_val, cfg, thresholds)

        pf = float(_pick(met, ["pf_x2", "pf_x2_lat1", "PF_strict_x2_lat1", "profit_factor"], 0.0))
        mdd = float(_pick(met, ["mdd_x2", "max_drawdown_x2", "max_drawdown"], 0.0))
        ntr = int(_pick(met, ["ntr_x2", "n_trades_x2", "n_trades"], 0))

        pf_capped = min(pf, pf_cap)
        pen_mdd = mdd_penalty * max(0.0, mdd - mdd_target)
        pen_tr = tr_penalty * max(0, tr_target - ntr) / max(1, tr_target)

        obj_score = pf_capped - pen_mdd - pen_tr

        print(f"[tuner] pf={pf:.3f} mdd={mdd:.3f} ntr={ntr} obj={obj_score:.3f}")
        return float(obj_score)

    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--tf", default=None, help="Override timeframe, e.g. 1m, 5m, 15m")
    ap.add_argument("--max-folds", type=int, default=None, help="Limit folds for quick smoke tests")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)

    # Override timeframe if specified
    if args.tf:
        if isinstance(cfg.get("data", None), dict):
            cfg["data"]["tf"] = args.tf
        else:
            cfg["tf"] = args.tf

    pair = args.pair
    device = cfg.get("device", args.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.set_num_threads(int(cfg.get("runtime", {}).get("torch_num_threads", 6)))

    # build dataset
    subprocess.run(["python", "pipeline.py", "--config", args.config, "build"], check=True)

    out_dir = Path(cfg.get("project", {}).get("out_dir", "artifacts")).expanduser()
    _ensure_dir(out_dir / "preds")
    _ensure_dir(out_dir / "reports")
    _ensure_dir(out_dir / "models_v71")

    ds_path = out_dir / "datasets" / f"train_{_norm_pair(pair)}_5m_v71.parquet"
    df = pd.read_parquet(ds_path)
    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    df = df.sort_values("ds").reset_index(drop=True)
    df = df.set_index("ds")

    feature_cols = cfg.get("features", {}).get("columns")
    if not feature_cols:
        feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in {"y_side", "yc_long", "yc_short", "yL_reg", "yS_reg", "y_regime", "y_event", "sample_weight", "is_event", "ds", "pair", "timeframe", "open_next", "vol_scale", "atr"}]

    print(f"[DATA] rows={len(df)} features={len(feature_cols)}")

    folds = make_folds(df, cfg)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    all_fold_metrics = []

    for fold in folds:
        print(f"\n=== FOLD {fold['fold_id']} ===")

        train_df = df[(df.index >= fold["train_start"]) & (df.index < fold["train_end"])].copy()
        val_df = df[(df.index >= fold["train_end"]) & (df.index < fold["val_end"])].copy()
        test_df = df[(df.index >= fold["val_end"]) & (df.index < fold["test_end"])].copy()

        # Train final model
        model, scaler = train_model_v71(
            train_df.reset_index(),
            val_df.reset_index(),
            feature_cols=feature_cols,
            cfg=cfg,
            device=device,
            fold_id=fold["fold_id"],
            out_dir=out_dir / "models_v71",
        )

        # Predict VAL
        pred_val = predict_v71(model, scaler, val_df.reset_index(), feature_cols, cfg, device)

        # Calibración
        t_grid = np.linspace(0.5, 5.0, 19)

        # SIDE
        P_side = pred_val[["p_flat","p_long","p_short"]].values
        y_side = _align_y_by_pred_index(val_df, pred_val, "y_side")
        T_side, nll_side = fit_temperature_multiclass(P_side, y_side, t_grid)
        pred_val.loc[:, ["p_flat","p_long","p_short"]] = apply_temperature_multiclass(P_side, T_side)
        print(f"[CAL] side T={T_side:.3f} nll={nll_side:.6f}")

        # REGIME
        P_reg = pred_val[["p_reg_range","p_reg_up","p_reg_dn","p_reg_spike"]].values
        y_reg = _align_y_by_pred_index(val_df, pred_val, "y_regime")
        T_reg, nll_reg = fit_temperature_multiclass(P_reg, y_reg, t_grid)
        pred_val.loc[:, ["p_reg_range","p_reg_up","p_reg_dn","p_reg_spike"]] = apply_temperature_multiclass(P_reg, T_reg)
        print(f"[CAL] regime T={T_reg:.3f} nll={nll_reg:.6f}")

        # EVENT
        P_evt = pred_val[["p_evt_none","p_evt_breakout","p_evt_rebound","p_evt_spike"]].values
        y_evt = _align_y_by_pred_index(val_df, pred_val, "y_event")
        T_evt, nll_evt = fit_temperature_multiclass(P_evt, y_evt, t_grid)
        pred_val.loc[:, ["p_evt_none","p_evt_breakout","p_evt_rebound","p_evt_spike"]] = apply_temperature_multiclass(P_evt, T_evt)
        print(f"[CAL] event T={T_evt:.3f} nll={nll_evt:.6f}")

        # Tuning
        obj = objective_factory(cfg, pred_val)
        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=int(cfg["tuning"]["n_trials"]))

        best_thresholds = study.best_params

        # Predict TEST
        pred_test = predict_v71(model, scaler, test_df.reset_index(), feature_cols, cfg, device)

        # Aplicar calibración a test
        pred_test.loc[:, ["p_flat","p_long","p_short"]] = apply_temperature_multiclass(pred_test[["p_flat","p_long","p_short"]].values, T_side)
        pred_test.loc[:, ["p_reg_range","p_reg_up","p_reg_dn","p_reg_spike"]] = apply_temperature_multiclass(pred_test[["p_reg_range","p_reg_up","p_reg_dn","p_reg_spike"]].values, T_reg)
        pred_test.loc[:, ["p_evt_none","p_evt_breakout","p_evt_rebound","p_evt_spike"]] = apply_temperature_multiclass(pred_test[["p_evt_none","p_evt_breakout","p_evt_rebound","p_evt_spike"]].values, T_evt)

        # Backtest
        met, diag = backtest_from_predictions_v71(pred_test, cfg, best_thresholds)

        # Save
        fold_dir = out_dir / "reports" / f"fold_{fold['fold_id']}"
        fold_dir.mkdir(exist_ok=True)
        pred_test.to_parquet(fold_dir / "pred_test.parquet")
        with open(fold_dir / "metrics_test.json", "w") as f:
            json.dump(met, f, indent=2)

        all_fold_metrics.append({"fold": fold["fold_id"], **met})

    # Summary
    sm = pd.DataFrame(all_fold_metrics)
    sm.to_csv(out_dir / "reports" / "walkforward_summary.csv", index=False)


if __name__ == "__main__":
    main()
