# scripts/run_v71_walkforward.py
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import yaml

PROB_DTYPE = np.float32


def assign_probs(df, cols, arr):
    arr = np.asarray(arr, dtype=PROB_DTYPE)
    df.loc[:, cols] = arr


def _deep_merge(a, b):
    """Deep merge two dictionaries."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return b
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _maybe_generate_kaggle_cfg(path: Path):
    """Auto-generate _kaggle_v71_wf_tune.yaml from v71_gpu_kaggle.yaml if missing."""
    # Solo para el caso esperado
    if path.name != "_kaggle_v71_wf_tune.yaml":
        return

    base_path = Path("configs/v71_gpu_kaggle.yaml")
    if not base_path.exists():
        raise FileNotFoundError(
            f"No existe {path} y tampoco {base_path}. "
            "Genera el config o agrega v71_gpu_kaggle.yaml al repo."
        )

    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    overlay = {
        "tf": "5m",
        "data": {
            "use_prebuilt": True,
            "kaggle_parquet_dir": "/kaggle/input/parquet-v1",
            "max_folds": 24,
        },
        "labels": {"loss_w_regime": 0.0, "loss_w_event": 0.0},
        "walkforward": {
            "train_days": 240,
            "val_days": 30,
            "test_days": 30,
            "step_days": 30,
        },
        "tuning": {
            "n_trials": 35,
            "objective": {
                "pf_cap": 10.0,
                "mdd_target": 0.15,
                "mdd_penalty": 2.0,
                "trades_target": 150,
                "trades_penalty": 1.0,
                "trades_max": 250,
                "overtrade_penalty": 1.0,
            },
        },
    }

    cfg = _deep_merge(base, overlay)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False), encoding="utf-8")
    print(f"[CONFIG] Auto-generated {path} from {base_path}")

from deeplscalp.backtest.sim_v71 import (backtest_from_predictions_v71,
                                         profit_factor_stats)
from deeplscalp.modeling.calibration_v71 import (apply_temperature_multiclass,
                                                 fit_temperature_multiclass)
from deeplscalp.modeling.train_v71 import predict_v71, train_model_v71
from deeplscalp.tuning.objective_v72 import robust_objective_v2


def ensure_ds_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantiza que df tenga columna 'ds' en UTC.
    Soporta:
      - 'ds' como columna
      - 'ds' como índice datetime
      - columnas alternativas: date/datetime/time/timestamp/open_time/close_time
    """
    df = df.copy()

    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], utc=True, errors="coerce")

    elif isinstance(df.index, pd.DatetimeIndex):
        df["ds"] = pd.to_datetime(df.index, utc=True, errors="coerce")

    else:
        candidates = ["date", "datetime", "time", "timestamp", "open_time", "close_time"]
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                s = df[c]
                if pd.api.types.is_datetime64_any_dtype(s):
                    df["ds"] = pd.to_datetime(s, utc=True, errors="coerce")
                elif pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
                    s_nonnull = pd.Series(s).dropna()
                    if len(s_nonnull) == 0:
                        df["ds"] = pd.NaT
                    else:
                        v = float(s_nonnull.iloc[0])
                        unit = "ms" if v > 1e12 else "s"
                        df["ds"] = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
                else:
                    df["ds"] = pd.to_datetime(s, utc=True, errors="coerce")
                break

        if found is None:
            raise KeyError(
                "No se encontró columna temporal. Tu parquet debe incluir 'ds' "
                "o una de: date/datetime/time/timestamp/open_time/close_time, "
                "o guardar el datetime como índice."
            )

    bad = int(df["ds"].isna().sum())
    if bad > 0:
        df = df[df["ds"].notna()].copy()

    df = df.sort_values("ds").reset_index(drop=True)
    return df


def _norm_pair(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_cfg(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        _maybe_generate_kaggle_cfg(p)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {p}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config inválido o vacío: {path}")
    return cfg


def _align_y_by_pred_index(val_df: pd.DataFrame, pred_df: pd.DataFrame, col: str) -> np.ndarray:
    """
    Alinea val_df[col] exactamente a pred_df.index (datetime utc).
    val_df debe tener columna 'ds' o índice datetime.
    """
    v = val_df.copy()
    if "ds" in v.columns:
        v["ds"] = pd.to_datetime(v["ds"], utc=True)
        v = v.set_index("ds", drop=False)
    else:
        v.index = pd.to_datetime(v.index, utc=True)

    idx = pred_df.index
    y = v.loc[idx, col].astype("int64").values
    return np.asarray(y, dtype=np.int64)


def make_folds(df: pd.DataFrame, cfg: dict) -> List[dict]:
    # Hardening: defaults razonables si faltan claves
    w = cfg.get("walkforward", {})
    train_days = int(w.get("train_days", 240))
    val_days = int(w.get("val_days", 30))
    test_days = int(w.get("test_days", 30))
    step_days = int(w.get("step_days", 30))
    purge_days = int(w.get("purge_days", 10))  # Anti-leakage gap

    max_folds = int(cfg.get("data", {}).get("max_folds", 24))

    # BUGFIX: usar índice si ya está en datetime; si no, usar columna ds
    if isinstance(df.index, pd.DatetimeIndex):
        ds_min = pd.to_datetime(df.index.min(), utc=True)
        ds_max = pd.to_datetime(df.index.max(), utc=True)
    else:
        ds_min = pd.to_datetime(df["ds"].min(), utc=True)
        ds_max = pd.to_datetime(df["ds"].max(), utc=True)

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

        folds.append(
            {
                "fold_id": fid,
                "train_start": train_start,
                "train_end": train_end,
                "val_end": val_end,
                "test_end": test_end,
            }
        )
        fid += 1
        if fid >= max_folds:
            break

        start = start + pd.Timedelta(days=step_days)

    return folds


def _pick(met: dict, keys: List[str], default=0.0):
    for k in keys:
        if k in met:
            return met[k]
    return default


def objective_factory(cfg: dict, pred_val: pd.DataFrame):
    obj_cfg = cfg.get("tuning", {}).get("objective", {})
    pf_cap = float(obj_cfg.get("pf_cap", 3.0))
    mdd_target = float(obj_cfg.get("mdd_target", 0.15))
    mdd_penalty = float(obj_cfg.get("mdd_penalty", 2.0))
    tr_target = int(obj_cfg.get("trades_target", 60))
    tr_penalty = float(obj_cfg.get("trades_penalty", 1.0))
    tr_max = int(obj_cfg.get("trades_max", 250))
    overtrade_penalty = float(obj_cfg.get("overtrade_penalty", 1.0))

    def obj(trial: optuna.Trial):
        # Hyperparams (rangos hardcodeados como en tu versión original)
        p_side_min = trial.suggest_float("p_side_min", 0.55, 0.80)
        score_q = trial.suggest_float("score_q", 0.80, 0.99)
        q_width_max = trial.suggest_float("q_width_max", 0.02, 0.20)
        ev_buffer = trial.suggest_float("ev_buffer", -0.0003, 0.0010)
        topk_frac = trial.suggest_float("topk_frac", 0.001, 0.02, log=True)  # 0.1% a 2%
        top_k = max(50, int(topk_frac * len(pred_val)))
        atr_min = trial.suggest_float("atr_min", 0.0, 0.0005)
        rv_min = trial.suggest_float("rv_min", 0.0, 0.0002)
        rv_max = trial.suggest_float("rv_max", 0.0003, 0.0030)

        thresholds = {
            "p_side_min": p_side_min,
            "score_q": score_q,
            "q_width_max": q_width_max,
            "ev_abs_min": ev_buffer,
            "top_k": top_k,
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

        # Usar objetivo robusto que previene autoengaño
        s = profit_factor_stats(np.array([1.0, pf, -1.0]), pf_cap)  # Dummy array para calcular stats
        zero_loss = s.zero_loss

        obj_score = robust_objective_v2(
            pf=pf,
            net_return=0.0,  # TODO: compute actual net return
            mdd=mdd,
            ntr=ntr,
            turnover=0.0,  # TODO: compute turnover
            avg_hold_bars=0.0,  # TODO: compute avg hold bars
            pf_cap=50.0,  # higher cap
            ntr_min=int(obj_cfg.get("ntr_min", 150)),
            lam_mdd=mdd_penalty,
            beta_ntr=tr_penalty,
            gamma_hold=0.2,
        )

        print(f"[tuner] pf={pf:.3f} mdd={mdd:.3f} ntr={ntr} zero_loss={zero_loss} obj={obj_score:.3f}")
        return float(obj_score)

    return obj


def _resolve_tf(args_tf: str | None, cfg: dict) -> str:
    if args_tf:
        return str(args_tf)
    if "tf" in cfg and cfg["tf"]:
        return str(cfg["tf"])
    if "data" in cfg and isinstance(cfg["data"], dict) and cfg["data"].get("tf"):
        return str(cfg["data"]["tf"])
    if "timeframes" in cfg and isinstance(cfg["timeframes"], dict) and cfg["timeframes"].get("exec_tf"):
        return str(cfg["timeframes"]["exec_tf"])
    return "5m"


def _resolve_prebuilt_dir(cfg: dict) -> str | None:
    # Acepta ambas convenciones: dataset.prebuilt_dir (vieja) y data.kaggle_parquet_dir (nueva)
    d1 = cfg.get("dataset", {}).get("prebuilt_dir")
    d2 = cfg.get("data", {}).get("kaggle_parquet_dir")
    return d1 or d2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--tf", default=None, help="Override timeframe, e.g. 1m, 5m, 15m")
    ap.add_argument("--max-folds", type=int, default=None, help="Limit folds for quick smoke tests")
    ap.add_argument(
        "--no-build",
        action="store_true",
        help="Si falta el parquet prebuilt, NO intentes pipeline build; falla con error claro.",
    )
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    cfg.setdefault("data", {})
    cfg.setdefault("dataset", {})
    cfg.setdefault("tuning", {})

    # CLI domina
    if args.max_folds is not None:
        cfg["data"]["max_folds"] = int(args.max_folds)

    tf = _resolve_tf(args.tf, cfg)
    cfg["tf"] = tf
    cfg["data"]["tf"] = tf

    pair = args.pair

    device = cfg.get("device", args.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.set_num_threads(int(cfg.get("runtime", {}).get("torch_num_threads", 6)))

    out_dir = Path(cfg.get("project", {}).get("out_dir", "artifacts")).expanduser()
    _ensure_dir(out_dir / "preds")
    _ensure_dir(out_dir / "reports")
    _ensure_dir(out_dir / "models_v71")
    _ensure_dir(out_dir / "datasets")

    # Resolver prebuilt
    prebuilt_dir = _resolve_prebuilt_dir(cfg)
    use_prebuilt = bool(cfg.get("data", {}).get("use_prebuilt", False) or prebuilt_dir)

    ds_path: Path
    if prebuilt_dir:
        prebuilt_path = Path(prebuilt_dir) / f"train_{_norm_pair(pair)}_{tf}_v71.parquet"
        if prebuilt_path.exists():
            ds_path = prebuilt_path
            print(f"[DATA] Using prebuilt dataset from {ds_path}")
        else:
            msg = (
                f"[DATA] Prebuilt dir specified but file not found:\n"
                f"       expected: {prebuilt_path}\n"
                f"       pair={pair} tf={tf}\n"
            )
            if args.no_build or use_prebuilt:
                raise FileNotFoundError(msg)
            print(msg + "       building via pipeline...")
            subprocess.run(["python", "pipeline.py", "--config", args.config, "build"], check=True)
            ds_path = out_dir / "datasets" / f"train_{_norm_pair(pair)}_{tf}_v71.parquet"
    else:
        if use_prebuilt and args.no_build:
            raise ValueError("[DATA] use_prebuilt activo pero no hay directorio de prebuilt configurado.")
        # comportamiento legacy
        subprocess.run(["python", "pipeline.py", "--config", args.config, "build"], check=True)
        ds_path = out_dir / "datasets" / f"train_{_norm_pair(pair)}_{tf}_v71.parquet"

    df = pd.read_parquet(ds_path)
    df = ensure_ds_column(df)

    # BUGFIX: conservar ds aunque se use como índice
    df = df.sort_values("ds").reset_index(drop=True)
    df = df.set_index("ds", drop=False)

    feature_cols = cfg.get("features", {}).get("columns")
    if not feature_cols:
        drop_cols = {
            "y_side",
            "yc_long",
            "yc_short",
            "yL_reg",
            "yS_reg",
            "y_regime",
            "y_event",
            "sample_weight",
            "is_event",
            "ds",
            "pair",
            "timeframe",
            "open_next",
            "vol_scale",
            "atr",
        }
        feature_cols = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in drop_cols
        ]

        # Leakage guard: remove potentially leaky features
        LEAKY_TOKENS = ("next", "lead", "future", "fwd", "target", "label")
        leaky = [c for c in feature_cols if any(t in c.lower() for t in LEAKY_TOKENS)]
        if leaky:
            print(f"[WARN] Removing {len(leaky)} potentially-leaky feature columns. Examples: {leaky[:15]}")
            feature_cols = [c for c in feature_cols if c not in leaky]

    print(f"[DATA] rows={len(df)} features={len(feature_cols)} tf={tf}")

    folds = make_folds(df, cfg)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    if "n_trials" not in cfg.get("tuning", {}):
        # Hardening: default si te falta en config
        cfg["tuning"]["n_trials"] = 35

    all_fold_metrics = []

    for fold in folds:
        print(f"\n=== FOLD {fold['fold_id']} ===")

        train_df = df[(df.index >= fold["train_start"]) & (df.index < fold["train_end"])].copy()
        val_df = df[(df.index >= fold["train_end"]) & (df.index < fold["val_end"])].copy()
        test_df = df[(df.index >= fold["val_end"]) & (df.index < fold["test_end"])].copy()

        model, scaler = train_model_v71(
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            feature_cols=feature_cols,
            cfg=cfg,
            device=device,
            fold_id=fold["fold_id"],
            out_dir=out_dir / "models_v71",
        )

        pred_val = predict_v71(
            model, scaler, val_df.reset_index(drop=True), feature_cols, cfg, device
        )

        # Calibración
        t_grid = np.linspace(0.5, 5.0, 19)

        # SIDE
        P_side = pred_val[["p_flat", "p_long", "p_short"]].values
        y_side = _align_y_by_pred_index(val_df, pred_val, "y_side")
        T_side, nll_side = fit_temperature_multiclass(P_side, y_side, t_grid)
        assign_probs(pred_val, ["p_flat", "p_long", "p_short"], apply_temperature_multiclass(P_side, T_side))
        print(f"[CAL] side T={T_side:.3f} nll={nll_side:.6f}")

        # REGIME (solo si existe y_regime)
        T_reg = 1.0
        if "y_regime" in val_df.columns:
            P_reg = pred_val[["p_reg_range", "p_reg_up", "p_reg_dn", "p_reg_spike"]].values
            y_reg = _align_y_by_pred_index(val_df, pred_val, "y_regime")
            T_reg, nll_reg = fit_temperature_multiclass(P_reg, y_reg, t_grid)
            assign_probs(pred_val, ["p_reg_range", "p_reg_up", "p_reg_dn", "p_reg_spike"], apply_temperature_multiclass(
                P_reg, T_reg
            ))
            print(f"[CAL] regime T={T_reg:.3f} nll={nll_reg:.6f}")
        else:
            print("[CAL] regime: y_regime no existe; se omite calibración (T=1.0).")

        # EVENT (solo si existe y_event)
        T_evt = 1.0
        if "y_event" in val_df.columns:
            P_evt = pred_val[["p_evt_none", "p_evt_breakout", "p_evt_rebound", "p_evt_spike"]].values
            y_evt = _align_y_by_pred_index(val_df, pred_val, "y_event")
            T_evt, nll_evt = fit_temperature_multiclass(P_evt, y_evt, t_grid)
            assign_probs(pred_val, ["p_evt_none", "p_evt_breakout", "p_evt_rebound", "p_evt_spike"], apply_temperature_multiclass(
                P_evt, T_evt
            ))
            print(f"[CAL] event T={T_evt:.3f} nll={nll_evt:.6f}")
        else:
            print("[CAL] event: y_event no existe; se omite calibración (T=1.0).")

        # Tuning
        obj = objective_factory(cfg, pred_val)
        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=int(cfg["tuning"]["n_trials"]))
        best_thresholds = study.best_params

        # Predict TEST
        pred_test = predict_v71(
            model, scaler, test_df.reset_index(drop=True), feature_cols, cfg, device
        )

        # Aplicar calibración a test
        assign_probs(pred_test, ["p_flat", "p_long", "p_short"], apply_temperature_multiclass(
            pred_test[["p_flat", "p_long", "p_short"]].values, T_side
        ))
        if T_reg != 1.0:
            assign_probs(pred_test, ["p_reg_range", "p_reg_up", "p_reg_dn", "p_reg_spike"], apply_temperature_multiclass(
                pred_test[["p_reg_range", "p_reg_up", "p_reg_dn", "p_reg_spike"]].values, T_reg
            ))
        if T_evt != 1.0:
            assign_probs(pred_test, ["p_evt_none", "p_evt_breakout", "p_evt_rebound", "p_evt_spike"], apply_temperature_multiclass(
                pred_test[["p_evt_none", "p_evt_breakout", "p_evt_rebound", "p_evt_spike"]].values, T_evt
            ))

        met, diag = backtest_from_predictions_v71(pred_test, cfg, best_thresholds)

        fold_dir = out_dir / "reports" / f"fold_{fold['fold_id']}"
        _ensure_dir(fold_dir)
        pred_test.to_parquet(fold_dir / "pred_test.parquet")

        with open(fold_dir / "metrics_test.json", "w", encoding="utf-8") as f:
            json.dump(met, f, indent=2)

        # Guardar trades auditables
        if "trade_ret_raw" in diag:
            trades = pd.DataFrame({
                "ret_raw": diag["trade_ret_raw"],
                "ret_net": diag.get("trade_ret_net", diag["trade_ret_raw"]),
                "fee_per_trade": diag.get("fee_per_trade", 0.0),
                "spread_per_trade": diag.get("spread_per_trade", 0.0),
                "slip_per_trade": diag.get("slip_per_trade", 0.0),
            })
            trades.to_csv(fold_dir / "trades.csv", index=False)

        all_fold_metrics.append({"fold": fold["fold_id"], **met})

    sm = pd.DataFrame(all_fold_metrics)
    sm.to_csv(out_dir / "reports" / "walkforward_summary.csv", index=False)
    print(f"[OK] wrote {out_dir / 'reports' / 'walkforward_summary.csv'}")


if __name__ == "__main__":
    main()
