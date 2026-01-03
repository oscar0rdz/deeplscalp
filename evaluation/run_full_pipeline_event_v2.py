import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from deeplscalp.utils.config import load_yaml, cfg_hash_md5, ensure_dir
from deeplscalp.utils.seed import set_global_seed
from deeplscalp.utils.device import pick_device, set_torch_threads

from deeplscalp.data.dataset_builder import build_dataset
from deeplscalp.data.labels_v6 import build_labels_v6
from deeplscalp.modeling.train import train_model_v6, predict_v6
from training.itransformer_full import ITransformerV6
from deeplscalp.optimize.optuna_search import optuna_find_thresholds
from deeplscalp.backtest.sim import backtest_from_predictions
from deeplscalp.reports.summary import write_summary_csv


def _ensure_dict(d: dict, k: str) -> dict:
    """Asegura que d[k] exista y sea dict; si no, lo crea."""
    if k not in d or not isinstance(d[k], dict):
        d[k] = {}
    return d[k]


def _get_fail_fast_cfg(cfg: dict) -> dict:
    """
    Soporta configs viejos y nuevos:
      - cfg["fail_fast"]
      - cfg["eval"]["fail_fast"]
    Retorna un dict único y consistente.
    """
    if isinstance(cfg.get("fail_fast"), dict):
        return cfg["fail_fast"]
    ev = cfg.get("eval")
    if isinstance(ev, dict) and isinstance(ev.get("fail_fast"), dict):
        return ev["fail_fast"]
    # Si no existe en ningún lado, lo crea en raíz como fuente de verdad
    cfg["fail_fast"] = {}
    return cfg["fail_fast"]


@dataclass
class FoldSplit:
    fold: int
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _attach_vol_scale(pred_df, split_df, cfg, tag: str):
    if not bool(cfg["labels"].get("vol_scaled", False)):
        return pred_df

    if "vol_scale" in pred_df.columns:
        return pred_df

    if "vol_scale" not in split_df.columns:
        raise ValueError(f"[{tag}] split_df no trae vol_scale, pero labels.vol_scaled=True. Revisa dataset_builder.")

    # join por índice
    pred_df = pred_df.join(split_df[["vol_scale"]], how="left")

    if pred_df["vol_scale"].isna().any():
        n = int(pred_df["vol_scale"].isna().sum())
        raise ValueError(f"[{tag}] vol_scale tiene {n} NaN tras join. Índices no alinean.")

    return pred_df


def _quick_slice(df, n_bars, seq_len):
    if df is None:
        return df
    n_bars = int(n_bars)
    if len(df) <= n_bars:
        return df
    # mantener contigüidad temporal + suficiente historia para seq
    return df.tail(n_bars + (seq_len - 1))


def make_folds(df: pd.DataFrame, cfg: dict) -> list[FoldSplit]:
    """
    Walk-forward splits:
    - train grows
    - val fixed window
    - test fixed window
    """
    split_cfg = cfg["data"]["split"]
    n_folds = int(split_cfg["n_folds"])
    train_min_days = int(split_cfg["train_min_days"])
    val_days = int(split_cfg["val_days"])
    test_days = int(split_cfg["test_days"])

    dt = df.index
    start = dt.min()
    end = dt.max()

    # anchor: first fold starts after train_min_days
    first_train_end = start + pd.Timedelta(days=train_min_days)

    folds: list[FoldSplit] = []
    for k in range(n_folds):
        train_end = first_train_end + pd.Timedelta(days=30 * k)  # step 30 días por fold (puedes ajustar)
        val_end = train_end + pd.Timedelta(days=val_days)
        test_end = val_end + pd.Timedelta(days=test_days)

        if test_end > end:
            break

        train = df.loc[(dt >= start) & (dt < train_end)].copy()
        val = df.loc[(dt >= train_end) & (dt < val_end)].copy()
        test = df.loc[(dt >= val_end) & (dt < test_end)].copy()

        folds.append(FoldSplit(fold=k, train=train, val=val, test=test))

    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--folds", type=str, default="", help="Ej: '0' o '0,1,2'. Vacío = todos.")
    parser.add_argument("--max_folds", type=int, default=0, help="0 = sin límite. Si >0, corre solo los primeros N folds.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Desactiva la detención temprana por métricas mínimas (útil en --quick).",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # --- robust defaults: evita KeyError y asegura estructura ---
    cfg.setdefault("eval", {})
    if not isinstance(cfg["eval"], dict):
        cfg["eval"] = {}

    cfg["eval"].setdefault("fail_fast", {})
    if not isinstance(cfg["eval"]["fail_fast"], dict):
        cfg["eval"]["fail_fast"] = {}

    # defaults fail_fast
    cfg["eval"]["fail_fast"].setdefault("enabled", True)
    cfg["eval"]["fail_fast"].setdefault("min_trades_val", 30)   # modo normal
    cfg["eval"]["fail_fast"].setdefault("min_pf_val", 1.05)     # mínimo razonable
    cfg["eval"]["fail_fast"].setdefault("max_mdd_val", 0.12)    # 12% en VAL como techo

    # risk defaults (para que sim.py nunca reviente)
    cfg.setdefault("risk", {})
    if not isinstance(cfg["risk"], dict):
        cfg["risk"] = {}

    # quick defaults (para recortar rápido)
    cfg.setdefault("quick", {})
    if not isinstance(cfg["quick"], dict):
        cfg["quick"] = {}
    cfg["quick"].setdefault("train_bars", 20000)
    cfg["quick"].setdefault("val_bars", 4000)
    cfg["quick"].setdefault("test_bars", 4000)
    cfg["quick"].setdefault("optuna_trials", 25)
    cfg["quick"].setdefault("min_trades_val", 5)  # en quick no exijas 30

    # override desde CLI
    if getattr(args, "no_fail_fast", False):
        cfg["eval"]["fail_fast"]["enabled"] = False

    if args.quick:
        # Quick = diagnóstico rápido pero estadísticamente útil
        cfg["train"]["epochs"] = int(cfg["train"].get("epochs_quick", 3))
        cfg["optuna"]["n_trials"] = int(cfg["quick"]["optuna_trials"])
        # no matar por pocos trades en quick
        cfg["eval"]["fail_fast"]["enabled"] = False

    run_dir = Path(cfg["run"]["output_dir"])
    ensure_dir(run_dir)

    # hash config -> para "resume safe"
    cfg_hash = cfg_hash_md5(cfg)
    (run_dir / "cfg_hash.txt").write_text(cfg_hash, encoding="utf-8")

    # reproducibilidad
    set_global_seed(int(cfg["run"]["seed"]))

    # runtime
    set_torch_threads(int(cfg["runtime"]["torch_threads"]))
    device = pick_device(cfg["runtime"]["device"])
    print(f"[device] {device}")

    # dataset + features
    df, feature_cols = build_dataset(cfg)
    print(f"[dataset] rows={len(df):,} features={len(feature_cols)}")

    # labels V6
    df = build_labels_v6(df, cfg)
    print(f"[labels_v6] applied, y_side distribution: {df['y_side'].value_counts().to_dict()}")

    folds = make_folds(df, cfg)
    if not folds:
        raise RuntimeError("No se pudieron crear folds. Revisa split windows vs rango de fechas.")

    # Filtrar folds
    folds_arg = (args.folds or "").strip()
    selected = None
    if folds_arg:
        selected = set(int(x.strip()) for x in folds_arg.split(",") if x.strip())

    max_folds = int(args.max_folds or 0)

    fold_ids = list(range(len(folds)))
    if selected is not None:
        fold_ids = [f for f in fold_ids if f in selected]
    if max_folds > 0:
        fold_ids = fold_ids[:max_folds]

    all_fold_rows = []

    for fold_id in fold_ids:
        fold = folds[fold_id]

        dbg = cfg.get("debug", {})
        if dbg:
            t = int(dbg.get("train_tail_bars", 0))
            v = int(dbg.get("val_tail_bars", 0))
            s = int(dbg.get("test_tail_bars", 0))
            if t > 0: fold.train = fold.train.iloc[-t:]
            if v > 0: fold.val = fold.val.iloc[-v:]
            if s > 0: fold.test = fold.test.iloc[-s:]

        print(f"\n=== FOLD {fold.fold} ===")
        print(f"train: {fold.train.index.min()} -> {fold.train.index.max()} | n={len(fold.train)}")
        print(f"val  : {fold.val.index.min()} -> {fold.val.index.max()} | n={len(fold.val)}")
        print(f"test : {fold.test.index.min()} -> {fold.test.index.max()} | n={len(fold.test)}")

        if args.quick:
            seq_len = int(cfg["features"]["seq_len"])
            fold.train = _quick_slice(fold.train, cfg["quick"]["train_bars"], seq_len)
            fold.val   = _quick_slice(fold.val,   cfg["quick"]["val_bars"],   seq_len)
            fold.test  = _quick_slice(fold.test,  cfg["quick"]["test_bars"],  seq_len)

            # quick: menos trials y umbral mínimo de trades más bajo
            cfg["optuna"]["min_trades_val"] = int(cfg["quick"]["min_trades_val"])

        # V6 model
        seq_len = int(cfg["features"]["seq_len"])
        model = ITransformerV6(
            lookback=seq_len,
            n_features=len(feature_cols),
            patch_len=int(cfg["model"].get("patch_len", 16)),
            d_model=int(cfg["model"].get("d_model", 192)),
            n_heads=int(cfg["model"].get("n_heads", 6)),
            n_layers_time=int(cfg["model"].get("n_layers_time", 2)),
            n_layers_feat=int(cfg["model"].get("n_layers_feat", 4)),
            dropout=float(cfg["model"].get("dropout", 0.12)),
            n_quantiles=len(cfg["model"]["quantiles"]),
        )

        model, scaler = train_model_v6(
            model=model,
            train_df=fold.train,
            val_df=fold.val,
            feature_cols=feature_cols,
            cfg=cfg,
            device=device,
            fold_id=fold.fold,
            out_dir=run_dir,
        )

        # predicciones V6 (en VAL y TEST)
        val_pred = predict_v6(model, scaler, fold.val, feature_cols, cfg, device)
        test_pred = predict_v6(model, scaler, fold.test, feature_cols, cfg, device)

        print("[debug] pred_df columns:", sorted(list(val_pred.columns)))

        # inyectar vol_scale en pred_df (VAL/TEST) antes de Optuna
        val_pred = _attach_vol_scale(val_pred, fold.val, cfg, tag="VAL")
        test_pred = _attach_vol_scale(test_pred, fold.test, cfg, tag="TEST")

        from deeplscalp.utils.preds import normalize_preds_for_trading
        val_pred = normalize_preds_for_trading(val_pred, cfg)
        test_pred = normalize_preds_for_trading(test_pred, cfg)

        # Optuna SOLO con VAL (esto evita leakage y sobreajuste al test)
        best = optuna_find_thresholds(
            pred_df=val_pred,
            cfg=cfg,
            fold_id=fold.fold,
        )

        best_thresholds = best["thresholds"]
        optuna_val_metrics = best.get("val_metrics", None)

        # Fail-fast: check VAL with best thresholds
        ff = cfg.get("eval", {}).get("fail_fast", {})
        ff_enabled = bool(ff.get("enabled", True))

        if ff_enabled:
            # aquí va tu lógica actual de fail-fast (mínimo trades, PF, MDD, etc.)
            pred_df_val = val_pred.copy()

            # Re-evalúa con el MISMO backtest y compara
            metrics_val, _ = backtest_from_predictions(pred_df_val, cfg, best_thresholds)

            if optuna_val_metrics is not None:
                if int(metrics_val.get("n_trades", 0)) != int(optuna_val_metrics.get("n_trades", 0)):
                    print("[FAIL_FAST][MISMATCH] Optuna y fail-fast difieren en n_trades.")
                    print("optuna_val_metrics=", optuna_val_metrics)
                    print("recalc_metrics_val=", metrics_val)
                    print("best_thresholds=", best_thresholds)
                    raise RuntimeError("Mismatch entre Optuna y fail-fast. Pipeline inconsistente (revisar pred_df/índices/desescalado).")

            min_trades_val = float(ff.get("min_trades_val", 30))
            min_pf_val = float(ff.get("min_pf_val", 1.05))
            max_mdd_val = float(ff.get("max_mdd_val", 0.12))
            if (metrics_val.get("n_trades", 0.0) < min_trades_val or
                metrics_val.get("profit_factor", 0.0) < min_pf_val or
                metrics_val.get("max_drawdown", 0.0) > max_mdd_val):
                print(f"[FAIL_FAST] fold={fold.fold} VAL no cumple: {metrics_val}. Deteniendo pipeline.")
                break

        # Backtest en TEST con thresholds fijos
        metrics, diag = backtest_from_predictions(
            pred_df=test_pred,
            cfg=cfg,
            thresholds=best["thresholds"],
        )

        row = {
            "fold": fold.fold,
            "best_value": best["best_value"],
            **best["thresholds"],
            **metrics,
            "diag_cost_rt": diag.get("cost_rt", None),
            "diag_mean_ret": diag.get("mean_ret", None),
        }
        all_fold_rows.append(row)

        print(f"[BEST] fold={fold.fold} value={best['best_value']:.6f} thresholds={best['thresholds']}")
        print(f"[fold {fold.fold}] TEST metrics: {metrics} | diag: {diag}")

    summary_path = run_dir / "summary.csv"
    write_summary_csv(pd.DataFrame(all_fold_rows), summary_path)
    print("\n[OK] FULL pipeline V2 finished.")
    print(f"summary: {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
