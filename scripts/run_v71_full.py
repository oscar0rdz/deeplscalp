# scripts/run_v71_full.py
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


def _norm_pair(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _infer_feature_cols(df: pd.DataFrame) -> List[str]:
    drop_prefix = ("y_", "yc_", "tbm_", "p_")
    drop_exact = {"ds", "pair", "timeframe", "open_next", "vol_scale"}
    cols = []
    for c in df.columns:
        if c in drop_exact:
            continue
        if c.startswith(drop_prefix):
            continue
        if c in ("open", "high", "low", "close", "volume", "atr"):
            cols.append(c)  # permitimos precio/atr si ya lo incluiste como feature
            continue
        # numéricas típicas
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    # evita duplicados y conserva orden
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _make_purged_splits(df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporal con purge/embargo para evitar leakage por horizonte TBM.
    Referencia conceptual: purged/embargo en validación de series financieras. :contentReference[oaicite:3]{index=3}
    """
    seq_len = int(cfg["features"]["seq_len"])
    base_h = int(cfg.get("labels", {}).get("base_horizon", 12))
    max_hold = int(cfg.get("risk", {}).get("max_hold_bars", 16))

    # embargo mínimo: evita que labels de train miren dentro de val/test
    embargo = int(cfg.get("split", {}).get("embargo_bars", max(base_h, max_hold) + 2))

    train_bars = int(cfg.get("split", {}).get("train_bars", 180000))
    val_bars = int(cfg.get("split", {}).get("val_bars", 50000))
    test_bars = int(cfg.get("split", {}).get("test_bars", 80000))

    n = len(df)
    train_end = min(train_bars, n)
    val_end = min(train_end + val_bars, n)
    test_end = min(val_end + test_bars, n)

    # purge al final de train y val
    train_end_p = max(0, train_end - embargo)
    val_end_p = max(train_end, val_end - embargo)

    train = df.iloc[:train_end_p].copy()

    # para val/test agregamos contexto previo (seq_len-1) para que el modelo tenga historia,
    # pero sin usar esos targets (por cómo funciona el dataset index >= seq_len-1).
    val_start_ctx = max(0, train_end - (seq_len - 1))
    val = df.iloc[val_start_ctx:val_end_p].copy()

    test_start_ctx = max(0, val_end - (seq_len - 1))
    test = df.iloc[test_start_ctx:test_end].copy()

    return train, val, test


def _dist(y: pd.Series) -> Dict[str, Dict]:
    vc = y.value_counts(dropna=False)
    tot = float(vc.sum()) if vc.sum() else 1.0
    out = {}
    for k, v in vc.items():
        out[str(k)] = {"n": int(v), "pct": float(v) / tot}
    return out


def _subperiod_report(pred: pd.DataFrame, cfg: Dict, thresholds: Dict, n_slices: int = 6) -> List[Dict]:
    from deeplscalp.backtest.sim_v71 import backtest_from_predictions_v71

    idx = pred.index
    if len(idx) < 1000:
        return []

    cuts = np.linspace(0, len(idx), n_slices + 1).astype(int)
    rep = []
    for i in range(n_slices):
        a, b = cuts[i], cuts[i + 1]
        chunk = pred.iloc[a:b].copy()
        m, d = backtest_from_predictions_v71(chunk, cfg, thresholds)
        rep.append({"slice": i, "start": str(chunk.index.min()), "end": str(chunk.index.max()), "metrics": m, "diag": d})
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    pair = args.pair
    device = cfg.get("device", args.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # threads CPU (tu Mac Intel lo agradece)
    rt = cfg.get("runtime", {})
    torch.set_num_threads(int(rt.get("torch_num_threads", 6)))

    # build dataset
    if not args.skip_build:
        t0 = time.time()
        subprocess.run([str(Path(".venv/bin/python")) if Path(".venv/bin/python").exists() else "python",
                        "pipeline.py", "--config", args.config, "build"], check=True)
        print(f"[TIME] build_done_sec={time.time() - t0:.1f}")

    out_dir = Path(cfg.get("project", {}).get("out_dir", "artifacts")).expanduser()
    _ensure_dir(out_dir / "preds")
    _ensure_dir(out_dir / "reports")
    _ensure_dir(out_dir / "models_v71")

    ds_path = out_dir / "datasets" / f"train_{_norm_pair(pair)}_5m_v71.parquet"
    if not ds_path.exists():
        raise FileNotFoundError(f"No existe dataset: {ds_path}")

    print(f"[RUN] dataset OK: {ds_path}")
    df = pd.read_parquet(ds_path)

    if "ds" not in df.columns:
        raise ValueError("Dataset debe incluir columna 'ds'.")
    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    df = df.sort_values("ds").reset_index(drop=True)
    df = df.set_index("ds")

    feature_cols = cfg.get("features", {}).get("columns")
    if not feature_cols:
        feature_cols = _infer_feature_cols(df)

    print(f"[DATA] rows={len(df)} seq_len={int(cfg['features']['seq_len'])} features={len(feature_cols)}")

    train_df, val_df, test_df = _make_purged_splits(df, cfg)

    # Reporte de distribuciones sobre TRAIN (sin NaNs)
    print(f"[SPLIT] train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    for lbl in ["y_side", "y_regime", "y_event"]:
        if lbl in train_df.columns:
            print(f"[DIST] {lbl}(train)={_dist(train_df[lbl])}")

    print(f"[DEVICE] {device}")

    # train
    from deeplscalp.modeling.train_v71 import train_model_v71, predict_v71  # tus módulos V7.1

    t0 = time.time()
    model, scaler = train_model_v71(
        train_df.reset_index(),  # muchos pipelines esperan ds como columna
        val_df.reset_index(),
        feature_cols=feature_cols,
        cfg=cfg,
        device=device,
        fold_id=0,
        out_dir=(out_dir / "models_v71"),
    )
    print(f"[TIME] train_done_sec={time.time() - t0:.1f}")

    # predict sobre TEST
    t1 = time.time()
    pred = predict_v71(
        model=model,
        scaler=scaler,
        df=test_df.reset_index(),
        feature_cols=feature_cols,
        cfg=cfg,
        device=device,
    )
    print(f"[TIME] predict_done_sec={time.time() - t1:.1f} rows={len(pred)}")

    pred_path = out_dir / "preds" / f"pred_{_norm_pair(pair)}_v71.parquet"
    pred.to_parquet(pred_path, index=True)
    print(f"[OUT] preds: {pred_path}")

    # backtest (base + stress + subperiods)
    from deeplscalp.backtest.sim_v71 import backtest_from_predictions_v71, stress_suite

    thresholds = dict(cfg.get("backtest", {}))
    t2 = time.time()
    metrics, diag = backtest_from_predictions_v71(pred, cfg, thresholds)
    print(f"[TIME] backtest_done_sec={time.time() - t2:.1f}")
    print("\n[METRICS]\n", metrics)
    print("\n[DIAG]\n", diag)

    stress = stress_suite(pred, cfg, thresholds)
    subp = _subperiod_report(pred, cfg, thresholds, n_slices=int(cfg.get("backtest", {}).get("subperiod_slices", 6)))

    report = {
        "pair": pair,
        "dataset_rows": int(len(df)),
        "split": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
        "features": feature_cols,
        "thresholds": thresholds,
        "metrics": metrics,
        "diag": diag,
        "stress": stress,
        "subperiods": subp,
    }

    rep_path = out_dir / "reports" / "run_v71_full_report.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OUT] report: {rep_path}")


if __name__ == "__main__":
    main()
