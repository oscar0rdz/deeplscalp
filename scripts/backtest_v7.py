#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch


def _norm_pair(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


def pick_device(req: str) -> str:
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-bars", type=int, default=4000, help="Usa solo las últimas N barras para acelerar.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    out_dir = Path(cfg["project"]["out_dir"]).expanduser()
    norm = _norm_pair(args.pair)

    # Torch threads (evita que se vuelva loco en CPU)
    try:
        torch.set_num_threads(min(8, os.cpu_count() or 8))
    except Exception:
        pass

    # Dataset
    cands = sorted((out_dir / "datasets").glob(f"*{norm}*5m*v7*.parquet"))
    if not cands:
        cands = sorted((out_dir / "datasets").glob(f"*{norm}*5m*.parquet"))
    if not cands:
        raise FileNotFoundError(f"No encontré parquet para {norm} en {out_dir}/datasets")

    path = cands[0]
    df = pd.read_parquet(path)

    if df.index.name == 'ds':
        df = df.reset_index()
    # ahora 'ds' es columna
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values("ds").reset_index(drop=True).set_index('ds')

    seq_len = int(cfg["features"]["seq_len"])
    # recorta para acelerar (con warmup para no romper ventanas)
    keep = int(args.max_bars) + seq_len + 50
    if len(df) > keep:
        df = df.iloc[-keep:].copy()

    print(f"[DATA] {path}")
    print(f"[DATA] rows={len(df)} seq_len={seq_len}")

    # Feature cols
    exclude = {
        "ds","open","high","low","close","volume","pair","timeframe",
        "y_side","yc_long","yc_short","yL_reg","yS_reg","sample_weight",
        "vol_scale","gap_flag",
        "tbm_label","tbm_yc","tbm_entry_px","tbm_exit_px","tbm_exit_i","tbm_ret_gross","tbm_horizon",
    }
    feature_cols = [c for c in df.columns
                    if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

    # Limpia NaNs en features (inferencia)
    df = df.dropna(subset=feature_cols + ["open","high","low","close"]).reset_index(drop=True)

    device = pick_device(args.device)
    print("[DEVICE]", device)
    print("[FEATURES]", len(feature_cols))

    # Modelo y predict
    from training.itransformer_v7 import ITransformerV7
    from deeplscalp.modeling.train import predict_v7
    from deeplscalp.backtest.sim_v7 import backtest_from_predictions_v7

    # IMPORTANTE: ajusta estos args a TU constructor real si difiere
    model = ITransformerV7(
        lookback=seq_len,
        n_features=len(feature_cols),
        d_model=int(cfg["model"]["d_model"]),
        n_heads=int(cfg["model"]["nhead"]),
        n_layers_time=2,
        n_layers_feat=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        n_quantiles=len(cfg["model"]["quantiles"]),
    )

    model_dir = out_dir / "models_v7"
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = model_dir / f"fold_{int(args.fold)}_best.pt"
    obj = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(obj["model"])
    model.to(device).eval()

    class Scaler:
        def __init__(self, mean_, std_):
            self.mean_ = mean_
            self.std_ = std_
        def transform(self, x):
            return (x - self.mean_) / (self.std_ + 1e-12)

    scaler = Scaler(obj["scaler_mean"], obj["scaler_std"])

    t0 = time.perf_counter()
    pred_df = predict_v7(model, scaler, df, feature_cols, cfg, device=device)
    t1 = time.perf_counter()
    print(f"[TIME] predict_v7: {t1 - t0:.2f}s rows={len(pred_df)}")

    thresholds = {
        # "Modo muestra" (más trades para validar ciclo)
        "p_side_min": 0.50,
        "p_tp_min": 0.20,
        "p_sl_max": 0.50,
        "ev_min": 0.00015,
        "use_topk": False,   # desactiva para acelerar y evitar gate duro
        "top_k": 12,
        "cooldown_bars": 1,
        # rolling thresholds
        "score_q": 0.65,
        "ood_q": 0.90,
        "thr_lookback_bars": 1500,
        "thr_min_periods": 300,
    }

    t2 = time.perf_counter()
    metrics, diag = backtest_from_predictions_v7(pred_df, cfg, thresholds)
    t3 = time.perf_counter()
    print(f"[TIME] backtest: {t3 - t2:.2f}s")

    print("\n[METRICS]")
    print(metrics)
    print("\n[DIAG]")
    print(diag)


if __name__ == "__main__":
    main()
