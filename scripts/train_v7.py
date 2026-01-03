#!/usr/bin/env python3
import argparse
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
    ap.add_argument("--val_days", type=int, default=10)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    out_dir = Path(cfg["project"]["out_dir"]).expanduser()
    norm = _norm_pair(args.pair)

    cands = sorted((out_dir / "datasets").glob(f"*{norm}*5m*.parquet"))
    if not cands:
        raise FileNotFoundError(f"No encontré parquet para {norm} en {out_dir}/datasets")
    path = cands[0]
    df = pd.read_parquet(path)
    if df.index.name == 'ds':
        df = df.reset_index()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values("ds").reset_index(drop=True).set_index('ds')

    # separa train/val cronológico
    end = df.index.max()
    cut = end - pd.Timedelta(days=int(args.val_days))
    train_df = df[df.index < cut].copy()
    val_df = df[df.index >= cut].copy()

    # define feature_cols: todo lo numérico excepto columnas no-features
    exclude = {
        "ds","open","high","low","close","volume","pair","timeframe",
        "y_side","yc_long","yc_short","yL_reg","yS_reg","sample_weight",
        "vol_scale","gap_flag",
        # si exportas tbm extras:
        "tbm_label","tbm_yc","tbm_entry_px","tbm_exit_px","tbm_exit_i","tbm_ret_gross","tbm_horizon",
    }
    feature_cols = [c for c in train_df.columns
                    if c not in exclude and np.issubdtype(train_df[c].dtype, np.number)]

    # limpia filas con NaN en features/targets
    need_targets = [c for c in ["y_side","yc_long","yc_short","yL_reg","yS_reg"] if c in train_df.columns]
    keep_cols = feature_cols + need_targets
    train_df = train_df.dropna(subset=keep_cols)
    val_df = val_df.dropna(subset=keep_cols)

    print(f"[DATA] {path}")
    print(f"[DATA] train={len(train_df)} val={len(val_df)} features={len(feature_cols)}")

    device = pick_device(args.device)
    print("[DEVICE]", device)

    # importa modelo + trainer v7
    from training.itransformer_v7 import ITransformerV7
    from deeplscalp.modeling.train import train_model_v7

    model = ITransformerV7(
        lookback=int(cfg["features"]["seq_len"]),
        n_features=len(feature_cols),
        d_model=int(cfg["model"]["d_model"]),
        n_heads=int(cfg["model"]["nhead"]),
        n_layers_time=2,
        n_layers_feat=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        n_quantiles=len(cfg["model"]["quantiles"]),
    )

    model_dir = out_dir / "models_v7"
    model_dir.mkdir(parents=True, exist_ok=True)

    model, scaler = train_model_v7(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        cfg=cfg,
        device=device,
        fold_id=int(args.fold),
        out_dir=model_dir,
    )

    print(f"[OK] entrenado. ckpt en {model_dir}")

if __name__ == "__main__":
    main()
