#!/usr/bin/env python3
import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

def _norm_pair(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pair", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    out_dir = Path(cfg["project"]["out_dir"]).expanduser()
    norm = _norm_pair(args.pair)

    # busca cualquier parquet del par
    cands = sorted((out_dir / "datasets").glob(f"*{norm}*5m*.parquet"))
    if not cands:
        raise FileNotFoundError(f"No encontré parquet para {norm} en {out_dir}/datasets")

    path = cands[0]
    df = pd.read_parquet(path)
    if df.index.name == 'ds':
        df = df.reset_index()
    print(f"[OK] dataset={path} rows={len(df)} cols={len(df.columns)}")
    print(df.head(3))

    # columnas mínimas típicas
    must = ["ds","open","high","low","close","volume"]
    for c in must:
        if c not in df.columns:
            raise KeyError(f"Falta columna base {c}")

    # revisa NaNs generales
    nan_rate = df.isna().mean().sort_values(ascending=False).head(15)
    print("\n[NaN top 15]")
    print(nan_rate)

    # si tu V7 exporta estas etiquetas:
    label_cols = ["y_side","yc_long","yc_short","yL_reg","yS_reg","vol_scale"]
    have = [c for c in label_cols if c in df.columns]
    print("\n[label cols presentes]", have)

    for c in ["y_side","yc_long","yc_short"]:
        if c in df.columns:
            vc = df[c].value_counts(dropna=False, normalize=True)
            print(f"\n[{c} distribution]\n{vc}")

    # sanity: vol_scale no debe ser 0 constante
    if "vol_scale" in df.columns:
        vs = df["vol_scale"].replace([np.inf,-np.inf], np.nan).dropna()
        print("\n[vol_scale] min/med/max:", float(vs.min()), float(vs.median()), float(vs.max()))

    print("\n[OK] smoke test terminó")

if __name__ == "__main__":
    main()
