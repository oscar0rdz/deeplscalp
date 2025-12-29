#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
import pandas as pd
import yaml

try:
    import pandas_ta as pta
except Exception:
    pta = None


TF_TO_PANDAS = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
}


def _norm_pair(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_ohlcv_file(datadir: Path, pair: str, tf: str) -> Path:
    norm = _norm_pair(pair)
    for ext in (".parquet", ".json", ".feather", ".csv"):
        p = datadir / f"{norm}-{tf}{ext}"
        if p.exists():
            return p
    globbed = sorted(datadir.glob(f"{norm}-{tf}.*"))
    if globbed:
        return globbed[0]
    raise FileNotFoundError(f"No OHLCV file for {pair} {tf} in {datadir}")


def read_ohlcv(path: Path, pair: str, tf: str) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".feather":
        df = pd.read_feather(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".json":
        raw = pd.read_json(path)
        if raw.shape[1] == 1 and isinstance(raw.iloc[0, 0], (list, tuple)):
            arr = np.array(raw.iloc[:, 0].to_list(), dtype="float64")
            df = pd.DataFrame(arr, columns=["timestamp", "open", "high", "low", "close", "volume"])
        else:
            df = raw.copy()
    else:
        raise ValueError(f"Unsupported ext: {ext}")

    cols = {c.lower(): c for c in df.columns}
    tscol = cols.get("timestamp") or cols.get("date") or cols.get("ds")
    if not tscol:
        raise ValueError(f"Timestamp col not found in {path} cols={list(df.columns)}")

    df = df.rename(columns={
        tscol: "timestamp",
        cols.get("open", "open"): "open",
        cols.get("high", "high"): "high",
        cols.get("low", "low"): "low",
        cols.get("close", "close"): "close",
        cols.get("volume", "volume"): "volume",
    })

    # timestamp puede venir como datetime (incluyendo tz-aware) en Parquet de Freqtrade
    if is_datetime64_any_dtype(df["timestamp"]):
        # Si ya trae tz, utc=True lo respeta; si es naive, lo convierte a UTC
        df["ds"] = pd.to_datetime(df["timestamp"], utc=True)
    elif is_numeric_dtype(df["timestamp"]):
        ts = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
        # Heurística: ms vs s
        med = ts.dropna().astype("int64").median() if ts.dropna().shape[0] else 0
        if med > 10_000_000_000:
            df["ds"] = pd.to_datetime(ts.astype("int64"), unit="ms", utc=True)
        else:
            df["ds"] = pd.to_datetime(ts.astype("int64"), unit="s", utc=True)
    else:
        df["ds"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    out = df[["ds", "open", "high", "low", "close", "volume"]].copy()
    out["pair"] = pair
    out["timeframe"] = tf
    out = out.sort_values("ds").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["ds", "open", "high", "low", "close"])
    return out


def audit_ohlcv(df: pd.DataFrame, tf: str) -> Dict:
    freq = TF_TO_PANDAS[tf]
    ds = df["ds"].sort_values()
    dup = int(ds.duplicated().sum())
    full = pd.date_range(ds.min(), ds.max(), freq=freq, tz="UTC")
    missing = int(len(full.difference(ds)))
    missing_pct = float(missing) / float(len(full)) if len(full) else 0.0
    delta = ds.diff().dropna().dt.total_seconds().to_numpy()
    return {
        "rows": int(df.shape[0]),
        "start": str(ds.min()),
        "end": str(ds.max()),
        "duplicates": dup,
        "missing_candles": missing,
        "missing_pct": missing_pct,
        "median_step_seconds": float(np.median(delta)) if len(delta) else 0.0,
        "max_gap_seconds": float(delta.max()) if len(delta) else 0.0,
    }


def clean_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    freq = TF_TO_PANDAS[tf]
    df = df.drop_duplicates(subset=["ds"]).sort_values("ds").copy()
    df["volume"] = df["volume"].clip(lower=0)
    step = pd.Timedelta(freq).total_seconds()
    gap_seconds = df["ds"].diff().dt.total_seconds().fillna(step)
    df["gap_flag"] = gap_seconds > (1.5 * step)
    return df.reset_index(drop=True)


def resample_ohlcv(df_1m: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    freq = TF_TO_PANDAS[target_tf]
    df = df_1m.set_index("ds").sort_index()
    out = pd.DataFrame(index=df["close"].resample(freq).ohlc().index)
    out["open"] = df["open"].resample(freq).first()
    out["high"] = df["high"].resample(freq).max()
    out["low"] = df["low"].resample(freq).min()
    out["close"] = df["close"].resample(freq).last()
    out["volume"] = df["volume"].resample(freq).sum()
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index().rename(columns={"index": "ds"})
    return out


def add_micro_features_from_1m(df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> pd.DataFrame:
    # Usamos SOLO columnas numéricas: evita que resample intente operar sobre strings.
    base = df_1m[["ds", "open", "high", "low", "close", "volume"]].copy().set_index("ds").sort_index()

    # Retorno 1m para realized volatility
    base["ret1m"] = np.log(base["close"]).diff()

    # Resample 5m (index base de 5m)
    r = base.resample(TF_TO_PANDAS["5m"])

    # DataFrame micro con ds explícito (no dependemos de grp.mean() ni de apply sobre DF)
    micro = pd.DataFrame({"ds": r.size().index})

    # Realized volatility 5m (sqrt(sum(ret^2)))
    micro["rv_5m"] = np.sqrt(r["ret1m"].apply(lambda s: float(np.nansum(np.square(s.to_numpy())))))

    # Volumen total 5m
    micro["vol_sum_5m"] = r["volume"].sum()

    # Ratio de volumen “alcista” (close > open)
    base["up_vol"] = np.where(base["close"] > base["open"], base["volume"], 0.0)
    micro["up_vol_5m"] = base["up_vol"].resample(TF_TO_PANDAS["5m"]).sum()
    micro["up_vol_ratio_5m"] = micro["up_vol_5m"] / micro["vol_sum_5m"].clip(lower=1e-12)

    # Unimos micro features al df_5m
    out = df_5m.merge(micro[["ds", "rv_5m", "vol_sum_5m", "up_vol_ratio_5m"]], on="ds", how="left")

    # Range relativo 5m (ya está correctamente agregado en df_5m)
    out["range_5m"] = (out["high"] - out["low"]) / out["close"].abs().clip(lower=1e-12)

    # Limpieza suave (si hay bins sin datos por gaps)
    out["rv_5m"] = out["rv_5m"].fillna(0.0)
    out["up_vol_ratio_5m"] = out["up_vol_ratio_5m"].fillna(0.0)
    out["vol_sum_5m"] = out["vol_sum_5m"].fillna(0.0)

    return out


def align_timeframes(pair_df_1m: pd.DataFrame, anchor_df_1m: pd.DataFrame, informative_tfs: List[str]) -> pd.DataFrame:
    df_5m = resample_ohlcv(pair_df_1m, "5m")
    df_5m = add_micro_features_from_1m(pair_df_1m, df_5m)

    base = df_5m.set_index("ds").sort_index()

    for tf in informative_tfs:
        inf = resample_ohlcv(pair_df_1m, tf).set_index("ds").sort_index()
        tmp = inf[["open", "high", "low", "close", "volume"]].copy()
        tmp.columns = [f"{tf}_{c}" for c in tmp.columns]
        tmp = tmp.reindex(base.index, method="ffill")
        base = base.join(tmp)

    btc5 = resample_ohlcv(anchor_df_1m, "5m").set_index("ds").sort_index()
    btc5["btc_ret5m"] = np.log(btc5["close"]).diff()
    btc5["btc_atr5m"] = (btc5["high"] - btc5["low"]).rolling(14, min_periods=14).mean()
    btc5 = btc5[["close", "btc_ret5m", "btc_atr5m"]].rename(columns={"close": "btc_close_5m"})
    btc5 = btc5.reindex(base.index, method="ffill")
    base = base.join(btc5)

    return base.reset_index()


def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def compute_features(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    df = df.copy().sort_values("ds").reset_index(drop=True)
    close, high, low, open_, vol = df["close"], df["high"], df["low"], df["open"], df["volume"]

    df["ret_1"] = np.log(close).diff()
    df["ret_3"] = np.log(close).diff(3)
    df["ret_6"] = np.log(close).diff(6)

    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_n = int(cfg["labels"]["atr_period"])
    df["atr"] = tr.rolling(atr_n, min_periods=atr_n).mean()
    df["atr_ratio"] = df["atr"] / close

    adx_n = int(cfg["features"].get("adx_period", 14))
    if pta is not None:
        adx = pta.adx(high, low, close, length=adx_n)
        df["adx"] = adx[f"ADX_{adx_n}"]
    else:
        df["adx"] = (df["ret_1"].abs().rolling(adx_n).mean() / (df["ret_1"].rolling(adx_n).std() + 1e-12)) * 10

    rsi_n = int(cfg["features"].get("rsi_period", 14))
    if pta is not None:
        df["rsi"] = pta.rsi(close, length=rsi_n)
    else:
        delta = close.diff()
        up = delta.clip(lower=0).rolling(rsi_n).mean()
        down = (-delta.clip(upper=0)).rolling(rsi_n).mean()
        rs = up / (down + 1e-12)
        df["rsi"] = 100 - (100 / (1 + rs))

    fast = int(cfg["features"].get("macd_fast", 12))
    slow = int(cfg["features"].get("macd_slow", 26))
    sig = int(cfg["features"].get("macd_signal", 9))
    macd_line = _ema(close, fast) - _ema(close, slow)
    macd_sig = _ema(macd_line, sig)
    df["macd_hist"] = macd_line - macd_sig
    df["macd_hist_slope"] = df["macd_hist"].diff()

    bb_n = int(cfg["features"].get("bb_period", 20))
    bb_std = float(cfg["features"].get("bb_std", 2.0))
    m = close.rolling(bb_n, min_periods=bb_n).mean()
    s = close.rolling(bb_n, min_periods=bb_n).std()
    df["bb_mid"] = m
    df["bb_up"] = m + bb_std * s
    df["bb_dn"] = m - bb_std * s
    df["bb_width"] = (df["bb_up"] - df["bb_dn"]) / (df["bb_mid"] + 1e-12)
    df["bb_z"] = (close - df["bb_mid"]) / (s + 1e-12)

    don_n = int(cfg["features"].get("donchian_period", 20))
    df["don_hi"] = high.rolling(don_n, min_periods=don_n).max()
    df["don_lo"] = low.rolling(don_n, min_periods=don_n).min()
    df["don_rng"] = (df["don_hi"] - df["don_lo"]).replace(0, np.nan)
    df["don_pos"] = (close - df["don_lo"]) / (df["don_rng"] + 1e-12)

    body = (close - open_).abs()
    upper_wick = high - close.where(close >= open_, open_)
    lower_wick = close.where(close <= open_, open_) - low
    df["upper_wick_ratio"] = upper_wick / (body + 1e-12)
    df["lower_wick_ratio"] = lower_wick / (body + 1e-12)
    df["close_in_range"] = ((close < df["don_hi"]) & (close > df["don_lo"])).astype(int)

    vol_n = int(cfg["features"].get("vol_lookback", 48))
    vol_mean = vol.rolling(vol_n, min_periods=vol_n).mean()
    vol_std = vol.rolling(vol_n, min_periods=vol_n).std()
    df["vol_z"] = (vol - vol_mean) / (vol_std + 1e-12)
    df["vol_rel"] = vol / (vol_mean + 1e-12)

    tp = (high + low + close) / 3.0
    day = df["ds"].dt.floor("D")
    cum_v = vol.groupby(day).cumsum()
    cum_pv = (tp * vol).groupby(day).cumsum()
    df["vwap_d"] = cum_pv / (cum_v + 1e-12)
    df["vwap_dev"] = (close - df["vwap_d"]) / (df["vwap_d"] + 1e-12)

    chop_n = int(cfg["features"].get("chop_period", 14))
    tr_sum = tr.rolling(chop_n, min_periods=chop_n).sum()
    hi = high.rolling(chop_n, min_periods=chop_n).max()
    lo = low.rolling(chop_n, min_periods=chop_n).min()
    df["chop"] = 100 * np.log10(tr_sum / (hi - lo + 1e-12)) / np.log10(chop_n)

    trf_fast = int(cfg["features"].get("twin_range_fast", 7))
    trf_slow = int(cfg["features"].get("twin_range_slow", 28))
    rng = (high - low).rolling(2, min_periods=2).mean()
    fr = _ema(rng, trf_fast)
    sr = _ema(rng, trf_slow)
    df["trf_range"] = (fr + sr) / 2.0
    filt = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df)):
        if i == 0 or np.isnan(df.loc[i, "trf_range"]):
            continue
        prev = filt[i - 1] if not np.isnan(filt[i - 1]) else close.iloc[i - 1]
        r = df.loc[i, "trf_range"]
        c = close.iloc[i]
        filt[i] = max(prev, c - r) if c > prev else min(prev, c + r)
    df["trf_line"] = filt
    df["trf_dir"] = np.sign(pd.Series(filt).diff())

    df["btc_vol_regime"] = df["btc_atr5m"] / (df["btc_close_5m"] + 1e-12)
    df["rv_norm"] = df["rv_5m"] / (df["atr_ratio"] + 1e-12)

    df["feat_breakout_pressure"] = (
        (df["don_pos"] > 0.9).astype(int)
        + (df["bb_width"] < df["bb_width"].rolling(96, min_periods=96).quantile(0.25)).astype(int)
        + (df["vol_z"] > 1.0).astype(int)
    )

    df["feat_rebound_pressure"] = (
        (df["vwap_dev"] < -0.005).astype(int)
        + (df["rsi"] < 30).astype(int)
        + (df["lower_wick_ratio"] > 1.5).astype(int)
    )

    return df


def dynamic_horizon(df: pd.DataFrame, cfg: Dict) -> np.ndarray:
    hmin = int(cfg["labels"]["horizon_min"])
    hmax = int(cfg["labels"]["horizon_max"])
    base_h = int(cfg["labels"].get("base_horizon", 6))
    atr = df["atr_ratio"].to_numpy()
    med = pd.Series(atr).rolling(288, min_periods=288).median().to_numpy()
    H = np.full(len(df), base_h, dtype=int)
    for i in range(len(df)):
        if np.isnan(atr[i]) or np.isnan(med[i]) or med[i] == 0:
            continue
        scale = med[i] / atr[i]
        H[i] = int(round(base_h * scale))
    return np.clip(H, hmin, hmax)


def label_triple_barrier(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr = df["atr"].to_numpy()

    tp_k = float(cfg["labels"]["tp_atr_mult"])
    sl_k = float(cfg["labels"]["sl_atr_mult"])

    use_dyn = cfg["labels"].get("dynamic_horizon", {}).get("enabled", True)
    H = dynamic_horizon(df, cfg) if use_dyn else np.full(len(df), int(cfg["labels"].get("base_horizon", 6)), dtype=int)

    labels = np.zeros(len(df), dtype=int)
    hit = np.array(["time"] * len(df), dtype=object)

    for i in range(len(df)):
        if np.isnan(atr[i]) or i + H[i] >= len(df):
            continue
        tp = close[i] + tp_k * atr[i]
        sl = close[i] - sl_k * atr[i]
        end = i + H[i]
        for j in range(i + 1, end + 1):
            if high[j] >= tp:
                labels[i] = 1
                hit[i] = "tp"
                break
            if low[j] <= sl:
                labels[i] = -1
                hit[i] = "sl"
                break

    df["tbm_label"] = labels
    df["tbm_hit"] = hit
    df["tbm_horizon"] = H
    return df


def export_neuralforecast_dataset(df: pd.DataFrame, pair: str, outpath: Path) -> None:
    df = df.copy().sort_values("ds").reset_index(drop=True)
    df["y"] = df["ret_1"]
    df["unique_id"] = pair

    feature_cols = [
        "atr_ratio", "adx", "rsi", "macd_hist", "macd_hist_slope",
        "bb_width", "bb_z", "don_pos",
        "upper_wick_ratio", "lower_wick_ratio", "close_in_range",
        "vol_z", "vol_rel", "vwap_dev", "chop",
        "trf_dir", "feat_breakout_pressure", "feat_rebound_pressure",
        "btc_ret5m", "btc_vol_regime",
        "rv_5m", "up_vol_ratio_5m", "range_5m", "rv_norm",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    out = df[["unique_id", "ds", "y"] + feature_cols + ["tbm_label", "tbm_hit", "tbm_horizon"]].copy()
    out = out.dropna(subset=["y"] + feature_cols)
    _ensure_dir(outpath.parent)
    out.to_parquet(outpath, index=False)


def process_pair(pair: str, cfg: Dict, raw_dir: Path, out_dir: Path, anchor_pair: str) -> Path:
    p1 = find_ohlcv_file(raw_dir, pair, cfg["timeframes"]["exec_tf"])
    a1 = find_ohlcv_file(raw_dir, anchor_pair, cfg["timeframes"]["exec_tf"])

    df1 = read_ohlcv(p1, pair, cfg["timeframes"]["exec_tf"])
    btc1 = read_ohlcv(a1, anchor_pair, cfg["timeframes"]["exec_tf"])

    df1c = clean_ohlcv(df1, cfg["timeframes"]["exec_tf"])
    btc1c = clean_ohlcv(btc1, cfg["timeframes"]["exec_tf"])

    aligned = align_timeframes(df1c, btc1c, cfg["timeframes"]["informative_tfs"])
    feats = compute_features(aligned, cfg)
    labeled = label_triple_barrier(feats, cfg)

    outpath = out_dir / "datasets" / f"nf_{_norm_pair(pair)}_5m.parquet"
    export_neuralforecast_dataset(labeled, pair, outpath)
    return outpath


def cmd_audit(cfg: Dict) -> None:
    raw_dir = Path(cfg["freqtrade"]["datadir"]).expanduser()
    out_dir = Path(cfg["project"]["out_dir"]).expanduser()
    _ensure_dir(out_dir / "data_audit")

    for pair in cfg["universe"]["pairs"]:
        for tf in cfg["timeframes"]["timeframes_to_download"]:
            try:
                p = find_ohlcv_file(raw_dir, pair, tf)
                df = read_ohlcv(p, pair, tf)
                rep = audit_ohlcv(df, tf)
                rep.update({"pair": pair, "timeframe": tf, "file": str(p)})
            except Exception as e:
                rep = {"pair": pair, "timeframe": tf, "error": str(e)}
            outp = out_dir / "data_audit" / f"audit_{_norm_pair(pair)}_{tf}.json"
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2)


def cmd_build(cfg: Dict) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    raw_dir = Path(cfg["freqtrade"]["datadir"]).expanduser()
    out_dir = Path(cfg["project"]["out_dir"]).expanduser()
    _ensure_dir(out_dir / "datasets")

    anchor = cfg["universe"]["anchor_pair"]
    pairs = [p for p in cfg["universe"]["pairs"] if p != anchor]
    workers = int(cfg["project"].get("workers", max(1, os.cpu_count() // 2)))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_pair, pair, cfg, raw_dir, out_dir, anchor) for pair in pairs]
        for f in as_completed(futs):
            print(f"[OK] wrote {f.result()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit")
    sub.add_parser("build")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.cmd == "audit":
        cmd_audit(cfg)
    else:
        cmd_build(cfg)


if __name__ == "__main__":
    main()
