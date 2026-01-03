import os, glob
import numpy as np
import pandas as pd
import yaml
from pandas.api.types import is_datetime64_any_dtype

EPS = 1e-12

def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_datetime_utc(s: pd.Series) -> pd.Series:
    # ds ya debería venir como datetime; solo blindamos
    if is_datetime64_any_dtype(s):
        return pd.to_datetime(s, utc=True)
    return pd.to_datetime(s, utc=True, errors="coerce")

def compute_atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    # Wilder smoothing (EWMA alpha=1/period)
    atr = tr.ewm(alpha=1.0/period, adjust=False).mean()
    return atr

def adaptive_hold(atr: np.ndarray, base: int, hmin: int, hmax: int, ref_span: int, smin: float, smax: float) -> np.ndarray:
    atr_s = pd.Series(atr)
    ref = atr_s.ewm(span=ref_span, adjust=False).mean().to_numpy()

    den = np.clip(atr, EPS, None)
    scale = ref / den
    scale = np.clip(scale, smin, smax)

    H = np.rint(base * scale).astype(np.int32)
    H = np.clip(H, hmin, hmax).astype(np.int32)
    return H

def label_event_driven(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    L = cfg["labeling"]
    costs = L["costs"]

    # Required columns
    for c in ("unique_id", "ds", "close", "high", "low"):
        if c not in df.columns:
            raise ValueError(f"Dataset missing required column: {c}")

    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True).copy()
    df["ds"] = ensure_datetime_utc(df["ds"])

    # ATR
    atr_col = "atr" if "atr" in df.columns else None
    if atr_col is None:
        atr = compute_atr_wilder(df["high"], df["low"], df["close"], int(L.get("atr_period", 14)))
        df["atr"] = atr
        atr_col = "atr"

    atr = df[atr_col].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low  = df["low"].to_numpy(dtype=np.float64)

    # Costs as return (round-trip + buffer)
    fee = float(costs.get("fee_per_side", 0.001))
    slip_bps = float(costs.get("slippage_bps_per_side", 2))
    spread_bps = float(costs.get("spread_bps", 2))
    buf_bps = float(costs.get("min_edge_buffer_bps", 5))

    cost_total = (2.0*fee) + (2.0*slip_bps/10000.0) + (spread_bps/10000.0) + (buf_bps/10000.0)

    # TP/SL distances (price units)
    tp_dist = float(L["tp_atr_mult"]) * atr
    sl_dist = float(L["sl_atr_mult"]) * atr

    # Convert to return terms for edge checks
    tp_ret = tp_dist / np.clip(close, EPS, None)
    sl_ret = sl_dist / np.clip(close, EPS, None)

    # Adaptive hold (vertical barrier)
    Hcfg = L["hold_adaptive"]
    H = adaptive_hold(
        atr=atr,
        base=int(Hcfg["base_bars"]),
        hmin=int(Hcfg["min_bars"]),
        hmax=int(Hcfg["max_bars"]),
        ref_span=int(Hcfg["ref_ema_span"]),
        smin=float(Hcfg["scale_clip_min"]),
        smax=float(Hcfg["scale_clip_max"]),
    )

    # Output arrays
    n = len(df)
    hit = np.empty(n, dtype=object)
    y = np.zeros(n, dtype=np.float64)
    hold_used = np.zeros(n, dtype=np.int32)
    sample_weight = np.ones(n, dtype=np.float32)

    # Skip trades where TP cannot beat costs (model should learn to NOT trade)
    skip = np.zeros(n, dtype=bool)
    if bool(L.get("skip_if_edge_below_cost", True)):
        skip = tp_ret <= cost_total
        sample_weight[skip] = 0.0

    # Event-driven loop (efficient enough with small H)
    for i in range(n):
        if not np.isfinite(close[i]) or close[i] <= 0:
            hit[i] = "time"
            y[i] = 0.0
            hold_used[i] = 0
            sample_weight[i] = 0.0
            continue

        if skip[i]:
            # Treat as non-tradable sample
            hit[i] = "time"
            y[i] = 0.0
            hold_used[i] = 0
            continue

        tp_price = close[i] + tp_dist[i]
        sl_price = close[i] - sl_dist[i]

        hmax_i = int(H[i])
        j_end = min(i + hmax_i, n - 1)

        # If there is no room ahead, label as time
        if j_end <= i:
            hit[i] = "time"
            y[i] = 0.0
            hold_used[i] = 0
            continue

        decided = False
        for j in range(i + 1, j_end + 1):
            # Conservative intrabar resolution:
            # If both barriers are touched in the same bar, assume SL hits first (avoids optimistic bias).
            sl_touched = low[j] <= sl_price
            tp_touched = high[j] >= tp_price

            if sl_touched and tp_touched:
                hit[i] = "sl"
                y[i] = (-sl_ret[i]) - cost_total
                hold_used[i] = j - i
                decided = True
                break
            elif sl_touched:
                hit[i] = "sl"
                y[i] = (-sl_ret[i]) - cost_total
                hold_used[i] = j - i
                decided = True
                break
            elif tp_touched:
                hit[i] = "tp"
                y[i] = (tp_ret[i]) - cost_total
                hold_used[i] = j - i
                decided = True
                break

        if not decided:
            # time barrier: exit at j_end close
            exit_ret = (close[j_end] - close[i]) / close[i]
            hit[i] = "time"
            y[i] = exit_ret - cost_total
            hold_used[i] = j_end - i

    out = df.copy()
    out["tbm_hit"] = hit
    out["y"] = y
    out["hold_bars"] = hold_used
    out["tp_ret"] = tp_ret.astype(np.float32)
    out["sl_ret"] = sl_ret.astype(np.float32)
    out["sample_weight"] = sample_weight

    return out

def cast_float32(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in ("unique_id", "ds", "tbm_hit"):
            continue
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype("int32")
    return df

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    in_glob = cfg["io"]["in_glob"]
    out_dir = cfg["io"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(in_glob))
    if not files:
        raise SystemExit(f"No input files found for glob: {in_glob}")

    comp = cfg["io"].get("compression", "zstd")
    do_cast = bool(cfg["io"].get("cast_float32", True))

    for f in files:
        # Handle feather files for raw OHLC
        if f.endswith('.feather'):
            df = pd.read_feather(f)
            # Extract pair from filename, e.g., ETH_USDT-5m.feather -> ETH/USDT
            base = os.path.basename(f)
            pair = base.split('-')[0].replace('_', '/')
            df['unique_id'] = pair
            df['ds'] = pd.to_datetime(df['date'], utc=True)
            # Keep only OHLC + unique_id, ds
            df = df[['unique_id', 'ds', 'open', 'high', 'low', 'close', 'volume']].copy()
        else:
            df = pd.read_parquet(f, engine="pyarrow")

        out = label_event_driven(df, cfg)
        if do_cast:
            out = cast_float32(out)

        if f.endswith('.feather'):
            base = os.path.basename(f).replace('-5m.feather', '_5m.parquet').replace('_', '')
            base = f"nf_event_{base}"
        else:
            base = os.path.basename(f).replace("nf_", "nf_event_")
        outpath = os.path.join(out_dir, base)
        out.to_parquet(outpath, index=False, compression=comp)

        # Minimal summary (sin pedirte revisiones extra)
        vc = out["tbm_hit"].value_counts(dropna=False).to_dict()
        print(f"[OK] wrote {outpath} rows={len(out)} cols={len(out.columns)} tbm_hit={vc}")

if __name__ == "__main__":
    main()
