import numpy as np
import pandas as pd

_WARNED = {"no_close": False, "proxy": False}


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_z(x: np.ndarray, win_mean: np.ndarray, win_std: np.ndarray) -> np.ndarray:
    z = np.zeros_like(x, dtype=np.float64)
    m = np.isfinite(win_std) & (win_std > 0)
    z[m] = (x[m] - win_mean[m]) / win_std[m]
    return z


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=np.float64)
    m = np.isfinite(den) & (den > 0)
    out[m] = num[m] / den[m]
    return out


def compute_event_masks(df: pd.DataFrame, params: dict) -> dict:
    """
    Eventos SOLO con info presente/pasada (sin labels, sin futuro).

    Modos:
    A) OHLCV disponible (close + opcional open/high/low/volume): eventos breakout/volburst/sweep/meanrev.
    B) OHLCV no disponible: eventos PROXY con features existentes (por ejemplo feat_breakout_pressure, bb_z, vol_z...).
       Esto evita "desactivar eventos" y mantiene el pipeline event-driven de forma realista.
    """
    p = params or {}

    # --- (1) mapping explícito opcional ---
    col_close = p.get("col_close")
    col_open  = p.get("col_open")
    col_high  = p.get("col_high")
    col_low   = p.get("col_low")
    col_vol   = p.get("col_volume")

    # --- (2) auto-detección si no viene mapeado ---
    if col_close is None:
        col_close = _find_col(df, ["close", "Close", "CLOSE", "c", "C", "last", "Last", "price", "close_price"])
    if col_open is None:
        col_open = _find_col(df, ["open", "Open", "OPEN", "o", "O", "open_price"])
    if col_high is None:
        col_high = _find_col(df, ["high", "High", "HIGH", "h", "H", "high_price"])
    if col_low is None:
        col_low = _find_col(df, ["low", "Low", "LOW", "l", "L", "low_price"])
    if col_vol is None:
        col_vol = _find_col(df, ["volume", "Volume", "VOLUME", "vol", "Vol", "v", "V", "quote_volume", "base_volume"])

    n = len(df)

    # --- MODO B: PROXY events si NO hay close ---
    if col_close is None:
        proxy_cols = p.get("proxy_event_cols")
        if not proxy_cols:
            proxy_cols = [
                "feat_breakout_pressure",
                "feat_rebound_pressure",
                "bb_z",
                "vol_z",
                "macd_hist_slope",
                "adx",
                "chop",
            ]
        thr = float(p.get("proxy_abs_thr", 1.25))

        mask = np.zeros(n, dtype=bool)
        used = []
        for c in proxy_cols:
            if c in df.columns:
                x = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)
                mask |= np.isfinite(x) & (np.abs(x) > thr)
                used.append(c)

        if used:
            if not _WARNED["proxy"]:
                _WARNED["proxy"] = True
                print(f"[event_masks] OHLCV no disponible: usando eventos PROXY con cols={used} | abs_thr={thr}")
            return {"proxy": mask}

        # Si ni siquiera hay proxy cols, degradamos a todo False pero sin reventar
        if not _WARNED["no_close"]:
            _WARNED["no_close"] = True
            print(
                "[event_masks] WARNING: No se encontró 'close' (ni alias) y tampoco proxy_event_cols disponibles. "
                "Se desactivan eventos (todas las máscaras=False). "
                "SOLUCIÓN: agrega OHLCV al dataset o define event_params.proxy_event_cols."
            )
            print(f"[event_masks] columnas disponibles (muestra): {list(df.columns)[:30]}")
        z = np.zeros(n, dtype=bool)
        return {"breakout": z, "volburst": z, "sweep": z, "meanrev": z}

    # --- MODO A: OHLCV events ---
    close = pd.to_numeric(df[col_close], errors="coerce").to_numpy(np.float64)
    open_ = pd.to_numeric(df[col_open], errors="coerce").to_numpy(np.float64) if col_open in df.columns else close
    high  = pd.to_numeric(df[col_high], errors="coerce").to_numpy(np.float64) if col_high in df.columns else close
    low   = pd.to_numeric(df[col_low ], errors="coerce").to_numpy(np.float64) if col_low  in df.columns else close
    vol   = pd.to_numeric(df[col_vol ], errors="coerce").to_numpy(np.float64) if col_vol  in df.columns else np.zeros(n, dtype=np.float64)

    # --- parámetros de eventos ---
    n_break = int(p.get("breakout_lookback", 48))
    n_ema   = int(p.get("ema_span", 96))
    n_vol   = int(p.get("vol_lookback", 96))
    k_vol   = float(p.get("vol_z", 2.0))
    k_ret   = float(p.get("ret_z", 2.5))
    dev_k   = float(p.get("dev_z", 2.0))
    wick_th = float(p.get("wick_ratio", 0.60))

    # Retorno 1-bar (log)
    ret1 = np.zeros_like(close, dtype=np.float64)
    ret1[1:] = np.log(np.maximum(close[1:], 1e-12)) - np.log(np.maximum(close[:-1], 1e-12))

    # z-score retorno
    r = pd.Series(ret1)
    ret_std = r.rolling(n_vol, min_periods=n_vol).std().to_numpy(np.float64)
    ret_z = _safe_div(ret1, ret_std)

    # z-score volumen
    v = pd.Series(vol)
    v_mean = v.rolling(n_vol, min_periods=n_vol).mean().to_numpy(np.float64)
    v_std  = v.rolling(n_vol, min_periods=n_vol).std().to_numpy(np.float64)
    vol_z = _safe_z(vol, v_mean, v_std)

    # EMA + desviación
    ema = pd.Series(close).ewm(span=n_ema, adjust=False, min_periods=n_ema).mean().to_numpy(np.float64)
    dev = close - ema
    dev_std = pd.Series(dev).rolling(n_vol, min_periods=n_vol).std().to_numpy(np.float64)
    dev_z = _safe_div(dev, dev_std)

    # Breakout (ventana previa)
    hh = pd.Series(high).rolling(n_break, min_periods=n_break).max().shift(1).to_numpy(np.float64)
    ll = pd.Series(low ).rolling(n_break, min_periods=n_break).min().shift(1).to_numpy(np.float64)
    ok_hh = np.isfinite(hh)
    ok_ll = np.isfinite(ll)
    breakout_up = ok_hh & (close > hh) & (vol_z > k_vol)
    breakout_dn = ok_ll & (close < ll) & (vol_z > k_vol)

    # Vol burst / shock
    volburst = (np.abs(ret_z) > k_ret) & (vol_z > (0.5 * k_vol))

    # Sweep proxy (wick grande)
    rng = np.maximum(high - low, 1e-12)
    wick_up = (high - np.maximum(open_, close)) / rng
    wick_dn = (np.minimum(open_, close) - low) / rng
    sweep = (np.maximum(wick_up, wick_dn) > wick_th) & (vol_z > 0.0)

    # Mean reversion
    meanrev = (np.abs(dev_z) > dev_k)

    return {
        "breakout": breakout_up | breakout_dn,
        "volburst": volburst,
        "sweep": sweep,
        "meanrev": meanrev,
    }


class EventStreamWindowDataset:
    """
    Dataset por ventanas (lookback) con muestreo event/no-event.
    Devuelve: (Xw, y, yc, w)
      Xw: (lookback, F)
      y : float
      yc: 0 sl, 1 time, 2 tp si existe tbm_hit; si no 1
      w : sample_weight si existe, si no 1.0
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        lookback: int,
        steps_per_epoch: int,
        p_event: float,
        event_params: dict,
        seed: int = 12345,
        weight_col: str = "sample_weight",
        y_col: str = "y",
        tbm_col: str = "tbm_hit",
    ):
        self.df = df.reset_index(drop=True)
        self.feature_cols = list(feature_cols)
        self.lookback = int(lookback)
        self.steps_per_epoch = int(steps_per_epoch)
        self.p_event = float(p_event)
        self.rng = np.random.default_rng(int(seed))

        self.weight_col = weight_col
        self.y_col = y_col
        self.tbm_col = tbm_col

        if len(self.feature_cols) == 0:
            raise ValueError("feature_cols vacío: no hay features numéricas para entrenar.")
        if self.y_col not in self.df.columns:
            raise ValueError(f"Falta columna target '{self.y_col}' en el dataset.")

        # Features
        self.X = self.df[self.feature_cols].to_numpy(np.float32, copy=True)

        # Target
        self.y = self.df[self.y_col].to_numpy(np.float32, copy=True)

        # Peso
        if self.weight_col in self.df.columns:
            w = self.df[self.weight_col].to_numpy(np.float32, copy=True)
            self.w = np.where(np.isfinite(w), w, 0.0).astype(np.float32)
        else:
            self.w = np.ones(len(self.df), dtype=np.float32)

        # Clase
        if self.tbm_col in self.df.columns and self.df[self.tbm_col].notna().any():
            m = {"sl": 0, "time": 1, "tp": 2}
            yc = self.df[self.tbm_col].map(m).fillna(1).astype("int64").to_numpy()
        else:
            yc = np.ones(len(self.df), dtype=np.int64)
        self.yc = yc

        # Eventos (OHLCV o proxy; si no hay nada, devuelve todo False sin crash)
        masks = compute_event_masks(self.df, event_params)
        event_any = np.zeros(len(self.df), dtype=bool)
        for mm in masks.values():
            event_any |= np.asarray(mm, dtype=bool)

        # Índices válidos
        valid = np.arange(len(self.df))
        valid = valid[valid >= (self.lookback - 1)]

        # Evita muestras muertas
        alive = (self.w > 0.0)
        valid_alive = valid[alive[valid]]

        ev_idx = valid_alive[event_any[valid_alive]]
        ne_idx = valid_alive[~event_any[valid_alive]]

        # fallback robusto
        if len(ev_idx) < 2000:
            ev_idx = valid_alive
        if len(ne_idx) < 2000:
            ne_idx = valid_alive

        self.ev_idx = ev_idx
        self.ne_idx = ne_idx

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, i):
        if self.rng.random() < self.p_event:
            j = int(self.ev_idx[self.rng.integers(0, len(self.ev_idx))])
        else:
            j = int(self.ne_idx[self.rng.integers(0, len(self.ne_idx))])

        s = j - self.lookback + 1
        Xw = self.X[s:j + 1]
        y = float(self.y[j])
        yc = int(self.yc[j])
        w = float(self.w[j])
        return Xw, y, yc, w
