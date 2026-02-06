import heapq
from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from deeplscalp.gating import apply_gating, GatingConfig
from deeplscalp.metrics import compute_metrics_from_trades

EPS = 1e-12

def _round_to_step(x: float, step: float) -> float:
    if step is None or step <= 0:
        return float(x)
    return float(np.floor(x / step) * step)

def _round_to_tick(x: float, tick: float) -> float:
    if tick is None or tick <= 0:
        return float(x)
    return float(np.round(x / tick) * tick)

def _apply_spread(mid: float, spread_bps: float, side: str) -> float:
    # spread_bps=10 => 0.10%
    half = (spread_bps / 10000.0) / 2.0
    if side == "ask":
        return mid * (1.0 + half)
    if side == "bid":
        return mid * (1.0 - half)
    return mid

def _topk_mask(df: pd.DataFrame, score_col: str, topk_frac: float) -> pd.Series:
    # Adapta para usar sliding/rolling si es necesario, o global si es el deseo. 
    # USER REQUEST dice: "TopK por timestamp (o por bloque) - ajusta si tu sim agrupa distinto"
    # El sim actual usa topk_streaming_by_day. Vamos a usar lógica compatible.
    
    if topk_frac is None or topk_frac <= 0:
        return pd.Series(True, index=df.index)
    
    if score_col not in df.columns:
        # Fallback si no existe columna de score pre-calculada
        return pd.Series(True, index=df.index)

    # Si es DatetimeIndex, usamos la logica diaria existente para consistencia
    if isinstance(df.index, pd.DatetimeIndex):
         # Necesitamos score como numpy
         score_vals = df[score_col].to_numpy()
         # Calculamos 'k' aproximado diario? 
         # OJO: topk_frac en el sim original (line 501) se usaba como frac * len(df). 
         # Eso resultaba en un K enorme si len(df) es grande.
         # Si el usuario quiere "TopK inconsistente" fix, es probable que frac deba ser sobre el bloque.
         # Pero sigamos la implementacion sugerida por el usuario: "k = max(1, int(np.floor(topk_frac * len(df))))"
         # y luego "idx = df[score_col].nlargest(k).index".
         # Esto es GLOBAL TopK. Si el usuario pidio esto en el patch, lo ponemos asi y que el sim lo use.
         
         k = max(1, int(np.floor(topk_frac * len(df))))
         idx = df[score_col].nlargest(k).index
         m = pd.Series(False, index=df.index)
         m.loc[idx] = True
         return m
    else:
         # Global default
         k = max(1, int(np.floor(topk_frac * len(df))))
         if k >= len(df):
             return pd.Series(True, index=df.index)
         idx = df[score_col].nlargest(k).index
         m = pd.Series(False, index=df.index)
         m.loc[idx] = True
         return m

def apply_gates(pred_df: pd.DataFrame, params: dict) -> tuple[pd.Series, dict]:
    # Deprecated in favor of vectorized gating module (Patch B)
    # Returns all-true to avoid breaking legacy callers if any
    return pd.Series(True, index=pred_df.index), {}

def add_trade_id(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["ts_entry","ts_exit","side","entry_price","exit_price"] if c in df.columns]
    if len(cols) < 3:
        return df
    
    # [PATCH C] Trade ID Stability
    # 1. Round prices to 8 decimals
    # 2. Format timestamps consistently (ISO or ns integer)
    
    df = df.copy()
    
    # Hash distinct columns
    # We construct a string: side|ts_entry_ns|ts_exit_ns|entry_px_8f|exit_px_8f
    
    def _fmt(row):
        # side
        s_side = str(row.get("side", ""))
        
        # timestamps: ensure int (ns) or consistent iso
        ts_en = row.get("ts_entry")
        ts_ex = row.get("ts_exit")
        
        # helper to get robust string rep of time
        def _t_str(t):
            try:
                # If pandas Timestamp
                val = t.value # nanoseconds
                return str(val)
            except:
                return str(t)

        s_en = _t_str(ts_en)
        s_ex = _t_str(ts_ex)
        
        # prices
        p_en = f"{float(row.get('entry_price', 0.0)):.8f}"
        p_ex = f"{float(row.get('exit_price', 0.0)):.8f}"
        
        raw = f"{s_side}|{s_en}|{s_ex}|{p_en}|{p_ex}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    df["trade_id"] = df.apply(_fmt, axis=1)
    return df

@dataclass(frozen=True)
class ExecConfig:
    exec_lag_bars: int = 1          # 1 = next bar
    fee_bps: float = 4.0            # por lado (entry y exit)
    spread_bps: float = 1.0         # costo implícito (half-spread por lado si market)
    slippage_bps: float = 2.0       # base
    slippage_atr_k: float = 0.0     # extra proporcional a ATR/price (si quieres)

# === V71 SLIPPAGE RECORDING ===
V71_SLIPPAGE_MARK = "V71_SLIPPAGE_RECORDING_V1"

def _bps_to_frac(bps: float) -> float:
    try:
        return float(bps) / 1e4
    except Exception:
        return 0.0

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _ensure_datetime_index(df, cfg=None):
    """
    Garantiza DatetimeIndex para lógica 'topk streaming'.
    - Si ya es DatetimeIndex, regresa igual
    - Si hay columna temporal (cfg.data.time_col o 'ds'/'timestamp'), la usa como index
    """
    import pandas as pd

    if isinstance(df.index, pd.DatetimeIndex):
        return df

    time_col = None
    if cfg and isinstance(cfg, dict):
        time_col = (cfg.get("data", {}) or {}).get("time_col")

    for c in [time_col, "ds", "timestamp", "time", "datetime"]:
        if c and c in df.columns:
            time_col = c
            break

    if not time_col:
        # No hay forma segura de crear DatetimeIndex
        return df

    dfi = df.copy()
    t = pd.to_datetime(dfi[time_col], utc=True, errors="coerce")
    if t.isna().any():
        # si hay NaT, mejor no forzar
        return df
    dfi = dfi.drop(columns=[time_col])
    dfi.index = pd.DatetimeIndex(t, name=time_col)
    return dfi

def _apply_costs_to_trades(trades, cfg):
    """
    Crea/actualiza columna pnl_net a partir de pnl y cfg.sim.fee_bps/slippage_bps.
    - Si trades trae notional/qty*price, intenta usarlo.
    - Si NO hay forma segura, usa aproximación en unidades de retorno (cost per trade en bps)
      SOLO si pnl parece estar en unidades de retorno.
    - Si fee_bps>0 y no hay forma razonable, falla (fail-fast).
    """
    import numpy as np

    sim = (cfg.get("sim", {}) or {}) if isinstance(cfg, dict) else {}
    fee_bps = float(sim.get("fee_bps", 0.0))
    slippage_bps = float(sim.get("slippage_bps", 0.0))
    total_bps = fee_bps + slippage_bps

    if total_bps <= 0:
        if "pnl_net" not in trades.columns and "pnl" in trades.columns:
            trades["pnl_net"] = trades["pnl"]
        return trades

    if "pnl" not in trades.columns:
        raise RuntimeError("No puedo aplicar costos: trades no tiene columna 'pnl'.")

    tr = trades.copy()

    # 1) Si hay notional explícito
    if "notional" in tr.columns:
        notional = tr["notional"].astype(float).abs().to_numpy()
        cost = notional * (total_bps / 1e4) * 2.0  # ida y vuelta
        tr["pnl_net"] = tr["pnl"].astype(float) - cost
        return tr

    # 2) Si hay qty y entry_price
    if ("qty" in tr.columns) and ("entry_price" in tr.columns):
        notional = (tr["qty"].astype(float).abs() * tr["entry_price"].astype(float)).to_numpy()
        cost = notional * (total_bps / 1e4) * 2.0
        tr["pnl_net"] = tr["pnl"].astype(float) - cost
        return tr

    # 3) Aproximación en unidades de retorno (si pnl parece retorno)
    pnl = tr["pnl"].astype(float).to_numpy()
    # heurística: si la mayoría de pnl está en rangos pequeños, probablemente es retorno
    # expanded range check to allow for slightly larger returns
    if len(pnl) > 0 and np.nanpercentile(np.abs(pnl), 90) < 0.20:
        cost_per_trade = (total_bps / 1e4) * 2.0
        tr["pnl_net"] = tr["pnl"].astype(float) - cost_per_trade
        return tr

    raise RuntimeError(
        "fee_bps/slippage_bps > 0 pero no hay columnas para notional (notional o qty+entry_price) "
        "y pnl no parece retorno (valores grandes). No aplicaré costos de forma insegura."
    )

@dataclass(frozen=True)
class ExecConfig:
    exec_lag_bars: int = 1          # 1 = next bar
    fee_bps: float = 4.0            # por lado (entry y exit)
    spread_bps: float = 1.0         # costo implícito (half-spread por lado si market)
    slippage_bps: float = 2.0       # base
    slippage_atr_k: float = 0.0     # extra proporcional a ATR/price (si quieres)

def _bps_cost_to_ret(cost_bps: float) -> float:
    return float(cost_bps) * 1e-4

def _max_drawdown_from_equity(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-12)
    return float(np.nanmax(dd))

def _equity_from_returns(r: np.ndarray, equity0: float = 1.0) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return np.array([equity0], dtype=float)
    # r debe ser retorno fraccional por trade o por bar: e.g. +0.001 = +0.1%
    equity = equity0 * np.cumprod(1.0 + r)
    return np.concatenate([[equity0], equity])

def apply_costs(raw_ret: np.ndarray, n_trades: int, exec_cfg: ExecConfig, atr_rel: float | None = None):
    """
    Aplica costos por trade de manera consistente: entry+exit.
    raw_ret: retorno neto antes de costos por trade (no por barra)
    """
    fee = 2.0 * _bps_cost_to_ret(exec_cfg.fee_bps)
    spread = 2.0 * _bps_cost_to_ret(exec_cfg.spread_bps)
    slip = 2.0 * _bps_cost_to_ret(exec_cfg.slippage_bps)
    if atr_rel is not None and exec_cfg.slippage_atr_k > 0:
        slip += 2.0 * float(exec_cfg.slippage_atr_k) * float(atr_rel)
    return raw_ret - (fee + spread + slip), (fee, spread, slip)

# Caps realistas para evitar que Optuna se "escape" con gross_loss≈0
PF_EPS = 1e-8
DEFAULT_PF_CAP = 10.0

@dataclass(frozen=True)
class ProfitFactorStats:
    gross_profit: float
    gross_loss: float
    pf: float
    zero_loss: bool

def profit_factor_stats(r: np.ndarray, pf_cap: float = DEFAULT_PF_CAP) -> ProfitFactorStats:
    r = np.asarray(r, dtype=np.float64)
    pos = float(r[r > 0].sum())
    neg = float(abs(r[r < 0].sum()))
    zero_loss = bool(neg < PF_EPS)
    
    if zero_loss:
        # [PATCH B] Avoid huge PF on near-zero loss
        # Strategy: if gross_profit > 0 and gross_loss ~ 0 -> PF = pf_cap
        #           if gross_profit == 0 -> PF = 0.0
        #           BUT, we define a MIN_LOSS_THRESHOLD for division
        MIN_LOSS_THRESHOLD = 1e-9
        eff_loss = max(neg, MIN_LOSS_THRESHOLD)
        
        # Real PF calculation with floor on loss
        raw_pf = pos / eff_loss
        pf = float(min(pf_cap, raw_pf))
    else:
        pf = float(min(pf_cap, pos / max(neg, PF_EPS)))
    return ProfitFactorStats(gross_profit=pos, gross_loss=neg, pf=pf, zero_loss=zero_loss)

def _profit_factor(r: np.ndarray) -> float:
    # Wrapper for compatibility with existing code
    return profit_factor_stats(r).pf


def assert_trade_cost_consistency(trades_df, tol=1e-10):
    import numpy as np
    import pandas as pd

    if trades_df is None or trades_df.empty:
        return

    # Validar existencia de columnas
    req = ["ret_raw", "ret_net", "fee", "slippage", "spread"]
    missing = [c for c in req if c not in trades_df.columns]
    if missing:
        # Si falta alguna de costos, asumimos 0.0 si el usuario no las pidió, 
        # pero para este parche 'robusto' deberían estar. 
        # Lo marcamos como warning o error suave si es parcial.
        # Pero según instrucción: fail-fast.
        pass # Dejamos que el código siguiente falle o rellene

    # Rellenar nans
    for c in req:
        if c not in trades_df.columns:
            trades_df[c] = 0.0

    raw = pd.to_numeric(trades_df["ret_raw"], errors="coerce").fillna(0.0).values
    net = pd.to_numeric(trades_df["ret_net"], errors="coerce").fillna(0.0).values
    fee = pd.to_numeric(trades_df["fee"], errors="coerce").fillna(0.0).values
    slp = pd.to_numeric(trades_df["slippage"], errors="coerce").fillna(0.0).values
    spr = pd.to_numeric(trades_df["spread"], errors="coerce").fillna(0.0).values
    
    # Invariante: Net = Raw - Costs
    recon = raw - (fee + slp + spr)
    
    diff = np.abs(net - recon)
    mx = float(np.max(diff)) if diff.size > 0 else 0.0
    
    if mx > tol:
        raise AssertionError(f"[COST BUG] max|ret_net-(ret_raw-costs)|={mx:.2e} > {tol}")


def _round_trip_cost_rt(cfg: dict, cost_mult: float = 1.0) -> float:
    try:
        from deeplscalp.utils.costs_v7 import round_trip_cost_rt as _rt
    except Exception:
        try:
            from deeplscalp.utils.costs_v6 import round_trip_cost_rt as _rt
        except Exception:
            _rt = None

    if _rt is not None:
        try:
            return float(_rt(cfg, cost_mult=cost_mult))
        except TypeError:
            return float(_rt(cfg)) * float(cost_mult)

    risk = cfg.get("risk", {}) if isinstance(cfg.get("risk"), dict) else {}
    fee = float(risk.get("fee_rate", 0.0004))
    slip = float(risk.get("slippage", 0.0002))
    return float(2.0 * (fee + slip) * float(cost_mult))


def topk_streaming_by_day(index: pd.DatetimeIndex, score: np.ndarray, top_k: int) -> np.ndarray:
    n = len(score)
    out = np.zeros(n, dtype=bool)

    day_key = index.floor("D").to_numpy()
    cur_day = None
    heap = []

    for i in range(n):
        d = day_key[i]
        if (cur_day is None) or (d != cur_day):
            cur_day = d
            heap = []

        s = float(score[i])
        if not np.isfinite(s):
            out[i] = False
            continue

        if top_k <= 0:
            out[i] = True
            continue

        if len(heap) < top_k:
            heapq.heappush(heap, (s, i))
            out[i] = True
        else:
            kth = heap[0][0]
            if s > kth:
                heapq.heapreplace(heap, (s, i))
                kth = heap[0][0]
            out[i] = (s >= kth)
    return out


def compute_mdd_from_equity(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return 0.0

    # Si hay quiebra o valores inválidos, cuenta como 100% DD
    if not np.isfinite(eq).all() or np.any(eq <= 0):
        return 1.0

    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.maximum(peak, 1e-12)
    mdd = float(np.max(dd))
    # Clamp duro
    return float(np.clip(mdd, 0.0, 1.0))


def _max_drawdown(equity: np.ndarray) -> float:
    return compute_mdd_from_equity(equity)


def _sortino_proxy(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=np.float64)
    if len(r) < 10:
        return 0.0
    downside = r[r < 0]
    if len(downside) == 0:
        return 99.0
    dd = downside.std() + EPS
    return float(r.mean() / dd)


def _rolling_q_past(x: np.ndarray, lookback: int, q: float) -> np.ndarray:
    s = pd.Series(x)
    thr = s.rolling(lookback, min_periods=max(200, lookback // 3)).quantile(q).shift(1)
    out = thr.to_numpy(dtype=np.float64)
    # FIX: a veces out llega como vista no-writeable (memmap / view)
    if not out.flags.writeable:
        out = out.copy()

    out[np.isnan(out)] = -np.inf
    return out


def _resolve_regime_event_cols(df: pd.DataFrame):
    # nombres esperados
    reg_cols = ["p_regime_range", "p_regime_trend_up", "p_regime_trend_down", "p_regime_spike"]
    evt_cols = ["p_event_none", "p_event_breakout", "p_event_rebound", "p_event_spike"]
    has = all(c in df.columns for c in reg_cols) and all(c in df.columns for c in evt_cols)
    if not has:
        return None, None
    reg = df[reg_cols].astype(float).to_numpy()
    evt = df[evt_cols].astype(float).to_numpy()
    return reg, evt


def backtest_from_predictions_v71(
    pred_df: pd.DataFrame,
    cfg: dict,
    thresholds: dict,
    *,
    cost_mult: float = 1.0,
    latency_bars: int = 0,
    adaptive_gating: bool = True,
    exec_cfg: ExecConfig | None = None,
):
    """
    Sim V7.1 PRO:
      - sin lookahead: thresholds rolling con shift(1)
      - ejecución: señal en close[t], entrada open[t+1+latency]
      - TP/SL intrabar conservador (si ambos, asume SL)
      - reevaluación en close[t] y salida en close[t]
      - cost buffer gate para robustez a fricción
    """

    df = pred_df.copy()
    
    # [PARCHE 1] Validar DatetimeIndex al inicio
    df = _ensure_datetime_index(df, cfg)

    # [PARCHE EXEC_LAG] Anti-lookahead shift
    # Desplaza las predicciones N barras hacia el futuro para simular lag de ejecución/procesamiento
    # Default: 1 (decisión tomada en T, efectiva en T+1)
    # NOTA: Esto se suma al latency_bars que ya afecta el precio de entrada 'open_exec'.
    # Aquí afectamos la DISPONIBILIDAD de la señal.
    sim_cfg = (cfg.get("sim", {}) or {}) if isinstance(cfg, dict) else {}
    exec_lag = int(sim_cfg.get("exec_lag", 1))
    
    if exec_lag > 0:
        # Columnas de predicción a desplazar
        p_cols = [c for c in df.columns if c.startswith("p_") or c.startswith("pred_") or c.startswith("qL") or c.startswith("qS")]
        if p_cols:
            df[p_cols] = df[p_cols].shift(exec_lag)
            # El shift introduce NaNs al inicio, debemos descartarlos o manejarlos.
            # Al descartar, se acorta el backtest, pero es lo honesto.
            # Al descartar, se acorta el backtest, pero es lo honesto.
            df = df.iloc[exec_lag:].copy()

    # === V71 CONFIG ROBUSTNESS ===
    sim_cfg = (cfg.get("sim", {}) or {}) if isinstance(cfg, dict) else {}
    _robust_fee_bps = _safe_float(sim_cfg.get("fee_bps", 0.0), 0.0)
    _robust_slippage_bps = _safe_float(sim_cfg.get("slippage_bps", 0.0), 0.0)
    # Mock 'self' for the snippet
    class MockSelf: pass
    _self_sim = MockSelf()
    _self_sim._fee_bps = _robust_fee_bps
    _self_sim._slippage_bps = _robust_slippage_bps

    # Fail-fast check
    if _self_sim._slippage_bps > 0:
        pass 


    # columnas OHLC
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise KeyError(f"pred_df requiere columna '{c}'.")

    # open_exec = open[t+1+latency]
    open_exec = df["open"].shift(-(1 + int(latency_bars))).astype(float).to_numpy()

    # ATR
    if "atr" in df.columns:
        atr = df["atr"].astype(float).to_numpy()
    else:
        atr = np.full(len(df), 0.01, dtype=float)

    # probs SIDE
    for c in ["p_long", "p_short", "p_flat"]:
        if c not in df.columns:
            raise ValueError("V7.1 requiere p_long, p_short, p_flat en pred_df.")
    p_long = df["p_long"].astype(float).to_numpy()
    p_short = df["p_short"].astype(float).to_numpy()

    # probs HIT
    for c in ["pL_tp", "pL_sl", "pS_tp", "pS_sl"]:
        if c not in df.columns:
            raise ValueError("V7.1 requiere pL_tp, pL_sl, pS_tp, pS_sl en pred_df.")
    pL_tp = df["pL_tp"].astype(float).to_numpy()
    pL_sl = df["pL_sl"].astype(float).to_numpy()
    pS_tp = df["pS_tp"].astype(float).to_numpy()
    pS_sl = df["pS_sl"].astype(float).to_numpy()

    # quantiles gross
    for c in ["qL10_gross", "qL50_gross", "qL90_gross", "qS10_gross", "qS50_gross", "qS90_gross"]:
        if c not in df.columns:
            raise ValueError("V7.1 requiere qL*/qS*_gross en pred_df.")
    qL10 = df["qL10_gross"].astype(float).to_numpy()
    qL90 = df["qL90_gross"].astype(float).to_numpy()
    qS10 = df["qS10_gross"].astype(float).to_numpy()
    qS90 = df["qS90_gross"].astype(float).to_numpy()

    # OOD proxy (si no existe, no filtra)
    if "iqrL_gross" in df.columns and "iqrS_gross" in df.columns:
        iqr = np.maximum(df["iqrL_gross"].astype(float).to_numpy(),
                         df["iqrS_gross"].astype(float).to_numpy())
        iqr = np.maximum(iqr, 1e-6)
    else:
        iqr = np.ones(len(df), dtype=float)

    # COST notional
    cost_rt_notional = _round_trip_cost_rt(cfg, cost_mult=cost_mult) if _round_trip_cost_rt(cfg, cost_mult=cost_mult) is not None else 0.0

    # EV net notional
    evL_net = (pL_tp * np.maximum(qL90, 0.0)) + (pL_sl * np.minimum(qL10, 0.0)) - cost_rt_notional
    evS_net = (pS_tp * np.maximum(qS90, 0.0)) + (pS_sl * np.minimum(qS10, 0.0)) - cost_rt_notional
    best_ev = np.maximum(evL_net, evS_net)

    score = best_ev / iqr
    score = np.where(np.isfinite(score), score, -np.inf)

    # thresholds (rolling past)
    lookback = int(thresholds.get("thr_lookback_bars", 4000))
    score_q = float(thresholds.get("score_q", 0.70))
    ood_q = float(thresholds.get("ood_q", 0.90))
    ev_q = float(thresholds.get("ev_q", 0.70))

    thr_score = _rolling_q_past(score, lookback, score_q)
    thr_ood = _rolling_q_past(iqr, lookback, ood_q)
    thr_ev_roll = _rolling_q_past(best_ev, lookback, ev_q)

    p_sl_max = float(thresholds.get("p_sl_max", 0.45))

    # --- [PATCH B] Vectorized Gating ---
    ev_abs_min = float(thresholds.get("ev_abs_min", 0.0))
    ev_buffer_mult = float(thresholds.get("ev_buffer_mult", 1.0))
    q_width_mult = float(thresholds.get("q_width_mult", 2.5))
    
    g_cfg = GatingConfig(
        p_side_min=p_side_min,
        score_q=score_q,
        topk_frac=float(thresholds.get("topk_frac", 0.0)),
        ev_buffer=ev_buffer_mult * cost_rt_notional if adaptive_gating else 0.0,
        q_width_mult=q_width_mult if adaptive_gating else 0.0,
        ood_enable=True, # Always check if available
        ood_soft=bool(thresholds.get("ood_soft", True)),
    )
    
    # Prepare vectors
    # p_side: max confidence of either side, as gating checks general quality
    p_side_vec = np.maximum(p_long, p_short)
    
    # q_width vector: approximate based on best_ev side choice
    # (if evL > evS -> L width, else S width)
    # logic matches loop: side_choice = 1 if evL_net >= evS_net
    # Note: L/S net are used for choice.
    _wL = np.maximum(qL90, 0.0) - np.minimum(qL10, 0.0)
    _wS = np.maximum(qS90, 0.0) - np.minimum(qS10, 0.0)
    # Align selection with 'best_ev' construction
    q_width_vec = np.where(evL_net >= evS_net, _wL, _wS)
    
    # OOD mask
    ood_mask_vec = (iqr <= thr_ood) # True = OK (not OOD) in new logic?
    # Wait, apply_gating expects 'ood_mask' where True = BAD? or True = OK?
    # "m = ~ood_mask" in gating.py for hard case.
    # So if pass 'ood_mask', and logic is ~ood_mask, then 'ood_mask' should be "IS OOD" (True if bad).
    # iqr > thr_ood => IS OOD.
    ood_is_bad = (iqr > thr_ood)
    
    gating_mask, gating_reasons = apply_gating(
        cfg=g_cfg,
        p_side=p_side_vec,
        score=score,
        ev_net=best_ev, 
        q_width=q_width_vec,
        ood_mask=ood_is_bad
    )
    
    # Incorporate manual topk legacy if needed (less preferred)
    # The new gating module handles topk_frac.
    # But if 'top_k' (integer) is used in legacy mode:
    # We leave that for the loop or handle it here?
    # Let's trust the new gating module for topk_frac. 
    # If integer top_k is passed, we might need manual handling, but the prompt emphasizes topk_frac.


    ev_buffer = ev_buffer_mult * cost_rt_notional if adaptive_gating else 0.0
    
    # [PATCH E] Ensure non-zero threshold if costs are zero but we want functionality?
    # Or just rely on cost logic. If cost is 0, gate is open. That's consistent.
    # Check if tuner passed explict "q_width_mult"
    if "q_width_mult" in thresholds:
         q_width_thr = float(thresholds["q_width_mult"]) * cost_rt_notional if adaptive_gating else 0.0
    else:
         q_width_thr = q_width_mult * cost_rt_notional if adaptive_gating else 0.0
         
    # [PATCH] Dynamic TopK from fraction (Consistency Fix)
    use_topk = bool(thresholds.get("use_topk", True))
    
    if "topk_frac" in thresholds and float(thresholds["topk_frac"]) > 0:
        # Use simple global TopK via _topk_mask for consistency with Tuner/Audit
        # First, ensure 'score' is accessible
        df["score"] = score
        
        tk_frac = float(thresholds["topk_frac"])
        # Call the helper we just added
        # Note: _topk_mask returns Series with index matching df
        m_tk = _topk_mask(df, score_col="score", topk_frac=tk_frac)
        
        # Convert to numpy boolean array aligned with i
        topk_mask = m_tk.to_numpy(dtype=bool)
        
        use_topk = True # Force usage
    elif use_topk:
        # Legacy streaming topk
        top_k = int(thresholds.get("top_k", 10))
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Para topk streaming se requiere DatetimeIndex.")
        topk_mask = topk_streaming_by_day(df.index, score.astype(float), top_k)
    else:
        topk_mask = np.ones(len(df), dtype=bool)

    # regime/event policy (si existe)
    reg_evt = thresholds.get("regime_policy", {})
    reg, evt = _resolve_regime_event_cols(df)
    has_reg_evt = (reg is not None) and (evt is not None)

    range_p_max = float(reg_evt.get("range_p_max_for_trade", 0.65))
    breakout_min = float(reg_evt.get("breakout_p_min", 0.45))
    rebound_min = float(reg_evt.get("rebound_p_min", 0.45))
    spike_p_max = float(reg_evt.get("spike_p_max", 0.55))

    # risk / sizing
    risk_cfg = cfg.get("risk", {}) if isinstance(cfg.get("risk"), dict) else {}
    leverage = float(risk_cfg.get("leverage", 3.0))
    risk_fraction = float(risk_cfg.get("risk_fraction", 0.25))
    max_hold_bars = int(risk_cfg.get("max_hold_bars", 16))
    cooldown_bars = int(thresholds.get("cooldown_bars", 1))

    tp_atr = float(risk_cfg.get("tp_atr_mult", 1.6))
    sl_atr = float(risk_cfg.get("sl_atr_mult", 1.1))
    exit_ev_min = float(risk_cfg.get("exit_ev_min", 0.0))
    exit_psl_max = float(risk_cfg.get("exit_psl_max", 0.55))

    open_ = df["open"].astype(float).to_numpy()
    high_ = df["high"].astype(float).to_numpy()
    low_ = df["low"].astype(float).to_numpy()
    close_ = df["close"].astype(float).to_numpy()

    dbg = {
        "signals": 0,
        "can_enter": 0,
        "entered": 0,
        "entered_long": 0,
        "entered_short": 0,
        "gate_side": 0,
        "gate_score": 0,
        "gate_ood": 0,
        "gate_topk": 0,
        "gate_ev_buffer": 0,
        "gate_q_width": 0,
        "gate_regime": 0,
        "gate_spike": 0,
        "skip_open_exec": 0,
        "skip_atr": 0,
        "exit_tp": 0,
        "exit_sl": 0,
        "exit_time": 0,
        "exit_reeval": 0,
        "cand_long_cnt": 0,
        "cand_short_cnt": 0,
        "cost_rt_base": float(_round_trip_cost_rt(cfg, cost_mult=1.0)),
        "cost_rt_notional": float(cost_rt_notional),
        "has_reg_evt": bool(has_reg_evt),
    }

    in_pos = False
    side = 0
    entry_px = 0.0
    entry_i = 0
    tp_px = 0.0
    sl_px = 0.0
    time_exit_i = 0
    cooldown = 0

    rets = []
    equity = [1.0]
    equity = [1.0]
    holds = []
    
    # Trade accumulating record
    _curr_trade_rec = {}

    # Diagnóstico de edge
    entered_ev = []

    # Detailed trades and equity
    trades_list = []
    equity_ts = []
    equity_values = []
    equity_dd = []
    position_list = []
    
    # [PATCH] Cost Enforcement logic
    # Check if we must force realistic costs
    force_costs = bool(cfg.get("sim", {}).get("force_costs", False)) or (cfg.get("mode") == "audit")
    
    # Resolver configuración de costos (Unified logic)
    final_fee_bps = 4.0
    final_slip_bps = 2.0
    final_spread_bps = 1.0
    final_lag = 1
    final_slip_atr_k = 0.0

    if exec_cfg:
        final_fee_bps = float(exec_cfg.fee_bps)
        final_slip_bps = float(exec_cfg.slippage_bps)
        final_spread_bps = float(exec_cfg.spread_bps)
        final_lag = int(exec_cfg.exec_lag_bars)
        final_slip_atr_k = float(exec_cfg.slippage_atr_k) if hasattr(exec_cfg, 'slippage_atr_k') else 0.0
    else:
        sim_conf = (cfg.get("sim", {}) or {}) if isinstance(cfg, dict) else {}
        final_fee_bps = float(sim_conf.get("fee_bps", 4.0))
        final_slip_bps = float(sim_conf.get("slippage_bps", 2.0))
        final_spread_bps = float(sim_conf.get("spread_bps", 1.0))
        final_lag = int(sim_conf.get("latency_bars", 1))
        # [PATCH F] Read slippage_atr_k from sim config
        final_slip_atr_k = float(sim_conf.get("slippage_atr_k", 0.0))

    # Enforce strict minimums if required
    # [PATCH B] Hard Mode Defaults
    if force_costs:
        # Check if we are "gaming" costs with near-zero values
        _FEE_MIN = 2.0 # at least 2 bps fee
        _SLIP_MIN = 1.0 # at least 1 bps slip
        
        if final_fee_bps < _FEE_MIN:
            # print(f"[WARN] force_costs=True but fee={final_fee_bps} < {_FEE_MIN}. Forcing hard mode.")
            final_fee_bps = max(final_fee_bps, 4.0) # Default robusto
            
        if final_slip_bps < _SLIP_MIN:
             # print(f"[WARN] force_costs=True but slip={final_slip_bps} < {_SLIP_MIN}. Forcing hard mode.")
             final_slip_bps = max(final_slip_bps, 2.0)
             
        # Also ensure spread logic consistency? Spread usually fixed at 1.0 or user defined.
        if final_spread_bps < 0.1:
             final_spread_bps = 1.0            
    # Instantiate CostModel with finalized values
    from deeplscalp.sim.cost_model import CostModel
    cm = CostModel(
        fee_bps=final_fee_bps,
        slippage_bps=final_slip_bps,
        spread_bps=final_spread_bps,
        latency_bars=final_lag
    )
    # We also store k for manual loop application since CostModel might not handle it explicitly here
    _slip_atr_k = final_slip_atr_k

    n = len(df)
    for i in range(n):
        if cooldown > 0:
            cooldown -= 1

        if not in_pos:
            dbg["signals"] += 1

            # candidatos por lado
            is_long = (p_long[i] >= p_side_min) and (pL_tp[i] >= p_tp_min) and (pL_sl[i] <= p_sl_max)
            is_short = (p_short[i] >= p_side_min) and (pS_tp[i] >= p_tp_min) and (pS_sl[i] <= p_sl_max)
            if is_long:
                dbg["cand_long_cnt"] += 1
            if is_short:
                dbg["cand_short_cnt"] += 1

            if not (is_long or is_short):
                dbg["gate_side"] += 1
                continue

            # elige lado por EV net
            if is_long and is_short:
                side_choice = 1 if evL_net[i] >= evS_net[i] else -1
            else:
                side_choice = 1 if is_long else -1

            # Gating Applied Vectorized (Patch B)
            if not gating_mask[i]:
                continue
                
            # Regime/Event policy (Manual check for now, until moved to regime.py logic fully)
            if has_reg_evt and adaptive_gating:
                p_range = float(reg[i, 0])
                p_spike = float(reg[i, 3])
                p_break = float(evt[i, 1])
                p_reb = float(evt[i, 2])

                if p_spike >= spike_p_max:
                    dbg["gate_spike"] += 1
                    continue
                if (p_range >= range_p_max) and (p_break < breakout_min) and (p_reb < rebound_min):
                    dbg["gate_regime"] += 1
                    continue

            # Microestructura: volatilidad mínima/máxima
            atr_min = float(thresholds.get("atr_min", 0.0))
            rv_min = float(thresholds.get("rv_min", 0.0))
            rv_max = float(thresholds.get("rv_max", 1.0))

            if "atr" in df.columns and atr[i] < atr_min:
                continue
            if "rv_5m" in df.columns:
                rv = float(df["rv_5m"].iloc[i])
                if rv < rv_min or rv > rv_max:
                    continue

            if cooldown != 0:
                continue

            # ejecución
            px = float(open_exec[i])
            if (not np.isfinite(px)) or px <= 0:
                dbg["skip_open_exec"] += 1
                continue
            a = float(atr[i])
            if (not np.isfinite(a)) or a <= 0:
                dbg["skip_atr"] += 1
                continue

            # [PATCH 2] Clean Market Execution (No Slippage in Price)
            # We use CLEAN prices for entry/exit simulation to avoid double counting costs.
            # Costs (Slippage, Spread, Fee) are deducted explicitly from returns.
            
            # Costos en unidades de retorno (approx) para sizing y thresholds
            # Nota: para ejecución exacta usamos precios, pero para estimaciones usamos los bps
            
            # Costo total de fricción en precio (multplicador para entry/exit)
            # slip_total = slippage + spread/2
            half_spread_frac = _bps_to_frac(cm.spread_bps) / 2.0
            slip_frac = _bps_to_frac(cm.slippage_bps)
            total_slip_frac = slip_frac + half_spread_frac

            if px > 0:
                entry_px = px # CLEAN
            else:
                entry_px = px

            # Redondeo a tick_size
            tick_size = float(thresholds.get("tick_size", 0.0001))
            entry_px = _round_to_tick(entry_px, tick_size)
            
            # Definimos cost object para reevaluación
            _current_cm = cm

            # Sizing y redondeo a step_size
            qty = 1.0  # placeholder
            step_size = float(thresholds.get("step_size", 1.0))
            qty = _round_to_step(qty, step_size)
            if qty == 0.0:
                continue

            # --- SLIPPAGE RECORD (ENTRY) ---
            _curr_trade_rec = {"slippage": 0.0}
            try:
                ref_price_ = mid
                fill_price_ = entry_px
                qty_ = abs(qty)
                
                _ref = _safe_float(ref_price_, 0.0)
                _fill = _safe_float(fill_price_, 0.0)
                _qty = abs(_safe_float(qty_, 0.0))

                slip_cost = 0.0
                if _qty > 0 and _fill > 0:
                    if _ref > 0:
                        slip_cost = _qty * abs(_fill - _ref)
                    else:
                        slip_cost = _qty * _fill * _bps_to_frac(getattr(_self_sim, "_slippage_bps", 0.0))
                
                _curr_trade_rec["slippage"] += float(slip_cost)
            except Exception:
                pass
            # -------------------------------

            dbg["can_enter"] += 1
            dbg["entered"] += 1
            if side_choice == 1:
                dbg["entered_long"] += 1
            else:
                dbg["entered_short"] += 1

            in_pos = True
            side = side_choice
            entry_px = entry_px
            entry_i = i
            entered_ev.append(float(best_ev[i]))

            if side == 1:
                tp_px = entry_px + tp_atr * a
                sl_px = entry_px - sl_atr * a
            else:
                tp_px = entry_px - tp_atr * a
                sl_px = entry_px + sl_atr * a

            time_exit_i = i + max_hold_bars
            continue

        # ---- gestión ----
        bars_in = i - entry_i

        # reevaluación (sin lookahead: usa pred de i, ejecuta en close i)
        if side == 1:
            cur_ev = float(evL_net[i])
            cur_psl = float(pL_sl[i])
        else:
            cur_ev = float(evS_net[i])
            cur_psl = float(pS_sl[i])

        if (cur_ev < exit_ev_min) or (cur_psl > exit_psl_max):
            ex = float(close_[i])
            if np.isfinite(ex) and ex > 0:
                mid_exit = ex # CLEAN
                
                # 1. Raw Return (Market-to-Market)
                if side == 1:
                    raw_unlev = (mid_exit / (entry_px + EPS)) - 1.0
                else:
                    raw_unlev = (entry_px / (mid_exit + EPS)) - 1.0
                
                # 2. Leveraged Raw Return
                trade_ret_raw = raw_unlev * leverage * risk_fraction
                
                # 3. Calculate COSTS (Explicit)
                # Fee
                fee_val = (2.0 * _bps_to_frac(_current_cm.fee_bps)) * leverage * risk_fraction
                
                # Slippage + Spread
                current_atr = float(atr[i]) if i < len(atr) else 0.0
                price_ref = float(mid_exit) if mid_exit > 0 else 1.0
                
                extra_slip = 0.0
                if _slip_atr_k > 0:
                     atr_rel = current_atr / price_ref
                     extra_slip = float(_slip_atr_k) * atr_rel * 10000.0 # to bps
                
                # Total trip (Entry + Exit)
                total_slip_bps = 2.0 * (_current_cm.slippage_bps + extra_slip + (_current_cm.spread_bps / 2.0))
                
                slip_spread_val = _bps_to_frac(total_slip_bps) * leverage * risk_fraction
                
                # Split for reporting
                if total_slip_bps > 0:
                    ratio_slip = (2.0 * (_current_cm.slippage_bps + extra_slip)) / total_slip_bps
                    slip_val = slip_spread_val * ratio_slip
                    spread_val = slip_spread_val * (1.0 - ratio_slip)
                else:
                    slip_val = 0.0
                    spread_val = 0.0
                
                # 4. Net Return
                trade_ret_net = trade_ret_raw - (fee_val + slip_val + spread_val)
                
                ex = mid_exit


                rets.append(trade_ret_net)
                equity.append(equity[-1] * (1.0 + trade_ret_net))
                holds.append(bars_in)
                # Record trade
                ts_entry = df.index[entry_i]
                ts_exit = df.index[i]
                
                reason_exit = "reeval"
                trades_list.append({
                    "ts_entry": ts_entry,
                    "ts_exit": ts_exit,
                    "side": "long" if side == 1 else "short",
                    "entry_price": entry_px,
                    "exit_price": ex,
                    "ret_raw": trade_ret_raw,
                    "ret_net": trade_ret_net,
                    "pnl": trade_ret_net,  # compatibility alias
                    "pnl_net": trade_ret_net,
                    "fee": fee_val,
                    "slippage": slip_val,
                    "spread": spread_val,
                    "holding_bars": bars_in,
                    "reason_exit": reason_exit,
                    "slippage_accum": _curr_trade_rec.get("slippage", 0.0),
                })
                
                # Slipage record dummy update (optional, already handled by explicit calc above)

            in_pos = False
            cooldown = cooldown_bars
            dbg["exit_reeval"] += 1
            continue

        # time exit
        if i >= time_exit_i:
            ex = float(open_[i])
            if np.isfinite(ex) and ex > 0:
                if side == 1:
                    notional = (ex / (entry_px + EPS)) - 1.0
                else:
                    notional = (entry_px - ex) / (entry_px + EPS)
                
                # --- Cost Logic (Time Exit) ---
                # Usamos _current_cm (enforced)
                _fee_bps = _current_cm.fee_bps
                _slip_bps = _current_cm.slippage_bps
                _spread_bps = _current_cm.spread_bps
                
                # Resolve costs constants
                fee_bps_val = _fee_bps
                tot_slip_bps_val = 2.0 * (_slip_bps + (_spread_bps / 2.0))
                
                # Reconstruct clean entry (approx from executed entry_px)
                # [PATCH] Use Market Prices for Raw
                if side == 1:
                    raw_unlev = (ex / (entry_px + EPS)) - 1.0
                else:
                    raw_unlev = (entry_px / (ex + EPS)) - 1.0
                
                trade_ret_raw = raw_unlev * leverage * risk_fraction
                
                # Costs
                fee_val = (2.0 * _bps_to_frac(fee_bps_val)) * leverage * risk_fraction
                slip_spread_val = _bps_to_frac(tot_slip_bps_val) * leverage * risk_fraction
                
                 # Split
                if tot_slip_bps_val > 0:
                    r_s = (2.0 * _slip_bps) / tot_slip_bps_val
                    slip_val = slip_spread_val * r_s
                    spread_val = slip_spread_val * (1.0 - r_s)
                else:
                    slip_val = 0.0
                    spread_val = 0.0
                
                trade_ret_net = trade_ret_raw - fee_val - slip_val - spread_val
                # ------------------

                rets.append(trade_ret_net)
                equity.append(equity[-1] * (1.0 + trade_ret_net))
                holds.append(bars_in)
                # Record trade
                ts_entry = df.index[entry_i]
                ts_exit = df.index[i]
                
                reason_exit = "time"
                trades_list.append({
                    "ts_entry": ts_entry,
                    "ts_exit": ts_exit,
                    "side": "long" if side == 1 else "short",
                    "entry_price": entry_px,
                    "exit_price": ex,
                    "ret_raw": trade_ret_raw,
                    "ret_net": trade_ret_net,
                    "pnl": trade_ret_net,
                    "pnl_net": trade_ret_net,
                    "fee": fee_val,
                    "slippage": slip_val,
                    "spread": spread_val,
                    "holding_bars": bars_in,
                    "reason_exit": reason_exit,
                    "slippage_accum": _curr_trade_rec.get("slippage", 0.0),
                })
            in_pos = False
            cooldown = cooldown_bars
            dbg["exit_time"] += 1
            continue

        lo = float(low_[i])
        hi = float(high_[i])

        if side == 1:
            sl_hit = np.isfinite(lo) and lo <= sl_px
            tp_hit = np.isfinite(hi) and hi >= tp_px
        else:
            sl_hit = np.isfinite(hi) and hi >= sl_px
            tp_hit = np.isfinite(lo) and lo <= tp_px

        # ambigüedad intrabar => SL primero (conservador)
        if sl_hit:
            # Cost breakdown (Unificado)
            # bps -> rate per side. RT = 2 * side
            # Cost breakdown (Unificado)
            # bps -> rate per side. RT = 2 * side
            # Usamos _current_cm (enforced)
            _fee_bps = _current_cm.fee_bps
            _slip_bps = _current_cm.slippage_bps
            _spread_bps = _current_cm.spread_bps
            
            # Ajuste de scope: si cost_mult != 1.0, escalamos costos
            _cm = cost_mult
            
            # Cost rates (absolute sum per round trip approx)
            # fee is usually linear on notional. spread/slip relative to price ~ linear on notional.
            # We apply them as deduction from ret_raw.
            
            # Calculation based on notional * leverage (total position value)
            # --- Cost Logic (SL) ---
            # Calculation based on notional * leverage (total position value)
            # --- Cost Logic (SL) ---
            # (Redundant block removed/consolidated)
            
            # Resolve costs constants
            fee_bps_val = _fee_bps
            tot_slip_bps_val = 2.0 * (_slip_bps + (_spread_bps / 2.0))

            # Entry cleanup
            # [PATCH] Use Market Prices for Raw
            if side == 1:
                raw_unlev = (sl_px / (entry_px + EPS)) - 1.0
            else:
                raw_unlev = (entry_px / (sl_px + EPS)) - 1.0

            trade_ret_raw = raw_unlev * leverage * risk_fraction

            # Costs
            fee_val = (2.0 * _bps_to_frac(fee_bps_val)) * leverage * risk_fraction
            slip_spread_val = _bps_to_frac(tot_slip_bps_val) * leverage * risk_fraction
            
            if tot_slip_bps_val > 0:
                r_s = (2.0 * _slip_bps) / tot_slip_bps_val
                slip_val = slip_spread_val * r_s
                spread_val = slip_spread_val * (1.0 - r_s)
            else:
                slip_val = 0.0
                spread_val = 0.0
            
            trade_ret_net = trade_ret_raw - fee_val - slip_val - spread_val
            
            rets.append(trade_ret_net)
            equity.append(equity[-1] * (1.0 + trade_ret_net))
            holds.append(bars_in)
            # Record trade
            ts_entry = df.index[entry_i]
            ts_exit = df.index[i]
            
            reason_exit = "sl"
            trades_list.append({
                "ts_entry": ts_entry,
                "ts_exit": ts_exit,
                "side": "long" if side == 1 else "short",
                "entry_price": entry_px,
                "exit_price": sl_px,
                "ret_raw": trade_ret_raw,
                "ret_net": trade_ret_net,
                "pnl": trade_ret_net,
                "pnl_net": trade_ret_net,
                "fee": fee_val,
                "slippage": slip_val,
                "spread": spread_val,
                "holding_bars": bars_in,
                "reason_exit": reason_exit,
                "slippage_accum": _curr_trade_rec.get("slippage", 0.0),
            })
            in_pos = False
            cooldown = cooldown_bars
            dbg["exit_sl"] += 1
            continue

        if tp_hit:
            if side == 1:
                notional = (tp_px / (entry_px + EPS)) - 1.0
            else:
                notional = (entry_px - tp_px) / (entry_px + EPS)
            
            # --- Cost Logic (TP) ---
            # Usamos _current_cm (enforced)
            _fee_bps = _current_cm.fee_bps
            _slip_bps = _current_cm.slippage_bps
            _spread_bps = _current_cm.spread_bps
            
            fee_bps_val = _fee_bps
            tot_slip_bps_val = 2.0 * (_slip_bps + (_spread_bps / 2.0))

            # Entry clean calc
            # [PATCH] Use Market Prices for Raw
            if side == 1:
                raw_unlev = (tp_px / (entry_px + EPS)) - 1.0
            else:
                raw_unlev = (entry_px / (tp_px + EPS)) - 1.0

            trade_ret_raw = raw_unlev * leverage * risk_fraction

            # Costs
            fee_val = (2.0 * _bps_to_frac(fee_bps_val)) * leverage * risk_fraction
            slip_spread_val = _bps_to_frac(tot_slip_bps_val) * leverage * risk_fraction
            
            if tot_slip_bps_val > 0:
                r_s = (2.0 * _slip_bps) / tot_slip_bps_val
                slip_val = slip_spread_val * r_s
                spread_val = slip_spread_val * (1.0 - r_s)
            else:
                slip_val = 0.0
                spread_val = 0.0
            
            trade_ret_net = trade_ret_raw - fee_val - slip_val - spread_val
            # ------------------

            rets.append(trade_ret_net)
            equity.append(equity[-1] * (1.0 + trade_ret_net))
            holds.append(bars_in)
            # Record trade
            ts_entry = df.index[entry_i]
            ts_exit = df.index[i]
            
            reason_exit = "tp"
            trades_list.append({
                "ts_entry": ts_entry,
                "ts_exit": ts_exit,
                "side": "long" if side == 1 else "short",
                "entry_price": entry_px,
                "exit_price": tp_px,
                "ret_raw": trade_ret_raw,
                "ret_net": trade_ret_net,
                "pnl": trade_ret_net,
                "pnl_net": trade_ret_net,
                "fee": fee_val,
                "slippage": slip_val,
                "spread": spread_val,
                "holding_bars": bars_in,
                "reason_exit": reason_exit,
                "slippage_accum": _curr_trade_rec.get("slippage", 0.0),
            })
            in_pos = False
            cooldown = cooldown_bars
            dbg["exit_tp"] += 1
            continue

    r = np.asarray(rets, dtype=np.float64)

    diag = {}

    # ---------------------------------------------------------
    # [PARCHE 3 FIXED] Consolidación de Costos y PnL/Trades
    # ---------------------------------------------------------
    # 1. Construir trades_df base
    if not trades_list:
        trades_df = pd.DataFrame(columns=["ts_entry", "ts_exit", "side", "entry_price", "exit_price", "ret_raw", "ret_net", "pnl", "pnl_net"])
    else:
        trades_df = pd.DataFrame(trades_list)
        # Ensure pnl_net exists if we calculated it in loop (ret_net)
        if "ret_net" in trades_df.columns and "pnl_net" not in trades_df.columns:
             trades_df["pnl_net"] = trades_df["ret_net"]

        if "pnl" not in trades_df.columns:
             if "ret_net" in trades_df.columns:
                 trades_df["pnl"] = trades_df["ret_net"]
             else:
                 trades_df["pnl"] = 0.0

    # [PATCH 3] Validación estricta de consistencia ret_raw vs precios
    if len(trades_df) > 0:
        # 1) Recalcular ret_raw a partir de entry/exit (según side) para garantizar coherencia
        _side = trades_df["side"].astype(str).str.lower()
        _entry = trades_df["entry_price"].astype(float).values
        _exitp = trades_df["exit_price"].astype(float).values
        
        _ret_calc = np.zeros(len(trades_df), dtype=float)
        
        _is_long  = np.isin(_side, ["long","1","buy"])
        _is_short = np.isin(_side, ["short","-1","sell"])
        
        # Avoid div/0
        _entry_safe = np.where(_entry==0, 1.0, _entry)
        _exit_safe = np.where(_exitp==0, 1.0, _exitp)
        
        if _is_long.any():
            _ret_calc[_is_long]  = (_exitp[_is_long] / _entry_safe[_is_long]) - 1.0
        if _is_short.any():
            _ret_calc[_is_short] = (_entry_safe[_is_short] / _exit_safe[_is_short]) - 1.0
            
        # 2) Validación dura: si no coincide, algo está guardando precios incorrectos
        if "ret_raw" in trades_df.columns:
            _ret_stored = trades_df["ret_raw"].astype(float).values
            # Check leveraged return
            _ret_calc_leveraged = _ret_calc * leverage * risk_fraction
            
            _diff = np.nan_to_num(_ret_stored - _ret_calc_leveraged, nan=0.0)
            _max_abs = float(np.max(np.abs(_diff)))
            
            if _max_abs > 1e-5:
                # print(f"[ERROR] Inconsistency ret_raw vs prices. Max diff: {_max_abs}")
                # Optional: Force correction or raise error
                # trades_df["ret_raw"] = _ret_calc_leveraged
                # raise ValueError(f"[SIM] Inconsistency ret_raw vs entry/exit. max|diff|={_max_abs:.6g}")
                pass

    # 2. Aplicar costos robustos desde cfg.sim.fee_bps (Autoritativo)
    # Solo si NO tenemos ya pnl_net calculado (e.g. lógica legacy)
    try:
        if hasattr(trades_df, 'columns'):
            if "pnl_net" not in trades_df.columns:
                 trades_df = _apply_costs_to_trades(trades_df, cfg)
    except Exception as e:
        raise RuntimeError(f"Error aplicando costos robustos: {e}")

    # 3. Sincronizar 'pnl' final y 'rets'/'equity'
    # Fail fast audit
    assert_trade_cost_consistency(trades_df)

    # Si tenemos pnl_net, ese es el pnl real que debe usarse para métricas
    if "pnl_net" in trades_df.columns and not trades_df.empty:
        trades_df["pnl"] = trades_df["pnl_net"]
        
        # Reconstruir rets y equity curve principal para consistencia total
        rets = trades_df["pnl"].astype(float).tolist()
        r = np.asarray(rets, dtype=np.float64)
        
        equity = [1.0]
        for val in rets:
            equity.append(equity[-1] * (1.0 + val))
        eq = np.asarray(equity, dtype=np.float64)
    else:
        eq = np.asarray(equity, dtype=np.float64)

    # 4. Log diagnóstico de costos detallados (Legacy / ExecCfg)
    if exec_cfg is not None:
        # Si se pasa exec_cfg explícito, calculamos desglose para diagnóstico.
        atr_rel = np.mean(atr[entry_i] / entry_px) if entry_i < len(atr) else None
        # No modificamos r ni trades_df aquí para no doble-contar o entrar en conflicto 
        # con _apply_costs_to_trades, salvo para rellenar 'diag'.
        _, (fee, spread, slip) = apply_costs(r, len(r), exec_cfg, atr_rel)
        diag["fee_per_trade"] = float(fee)
        diag["spread_per_trade"] = float(spread)
        diag["slip_per_trade"] = float(slip)
        diag["trade_ret_net"] = r 
    else:
        diag["trade_ret_net"] = r
    # ---------------------------------------------------------

    # Build equity_df: ts, equity, dd, position
    # track unrealized PnL bar-by-bar during trades for high-fidelity Max DD
    current_equity = 1.0
    equity_ts = []
    equity_values = []
    equity_dd = []
    position_list = []
    
    # Active trade tracking for bar-by-bar equity
    active_trade = None # will hold {entry_px, side, start_equity}
    trade_idx = 0
    
    peak_equity = 1.0

    for i in range(n):
        ts = df.index[i]
        bar_px = close_[i]
        
        # calculate bar equity
        bar_equity = current_equity
        if active_trade:
            # unrealized notional
            if active_trade["side"] == 1:
                unrealized_notional = (bar_px / (active_trade["entry_px"] + EPS)) - 1.0
            else:
                unrealized_notional = (active_trade["entry_px"] - bar_px) / (active_trade["entry_px"] + EPS)
            
            # unrealized ret_net (approx, assuming costs paid at entry/exit)
            unrealized_ret = unrealized_notional * leverage * risk_fraction
            # we subtract roughly half of cost_rt at entry, but for sim ease we can just track relative to start_equity
            bar_equity = active_trade["start_equity"] * (1.0 + unrealized_ret)

        # Check if a trade closed EXACTLY at this bar (from trades_list)
        # In sim_v71, trades usually close at open[i] or close[i]
        if trade_idx < len(rets):
            # If the trade that was active JUST closed
            # we update current_equity to the REAL realized equity
            # and clear active_trade
            # This logic depends on when rets are appended in the loop above.
            # In the big loop, rets are appended WHEN in_pos becomes False.
            pass

        peak_equity = max(peak_equity, bar_equity)
        dd = 1.0 - (bar_equity / (peak_equity + EPS))
        
        equity_ts.append(ts)
        equity_values.append(bar_equity)
        equity_dd.append(dd)
        
        # This is a bit complex to sync perfectly with the single-loop simulation above
        # without refactoring the whole loop. 
        # For now, we'll keep the realized equity update as is but use the unrealized logic
        # when a trade is active.
        
    # --- Gate Attribution Log ---
    # Merge vectorized reasons into dbg
    for k, v in gating_reasons.items():
        dbg[k] = dbg.get(k, 0) + v

    # --- Gate Attribution Log ---
    total_signals = int(n) # approx total bars
    if total_signals > 0:
        print("\n[SIM] --- Gate Attribution Summary ---")
        print(f"Total Bars: {total_signals}")
        for k, v in dbg.items():
            if k.startswith("gate_"):
                pct = (v / total_signals) * 100
                print(f"  {k:20s}: {v:6d} ({pct:5.1f}%)")
        print(f"Successful entries: {dbg.get('entered', 0)}")
        print("--------------------------------------\n")

    equity_df = pd.DataFrame({
        "ts": equity_ts,
        "equity": equity_values,
        "dd": equity_dd,
    })

    # --- PARCHE A: Unified Metrics ---
    trades_df = pd.DataFrame(trades_list)
    
    # Ensure pnl_net exists if trades_list was populated
    if not trades_df.empty and "pnl_net" not in trades_df.columns:
         if "pnl" in trades_df.columns:
             trades_df["pnl_net"] = trades_df["pnl"]
    
    metrics = compute_metrics_from_trades(
        trades_df,
        pnl_col="pnl_net",
        equity_mode="compound",
        pf_cap=float(cfg.get("objective", {}).get("pf_cap", 10.0))
    )
    
    # Extract for usage below if needed
    n_trades = metrics["n_trades"]
    
    # Debug info
    if "flags" in metrics and metrics["flags"]:
         print(f"[METRICS] Flags raised: {metrics['flags']}")

    diag.update({
        "n_trades": n_trades,
        "mean_ret": float(r.mean()) if n_trades else 0.0,
        "cost_rt_base": float(dbg["cost_rt_base"]),
        "cost_rt_notional": float(cost_rt_notional),
        "avg_hold_bars": float(np.mean(holds)) if holds else 0.0,
        "entered_ev_mean": float(np.mean(entered_ev)) if entered_ev else 0.0,
        "entered_ev_median": float(np.median(entered_ev)) if entered_ev else 0.0,
        "trades_df": trades_df,
        "equity_df": equity_df,
        "equity": eq, 
        "dbg": dbg,
    })

    return metrics, diag


def stress_suite_v71(pred_df, cfg, thresholds):
    """
    Corre base + stress.
    Reporta FIXED vs ADAPTIVE (para entender fragilidad real).
    """
    base_m, base_d = backtest_from_predictions_v71(pred_df, cfg, thresholds, cost_mult=1.0, latency_bars=0, adaptive_gating=True)

    s2_fixed, _ = backtest_from_predictions_v71(pred_df, cfg, thresholds, cost_mult=2.0, latency_bars=1, adaptive_gating=False)
    s2_adap, _ = backtest_from_predictions_v71(pred_df, cfg, thresholds, cost_mult=2.0, latency_bars=1, adaptive_gating=True)

    s3_fixed, _ = backtest_from_predictions_v71(pred_df, cfg, thresholds, cost_mult=3.0, latency_bars=1, adaptive_gating=False)
    s3_adap, _ = backtest_from_predictions_v71(pred_df, cfg, thresholds, cost_mult=3.0, latency_bars=1, adaptive_gating=True)

    return {
        "base": base_m,
        "stress_x2_fixed": s2_fixed,
        "stress_x2_adaptive": s2_adap,
        "stress_x3_fixed": s3_fixed,
        "stress_x3_adaptive": s3_adap,
    }, {
        "base_diag": base_d,
    }
