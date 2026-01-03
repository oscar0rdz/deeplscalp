# deeplscalp/backtest/sim_v7.py
import heapq
import numpy as np
import pandas as pd

# Compatibilidad: usa tu función actual si existe
try:
    from deeplscalp.utils.costs_v6 import round_trip_cost_rt
except Exception:
    # fallback si migras a get_cost_rt (según patches previos)
    try:
        from deeplscalp.utils.costs import get_cost_rt as round_trip_cost_rt
    except Exception:
        round_trip_cost_rt = None


def topk_streaming_by_day(index: pd.DatetimeIndex, score: np.ndarray, top_k: int) -> np.ndarray:
    """
    Marca True si el score actual está dentro del top-K *observado hasta ahora* en el día.
    (Sin lookahead; esto es lo que sí puedes hacer en vivo.)
    """
    n = len(score)
    out = np.zeros(n, dtype=bool)

    day_key = index.floor("D").to_numpy()
    cur_day = None
    heap = []  # min-heap con (score, idx)

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
            out[i] = True  # mientras no llenas, permites explorar
        else:
            kth = heap[0][0]
            if s > kth:
                heapq.heapreplace(heap, (s, i))
                kth = heap[0][0]
            out[i] = (s >= kth)

    return out


def _profit_factor(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    pos = r[r > 0].sum()
    neg = r[r < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / abs(neg))


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = 1.0 - (equity / (peak + 1e-12))
    return float(np.max(dd))


def _sortino_proxy(returns: np.ndarray) -> float:
    """
    Sortino proxy por trade:
      mean(r) / std(neg_r)
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    neg = r[r < 0]
    if neg.size == 0:
        return float("inf") if r.mean() > 0 else 0.0
    dd = neg.std() + 1e-12
    return float(r.mean() / dd)


def _compute_atr_from_ohlc(df: pd.DataFrame, period: int) -> np.ndarray:
    """
    ATR simple (SMA del True Range) usando SOLO pasado.
    Retorna array float con NaNs al principio.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    return atr.to_numpy(dtype=float)


def _rolling_quantile_past(arr: np.ndarray, q: float, window: int, min_periods: int) -> np.ndarray:
    """
    Umbral rolling quantile usando SOLO pasado (shift(1)).
    threshold[i] = quantile(arr[i-window:i], q)
    """
    s = pd.Series(arr.astype(np.float64))
    thr = s.rolling(window, min_periods=min_periods).quantile(q).shift(1)
    return thr.to_numpy(dtype=float)


def _apply_slippage_price(px: float, side: int, is_entry: bool, slip: float) -> float:
    """
    side: +1 long, -1 short
    slip: proporción (ej 0.0002 = 2 bps)

    Entry:
      long compra peor -> +slip
      short vende peor  -> -slip

    Exit:
      long vende peor -> -slip
      short compra peor -> +slip
    """
    if not np.isfinite(px) or px <= 0:
        return px
    if slip <= 0:
        return px

    if is_entry:
        if side == 1:
            return px * (1.0 + slip)
        else:
            return px * (1.0 - slip)
    else:
        if side == 1:
            return px * (1.0 - slip)
        else:
            return px * (1.0 + slip)


def backtest_from_predictions_v7(pred_df: pd.DataFrame, cfg: dict, thresholds: dict):
    df = pred_df.copy()

    # --- robust config access (nunca KeyError) ---
    risk_cfg = cfg.get("risk")
    if not isinstance(risk_cfg, dict):
        risk_cfg = {}

    # --- Fallbacks duros para que nunca truene ---
    if "open_next" not in df.columns:
        if "open" not in df.columns:
            raise KeyError("pred_df requiere 'open' o 'open_next' para ejecución realista.")
        df["open_next"] = df["open"].shift(-1)

    if "iqrL_gross" not in df.columns and "iqrS_gross" not in df.columns:
        # si llegara a faltar, no bloqueamos: OOD=0 (no filtra)
        df["iqrL_gross"] = 0.0
        df["iqrS_gross"] = 0.0

    # --- V6: SIDE + LONG/SHORT con reevaluación ---
    cost_rt = float(round_trip_cost_rt(cfg))

    # Probabilidades SIDE (V6)
    if "p_long" not in df.columns or "p_short" not in df.columns or "p_flat" not in df.columns:
        raise ValueError("V6 requiere p_long, p_short, p_flat en pred_df.")

    p_long = df["p_long"].astype(float).values
    p_short = df["p_short"].astype(float).values
    p_flat = df["p_flat"].astype(float).values

    # Probabilidades HIT L/S (V6)
    if "pL_tp" not in df.columns or "pL_sl" not in df.columns or "pS_tp" not in df.columns or "pS_sl" not in df.columns:
        raise ValueError("V6 requiere pL_tp, pL_sl, pS_tp, pS_sl en pred_df.")

    pL_tp = df["pL_tp"].astype(float).values
    pL_sl = df["pL_sl"].astype(float).values
    pS_tp = df["pS_tp"].astype(float).values
    pS_sl = df["pS_sl"].astype(float).values

    # Quantiles GROSS L/S (V6)
    if "qL50_gross" not in df.columns or "qL10_gross" not in df.columns or "qL90_gross" not in df.columns:
        raise ValueError("V6 requiere qL*_gross en pred_df.")
    if "qS50_gross" not in df.columns or "qS10_gross" not in df.columns or "qS90_gross" not in df.columns:
        raise ValueError("V6 requiere qS*_gross en pred_df.")

    qL10 = df["qL10_gross"].astype(float).values
    qL50 = df["qL50_gross"].astype(float).values
    qL90 = df["qL90_gross"].astype(float).values

    qS10 = df["qS10_gross"].astype(float).values
    qS50 = df["qS50_gross"].astype(float).values
    qS90 = df["qS90_gross"].astype(float).values

    # EV L/S net (aplica costo una sola vez)
    evL_net = (pL_tp * np.maximum(qL90, 0.0)) + (pL_sl * np.minimum(qL10, 0.0)) - cost_rt
    evS_net = (pS_tp * np.maximum(qS90, 0.0)) + (pS_sl * np.minimum(qS10, 0.0)) - cost_rt

    # Mejor EV por lado, pero solo si supera fricción (ya aplicado en labels)
    best_ev = np.maximum(evL_net, evS_net)
    risk = np.maximum(df["iqrL_gross"].astype(float).values, df["iqrS_gross"].astype(float).values)
    risk = np.maximum(risk, 1e-6)

    # score riesgo-ajustado (mejor lado)
    score = best_ev / risk
    df["score"] = score.astype("float32")

    # gates V6
    score_pct = float(thresholds.get("score_pct", 0.75))
    ood_pct = float(thresholds.get("ood_pct", 0.90))
    p_side_min = float(thresholds.get("p_side_min", 0.50))
    p_tp_min = float(thresholds.get("p_tp_min", 0.20))
    p_sl_max = float(thresholds.get("p_sl_max", 0.45))
    ev_min = float(thresholds.get("ev_min", 0.0003))  # 0.03% neto mínimo

    score_arr = df["score"].astype(float).values
    iqr_arr = risk  # usa el max IQR

    score_min = float(np.quantile(score_arr[np.isfinite(score_arr)], score_pct))
    ood_thr = float(np.quantile(iqr_arr[np.isfinite(iqr_arr)], ood_pct))

    # top-K streaming (sin lookahead)
    top_k = int(thresholds.get("top_k", 12))
    df["_topk"] = topk_streaming_by_day(df.index, df["score"].astype(float).values, top_k)

    open_arr = df["open"].astype(float).values
    open_next_arr = df["open_next"].astype(float).values
    close_arr = df["close"].astype(float).values
    high_arr = df["high"].astype(float).values
    low_arr = df["low"].astype(float).values

    # ATR para TP/SL fijos
    atr_arr = df["atr"].astype(float).values if "atr" in df.columns else np.full(len(df), 0.01, dtype=float)

    n = len(df)
    if n < 5:
        return (
            {"n_trades": 0, "profit_factor": 0.0, "max_drawdown": 0.0, "winrate": 0.0, "net_ret": 1.0,
             "avg_ret_per_trade": 0.0, "median_ret_per_trade": 0.0},
            {"n_trades": 0, "mean_ret": 0.0, "cost_rt": cost_rt}
        )

    # Thresholds restantes
    cooldown_bars = thresholds.get("cooldown_bars", 1)

    # Configs risk V6
    leverage = float(risk_cfg.get("leverage", 3.0))
    base_h = int(risk_cfg.get("base_horizon", 14))
    tp_atr_mult = float(risk_cfg.get("tp_atr_mult", 1.4))
    sl_atr_mult = float(risk_cfg.get("sl_atr_mult", 1.1))
    exit_ev_min = float(risk_cfg.get("exit_ev_min", 0.0))
    exit_psl_max = float(risk_cfg.get("exit_psl_max", 0.55))
    max_hold_bars = int(risk_cfg.get("max_hold_bars", 16))

    # Sim V6 long/short con reevaluación
    in_pos = False
    side = 0  # 1=long, -1=short
    entry_px = 0.0
    entry_i = 0
    cooldown = 0

    returns = []
    equity = [1.0]
    holds = []
    iqr_entries = []

    # diagnósticos duros
    dbg = {
        "signals": 0,
        "can_enter": 0,
        "entered": 0,
        "entered_long": 0,
        "entered_short": 0,
        "skip_open_next": 0,
        "skip_atr": 0,
        "exit_tp": 0,
        "exit_sl": 0,
        "exit_time": 0,
        "exit_reeval": 0,
    }

    for i in range(1, n):
        if cooldown > 0:
            cooldown -= 1

        if not in_pos:
            # Entry gate V6: SIDE decision
            is_long = (p_long[i] >= p_side_min) and (pL_tp[i] >= p_tp_min) and (pL_sl[i] <= p_sl_max) and (evL_net[i] >= ev_min)
            is_short = (p_short[i] >= p_side_min) and (pS_tp[i] >= p_tp_min) and (pS_sl[i] <= p_sl_max) and (evS_net[i] >= ev_min)

            if is_long and is_short:
                # elige mejor EV
                side_choice = 1 if evL_net[i] >= evS_net[i] else -1
            elif is_long:
                side_choice = 1
            elif is_short:
                side_choice = -1
            else:
                side_choice = 0

            can_enter = (
                side_choice != 0
                and cooldown == 0
                and bool(df["_topk"].iloc[i])
                and (score_arr[i] >= score_min)
                and (iqr_arr[i] <= ood_thr)
            )

            dbg["signals"] += 1
            if can_enter:
                dbg["can_enter"] += 1
                px = float(open_next_arr[i])
                if (not np.isfinite(px)) or px <= 0:
                    dbg["skip_open_next"] += 1
                    continue
                atr_i = float(atr_arr[i])
                if (not np.isfinite(atr_i)) or atr_i <= 0:
                    dbg["skip_atr"] += 1
                    continue
                dbg["entered"] += 1
                if side_choice == 1:
                    dbg["entered_long"] += 1
                else:
                    dbg["entered_short"] += 1

                in_pos = True
                side = side_choice
                entry_px = px
                entry_i = i

                # barreras TBM fijas
                if side == 1:  # long
                    tp_px = entry_px + tp_atr_mult * atr_i
                    sl_px = entry_px - sl_atr_mult * atr_i
                else:  # short
                    tp_px = entry_px - tp_atr_mult * atr_i
                    sl_px = entry_px + sl_atr_mult * atr_i

                time_exit_i = i + max_hold_bars

                iqr_entries.append(float(iqr_arr[i]))
            continue

        # --- Gestión de posición V6 ---
        bars_in = i - entry_i

        # Reevaluación: si EV actual < exit_ev_min o p_sl > exit_psl_max, sal
        if side == 1:
            current_ev = evL_net[i]
            current_psl = pL_sl[i]
        else:
            current_ev = evS_net[i]
            current_psl = pS_sl[i]

        if current_ev < exit_ev_min or current_psl > exit_psl_max:
            # exit en close (reevaluación)
            ex = float(close_arr[i])
            if np.isfinite(ex) and ex > 0:
                if side == 1:
                    cur_ret = (ex / (entry_px + 1e-12)) - 1.0
                else:
                    cur_ret = (entry_px - ex) / entry_px  # short
                trade_ret = cur_ret * leverage - cost_rt
                returns.append(trade_ret)
                equity.append(equity[-1] * (1.0 + trade_ret))
                holds.append(bars_in)
            in_pos = False
            cooldown = cooldown_bars
            dbg["exit_reeval"] += 1
            continue

        # TIME EXIT
        if i >= time_exit_i:
            ex = float(open_arr[i])
            if np.isfinite(ex) and ex > 0:
                if side == 1:
                    cur_ret = (ex / (entry_px + 1e-12)) - 1.0
                else:
                    cur_ret = (entry_px - ex) / entry_px
                trade_ret = cur_ret * leverage - cost_rt
                returns.append(trade_ret)
                equity.append(equity[-1] * (1.0 + trade_ret))
                holds.append(bars_in)
            in_pos = False
            cooldown = cooldown_bars
            dbg["exit_time"] += 1
            continue

        # Intrabar hits
        lo = float(low_arr[i])
        hi = float(high_arr[i])

        if side == 1:  # long
            sl_hit = np.isfinite(lo) and lo <= sl_px
            tp_hit = np.isfinite(hi) and hi >= tp_px
        else:  # short
            sl_hit = np.isfinite(hi) and hi >= sl_px
            tp_hit = np.isfinite(lo) and lo <= tp_px

        if sl_hit:
            if side == 1:
                fill_ret = (sl_px / (entry_px + 1e-12)) - 1.0
            else:
                fill_ret = (entry_px - sl_px) / entry_px
            trade_ret = fill_ret * leverage - cost_rt
            returns.append(trade_ret)
            equity.append(equity[-1] * (1.0 + trade_ret))
            holds.append(bars_in)
            in_pos = False
            cooldown = cooldown_bars
            dbg["exit_sl"] += 1
            continue

        if tp_hit:
            if side == 1:
                fill_ret = (tp_px / (entry_px + 1e-12)) - 1.0
            else:
                fill_ret = (entry_px - tp_px) / entry_px
            trade_ret = fill_ret * leverage - cost_rt
            returns.append(trade_ret)
            equity.append(equity[-1] * (1.0 + trade_ret))
            holds.append(bars_in)
            in_pos = False
            cooldown = cooldown_bars
            dbg["exit_tp"] += 1
            continue

    # Cerrar posición al final
    if in_pos:
        i = n - 1
        ex = float(close_arr[i])
        if np.isfinite(ex) and ex > 0:
            if side == 1:
                cur_ret = (ex / (entry_px + 1e-12)) - 1.0
            else:
                cur_ret = (entry_px - ex) / entry_px
            trade_ret = cur_ret * leverage - cost_rt
            returns.append(trade_ret)
            equity.append(equity[-1] * (1.0 + trade_ret))
            holds.append(i - entry_i)

    r = np.asarray(returns, dtype=np.float64)
    eq = np.asarray(equity, dtype=np.float64)

    n_trades = int(len(r))
    winrate = float((r > 0).mean()) if n_trades else 0.0

    metrics = {
        "n_trades": float(n_trades),
        "profit_factor": float(_profit_factor(r)) if n_trades else 0.0,
        "max_drawdown": float(_max_drawdown(eq)),
        "winrate": float(winrate),
        "net_ret": float(eq[-1]) if len(eq) else 1.0,
        "avg_ret_per_trade": float(r.mean()) if n_trades else 0.0,
        "median_ret_per_trade": float(np.median(r)) if n_trades else 0.0,
    }

    diag = {
        "n_trades": n_trades,
        "mean_ret": float(r.mean()) if n_trades else 0.0,
        "cost_rt": cost_rt,
        "avg_hold_bars": float(np.mean(holds)) if holds else 0.0,
        "avg_iqr_entry": float(np.mean(iqr_entries)) if iqr_entries else 0.0,
    }

    diag["dbg"] = dbg
    print(f"[sim_v6_debug] {dbg}")

    return metrics, diag


def backtest_from_predictions_v7(pred_df: pd.DataFrame, cfg: dict, thresholds: dict):
    """
    Backtest V7 (no-lookahead):
      - Señal se considera al cierre de t
      - Entrada en open[t+1] (usando open_next[t])
      - Reevaluación: decide salir con predicciones de t-1 y sale en open[t]
      - Time exit: open[t] al alcanzar max_hold_bars desde la barra de ejecución

    Requiere columnas típicas de predict_v7():
      p_long, p_short, p_flat
      pL_tp, pL_sl, pS_tp, pS_sl
      qL10_gross, qL90_gross, qL50_gross
      qS10_gross, qS90_gross, qS50_gross
      iqrL_gross, iqrS_gross (si faltan, no filtra)
      open, high, low, close, open_next

    cfg:
      risk:
        leverage: 3.0/5.0
        tp_atr_mult, sl_atr_mult
        max_hold_bars
        exit_ev_min, exit_psl_max
        slippage: 0.0002 (ej 2 bps)
      labels:
        atr_period: 14 (si falta atr en pred_df)

    thresholds:
      p_side_min, p_tp_min, p_sl_max, ev_min
      p_flat_max (opcional)
      top_k
      score_q, ood_q, thr_lookback_bars
      use_topk (bool)
      cooldown_bars
    """
    df = pred_df.copy()

    # --- robust cfg ---
    risk_cfg = cfg.get("risk")
    if not isinstance(risk_cfg, dict):
        risk_cfg = {}

    labels_cfg = cfg.get("labels")
    if not isinstance(labels_cfg, dict):
        labels_cfg = {}

    # --- columnas de ejecución ---
    if "open_next" not in df.columns:
        if "open" not in df.columns:
            raise KeyError("pred_df requiere 'open' o 'open_next' para ejecución realista.")
        df["open_next"] = df["open"].shift(-1)

    # iqr fallback (no filtra)
    if "iqrL_gross" not in df.columns:
        df["iqrL_gross"] = 0.0
    if "iqrS_gross" not in df.columns:
        df["iqrS_gross"] = 0.0

    # --- costo ---
    if round_trip_cost_rt is None:
        raise ImportError("No se encontró función de costo: round_trip_cost_rt o get_cost_rt.")
    cost_rt = float(round_trip_cost_rt(cfg))

    # --- SIDE probs ---
    req_side = ["p_long", "p_short", "p_flat"]
    for c in req_side:
        if c not in df.columns:
            raise ValueError(f"Falta columna {c} en pred_df (SIDE).")

    p_long = df["p_long"].astype(float).values
    p_short = df["p_short"].astype(float).values
    p_flat = df["p_flat"].astype(float).values

    # --- HIT probs ---
    req_hit = ["pL_tp", "pL_sl", "pS_tp", "pS_sl"]
    for c in req_hit:
        if c not in df.columns:
            raise ValueError(f"Falta columna {c} en pred_df (HIT).")

    pL_tp = df["pL_tp"].astype(float).values
    pL_sl = df["pL_sl"].astype(float).values
    pS_tp = df["pS_tp"].astype(float).values
    pS_sl = df["pS_sl"].astype(float).values

    # --- Quantiles GROSS ---
    req_qL = ["qL10_gross", "qL50_gross", "qL90_gross"]
    req_qS = ["qS10_gross", "qS50_gross", "qS90_gross"]
    for c in req_qL + req_qS:
        if c not in df.columns:
            raise ValueError(f"Falta columna {c} en pred_df (Quantiles).")

    qL10 = df["qL10_gross"].astype(float).values
    qL50 = df["qL50_gross"].astype(float).values
    qL90 = df["qL90_gross"].astype(float).values

    qS10 = df["qS10_gross"].astype(float).values
    qS50 = df["qS50_gross"].astype(float).values
    qS90 = df["qS90_gross"].astype(float).values

    # --- EV net (costo una sola vez) ---
    # Nota: esto es un proxy. No modela TIME explícitamente, pero ya "castiga" con q10/q90 + p_hit.
    evL_net = (pL_tp * np.maximum(qL90, 0.0)) + (pL_sl * np.minimum(qL10, 0.0)) - cost_rt
    evS_net = (pS_tp * np.maximum(qS90, 0.0)) + (pS_sl * np.minimum(qS10, 0.0)) - cost_rt

    best_ev = np.maximum(evL_net, evS_net)
    risk = np.maximum(df["iqrL_gross"].astype(float).values, df["iqrS_gross"].astype(float).values)
    risk = np.maximum(risk, 1e-6)

    score = best_ev / risk
    df["score"] = score.astype("float32")

    # --- umbrales sin lookahead (rolling, usando solo pasado) ---
    score_q = float(thresholds.get("score_q", 0.75))           # cuantíl rolling del score
    ood_q = float(thresholds.get("ood_q", 0.90))               # cuantíl rolling del IQR (OOD)
    thr_lookback = int(thresholds.get("thr_lookback_bars", 2000))
    thr_min_periods = int(thresholds.get("thr_min_periods", max(200, thr_lookback // 4)))

    score_arr = df["score"].astype(float).values
    iqr_arr = risk

    score_thr_arr = _rolling_quantile_past(score_arr, score_q, thr_lookback, thr_min_periods)
    ood_thr_arr = _rolling_quantile_past(iqr_arr, ood_q, thr_lookback, thr_min_periods)

    # fallback inicial: si aún no hay umbral, no filtrar por score/ood en warmup
    score_thr_arr = np.where(np.isfinite(score_thr_arr), score_thr_arr, -np.inf)
    ood_thr_arr = np.where(np.isfinite(ood_thr_arr), ood_thr_arr, np.inf)

    # --- top-K streaming ---
    top_k = int(thresholds.get("top_k", 12))
    use_topk = bool(thresholds.get("use_topk", True))
    if use_topk:
        df["_topk"] = topk_streaming_by_day(df.index, score_arr, top_k)
    else:
        df["_topk"] = True

    # --- precios ---
    for c in ["open", "high", "low", "close", "open_next"]:
        if c not in df.columns:
            raise ValueError(f"Falta {c} en pred_df para simulación.")

    open_arr = df["open"].astype(float).values
    open_next_arr = df["open_next"].astype(float).values
    close_arr = df["close"].astype(float).values
    high_arr = df["high"].astype(float).values
    low_arr = df["low"].astype(float).values

    # --- ATR ---
    if "atr" in df.columns:
        atr_arr = df["atr"].astype(float).values
    else:
        atr_period = int(labels_cfg.get("atr_period", 14))
        atr_arr = _compute_atr_from_ohlc(df, atr_period)

    # --- thresholds ---
    p_side_min = float(thresholds.get("p_side_min", 0.50))
    p_tp_min = float(thresholds.get("p_tp_min", 0.20))
    p_sl_max = float(thresholds.get("p_sl_max", 0.45))
    ev_min = float(thresholds.get("ev_min", 0.0003))
    p_flat_max = float(thresholds.get("p_flat_max", 0.80))
    cooldown_bars = int(thresholds.get("cooldown_bars", 1))

    # --- risk config ---
    leverage = float(risk_cfg.get("leverage", 3.0))
    tp_atr_mult = float(risk_cfg.get("tp_atr_mult", 1.4))
    sl_atr_mult = float(risk_cfg.get("sl_atr_mult", 1.1))
    exit_ev_min = float(risk_cfg.get("exit_ev_min", 0.0))
    exit_psl_max = float(risk_cfg.get("exit_psl_max", 0.55))
    max_hold_bars = int(risk_cfg.get("max_hold_bars", 16))

    # slippage (por lado / entry-exit)
    slip = float(risk_cfg.get("slippage", 0.0))

    # costo efectivo en equity por leverage
    cost_eff = cost_rt * leverage

    n = len(df)
    if n < 10:
        return (
            {
                "n_trades": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "winrate": 0.0,
                "net_ret": 1.0,
                "avg_ret_per_trade": 0.0,
                "median_ret_per_trade": 0.0,
                "sortino_proxy": 0.0,
            },
            {"n_trades": 0, "mean_ret": 0.0, "cost_rt": cost_rt, "cost_eff": cost_eff},
        )

    # --- estado ---
    in_pos = False
    side = 0  # +1 long, -1 short
    entry_px = 0.0
    entry_exec_i = -1
    tp_px = np.nan
    sl_px = np.nan
    time_exit_i = -1
    cooldown = 0

    returns = []
    equity = [1.0]
    holds = []
    iqr_entries = []

    dbg = {
        "signals": 0,
        "can_enter": 0,
        "entered": 0,
        "entered_long": 0,
        "entered_short": 0,
        "skip_open_next": 0,
        "skip_atr": 0,
        "exit_tp": 0,
        "exit_sl": 0,
        "exit_time": 0,
        "exit_reeval": 0,
        "warmup_skip": 0,
    }

    # Reglas de "no lookahead":
    # - En i, si estás en posición, reevaluación se decide con predicciones de i-1 y se ejecuta en open[i].
    # - Por lo tanto, la reevaluación se evalúa al inicio de la iteración i, antes de checar TP/SL intrabar.

    for i in range(1, n):
        if cooldown > 0:
            cooldown -= 1

        # Warmup: si ATR o thresholds rolling aún no están listos, no operes
        if (not np.isfinite(atr_arr[i])) or (not np.isfinite(open_arr[i])) or (open_arr[i] <= 0):
            dbg["warmup_skip"] += 1
            continue

        # -------------------------
        # GESTIÓN (si ya estás dentro)
        # -------------------------
        if in_pos:
            # Reevaluación basada en i-1 (sin lookahead)
            j = i - 1
            if side == 1:
                current_ev = evL_net[j]
                current_psl = pL_sl[j]
            else:
                current_ev = evS_net[j]
                current_psl = pS_sl[j]

            if (current_ev < exit_ev_min) or (current_psl > exit_psl_max):
                ex = float(open_arr[i])
                ex = _apply_slippage_price(ex, side, is_entry=False, slip=slip)
                if np.isfinite(ex) and ex > 0:
                    if side == 1:
                        px_ret = (ex / (entry_px + 1e-12)) - 1.0
                    else:
                        px_ret = (entry_px / (ex + 1e-12)) - 1.0  # short notional return
                    trade_ret = (px_ret * leverage) - cost_eff
                    trade_ret = max(trade_ret, -0.95)
                    returns.append(trade_ret)
                    equity.append(equity[-1] * (1.0 + trade_ret))
                    holds.append(i - entry_exec_i + 1)
                in_pos = False
                cooldown = cooldown_bars
                dbg["exit_reeval"] += 1
                continue

            # Time exit en open[i]
            if i >= time_exit_i:
                ex = float(open_arr[i])
                ex = _apply_slippage_price(ex, side, is_entry=False, slip=slip)
                if np.isfinite(ex) and ex > 0:
                    if side == 1:
                        px_ret = (ex / (entry_px + 1e-12)) - 1.0
                    else:
                        px_ret = (entry_px / (ex + 1e-12)) - 1.0
                    trade_ret = (px_ret * leverage) - cost_eff
                    trade_ret = max(trade_ret, -0.95)
                    returns.append(trade_ret)
                    equity.append(equity[-1] * (1.0 + trade_ret))
                    holds.append(i - entry_exec_i + 1)
                in_pos = False
                cooldown = cooldown_bars
                dbg["exit_time"] += 1
                continue

            # Intrabar TP/SL (bar i)
            lo = float(low_arr[i])
            hi = float(high_arr[i])

            if side == 1:
                sl_hit = np.isfinite(lo) and (lo <= sl_px)
                tp_hit = np.isfinite(hi) and (hi >= tp_px)
            else:
                sl_hit = np.isfinite(hi) and (hi >= sl_px)
                tp_hit = np.isfinite(lo) and (lo <= tp_px)

            # Conservador: si ambos, prioriza SL (peor caso)
            if sl_hit and tp_hit:
                sl_hit = True
                tp_hit = False

            if sl_hit:
                ex = float(sl_px)
                ex = _apply_slippage_price(ex, side, is_entry=False, slip=slip)
                if side == 1:
                    px_ret = (ex / (entry_px + 1e-12)) - 1.0
                else:
                    px_ret = (entry_px / (ex + 1e-12)) - 1.0
                trade_ret = (px_ret * leverage) - cost_eff
                trade_ret = max(trade_ret, -0.95)
                returns.append(trade_ret)
                equity.append(equity[-1] * (1.0 + trade_ret))
                holds.append(i - entry_exec_i + 1)
                in_pos = False
                cooldown = cooldown_bars
                dbg["exit_sl"] += 1
                continue

            if tp_hit:
                ex = float(tp_px)
                ex = _apply_slippage_price(ex, side, is_entry=False, slip=slip)
                if side == 1:
                    px_ret = (ex / (entry_px + 1e-12)) - 1.0
                else:
                    px_ret = (entry_px / (ex + 1e-12)) - 1.0
                trade_ret = (px_ret * leverage) - cost_eff
                trade_ret = max(trade_ret, -0.95)
                returns.append(trade_ret)
                equity.append(equity[-1] * (1.0 + trade_ret))
                holds.append(i - entry_exec_i + 1)
                in_pos = False
                cooldown = cooldown_bars
                dbg["exit_tp"] += 1
                continue

            continue

        # -------------------------
        # ENTRADA (si NO estás dentro)
        # -------------------------
        dbg["signals"] += 1

        # Señal en i (cierre de i), ejecución en open_next[i] = open[i+1]
        is_long = (
            (p_long[i] >= p_side_min)
            and (pL_tp[i] >= p_tp_min)
            and (pL_sl[i] <= p_sl_max)
            and (evL_net[i] >= ev_min)
            and (p_flat[i] <= p_flat_max)
        )
        is_short = (
            (p_short[i] >= p_side_min)
            and (pS_tp[i] >= p_tp_min)
            and (pS_sl[i] <= p_sl_max)
            and (evS_net[i] >= ev_min)
            and (p_flat[i] <= p_flat_max)
        )

        if is_long and is_short:
            side_choice = 1 if evL_net[i] >= evS_net[i] else -1
        elif is_long:
            side_choice = 1
        elif is_short:
            side_choice = -1
        else:
            side_choice = 0

        can_enter = (
            (side_choice != 0)
            and (cooldown == 0)
            and bool(df["_topk"].iloc[i])
            and (score_arr[i] >= score_thr_arr[i])
            and (iqr_arr[i] <= ood_thr_arr[i])
        )

        if not can_enter:
            continue

        dbg["can_enter"] += 1

        px = float(open_next_arr[i])
        if (not np.isfinite(px)) or px <= 0:
            dbg["skip_open_next"] += 1
            continue

        atr_i = float(atr_arr[i])
        if (not np.isfinite(atr_i)) or atr_i <= 0:
            dbg["skip_atr"] += 1
            continue

        # Ejecuta entrada en bar i+1 (open_next[i])
        exec_i = i + 1
        if exec_i >= n:
            break

        entry = float(open_arr[exec_i]) if np.isfinite(open_arr[exec_i]) and open_arr[exec_i] > 0 else px
        entry = _apply_slippage_price(entry, side_choice, is_entry=True, slip=slip)

        if (not np.isfinite(entry)) or entry <= 0:
            dbg["skip_open_next"] += 1
            continue

        # Define barreras
        if side_choice == 1:
            tp_ = entry + tp_atr_mult * atr_i
            sl_ = entry - sl_atr_mult * atr_i
        else:
            tp_ = entry - tp_atr_mult * atr_i
            sl_ = entry + sl_atr_mult * atr_i

        in_pos = True
        side = side_choice
        entry_px = entry
        entry_exec_i = exec_i
        tp_px = float(tp_)
        sl_px = float(sl_)
        time_exit_i = exec_i + max_hold_bars

        iqr_entries.append(float(iqr_arr[i]))

        dbg["entered"] += 1
        if side_choice == 1:
            dbg["entered_long"] += 1
        else:
            dbg["entered_short"] += 1

    # Cerrar al final (exit en close[n-1] con slippage)
    if in_pos:
        i = n - 1
        ex = float(close_arr[i])
        ex = _apply_slippage_price(ex, side, is_entry=False, slip=slip)
        if np.isfinite(ex) and ex > 0:
            if side == 1:
                px_ret = (ex / (entry_px + 1e-12)) - 1.0
            else:
                px_ret = (entry_px / (ex + 1e-12)) - 1.0
            trade_ret = (px_ret * leverage) - cost_eff
            trade_ret = max(trade_ret, -0.95)
            returns.append(trade_ret)
            equity.append(equity[-1] * (1.0 + trade_ret))
            holds.append(i - entry_exec_i + 1)

    r = np.asarray(returns, dtype=np.float64)
    eq = np.asarray(equity, dtype=np.float64)

    n_trades = int(len(r))
    winrate = float((r > 0).mean()) if n_trades else 0.0

    metrics = {
        "n_trades": float(n_trades),
        "profit_factor": float(_profit_factor(r)) if n_trades else 0.0,
        "max_drawdown": float(_max_drawdown(eq)),
        "winrate": float(winrate),
        "net_ret": float(eq[-1]) if len(eq) else 1.0,
        "avg_ret_per_trade": float(r.mean()) if n_trades else 0.0,
        "median_ret_per_trade": float(np.median(r)) if n_trades else 0.0,
        "sortino_proxy": float(_sortino_proxy(r)) if n_trades else 0.0,
    }

    diag = {
        "n_trades": n_trades,
        "mean_ret": float(r.mean()) if n_trades else 0.0,
        "cost_rt": float(cost_rt),
        "cost_eff": float(cost_eff),
        "leverage": float(leverage),
        "slippage": float(slip),
        "avg_hold_bars": float(np.mean(holds)) if holds else 0.0,
        "avg_iqr_entry": float(np.mean(iqr_entries)) if iqr_entries else 0.0,
        "dbg": dbg,
    }

    print(f"[sim_v7_debug] {dbg}")

    return metrics, diag
