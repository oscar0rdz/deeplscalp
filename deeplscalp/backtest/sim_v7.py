import heapq
import numpy as np
import pandas as pd

# Cost function fallback (para no romper según tu repo)
def _rt_cost(cfg: dict) -> float:
    try:
        from deeplscalp.utils.costs_v7 import round_trip_cost_rt
        return float(round_trip_cost_rt(cfg))
    except Exception:
        try:
            from deeplscalp.utils.costs_v6 import round_trip_cost_rt
            return float(round_trip_cost_rt(cfg))
        except Exception:
            from deeplscalp.utils.costs import get_cost_rt
            return float(get_cost_rt(cfg))


def topk_streaming_by_day(index: pd.DatetimeIndex, score: np.ndarray, top_k: int) -> np.ndarray:
    """
    True si el score actual está dentro del top-K observado hasta ahora en el día.
    Sin lookahead (usable en vivo).
    """
    n = len(score)
    out = np.zeros(n, dtype=bool)

    day_key = index.floor("D").to_numpy()
    cur_day = None
    heap = []  # min-heap (score, idx)

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


def _profit_factor(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=np.float64)
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


def _sortino_proxy(r: np.ndarray) -> float:
    """
    Sortino proxy: mean / std(downside) * sqrt(N)
    No anualiza (proxy estable para comparar runs).
    """
    r = np.asarray(r, dtype=np.float64)
    if len(r) < 5:
        return 0.0
    neg = r[r < 0]
    dd = np.std(neg) if len(neg) else 0.0
    if dd <= 1e-12:
        return float("inf") if np.mean(r) > 0 else 0.0
    return float(np.mean(r) / dd * np.sqrt(len(r)))


def backtest_from_predictions_v7(pred_df: pd.DataFrame, cfg: dict, thresholds: dict):
    """
    V7 FAST:
    - Entrada por SIDE + EV_net.
    - Umbrales rolling vectorizados (sin loop O(n*lookback) en Python).
    - Reevaluación sin lookahead: decide en t-1, ejecuta salida en open[t].
    - Leverage 3x/5x: escala retorno y fricción.
    - Métricas: PF, MDD, Sortino proxy.
    """
    df = pred_df.copy()
    print(f"[DEBUG sim_v7] index type: {type(df.index)}, name: {df.index.name}, has ds: {'ds' in df.columns}")
    if not isinstance(df.index, pd.DatetimeIndex):
        # Si viene con columna ds, conviértela
        if "ds" in df.columns:
            df = df.set_index(pd.to_datetime(df["ds"], utc=True))
        else:
            raise ValueError("pred_df debe tener DatetimeIndex o columna 'ds'.")

    # --- config ---
    risk_cfg = cfg.get("risk") if isinstance(cfg.get("risk"), dict) else {}
    leverage = float(risk_cfg.get("leverage", 3.0))
    max_hold_bars = int(risk_cfg.get("max_hold_bars", 16))
    tp_atr_mult = float(risk_cfg.get("tp_atr_mult", 1.4))
    sl_atr_mult = float(risk_cfg.get("sl_atr_mult", 1.1))
    exit_ev_min = float(risk_cfg.get("exit_ev_min", 0.0))
    exit_psl_max = float(risk_cfg.get("exit_psl_max", 0.55))
    slippage = float(risk_cfg.get("slippage", 0.0002))

    # thresholds
    p_side_min = float(thresholds.get("p_side_min", 0.52))
    p_tp_min   = float(thresholds.get("p_tp_min", 0.22))
    p_sl_max   = float(thresholds.get("p_sl_max", 0.45))
    ev_min     = float(thresholds.get("ev_min", 0.0003))
    use_topk   = bool(thresholds.get("use_topk", True))
    top_k      = int(thresholds.get("top_k", 12))
    cooldown_bars = int(thresholds.get("cooldown_bars", 1))

    score_q = float(thresholds.get("score_q", 0.75))
    ood_q   = float(thresholds.get("ood_q", 0.90))
    lb      = int(thresholds.get("thr_lookback_bars", 2000))
    mp      = int(thresholds.get("thr_min_periods", 500))

    # --- required cols ---
    req = [
        "p_long","p_short","p_flat",
        "pL_tp","pL_sl","pS_tp","pS_sl",
        "qL10_gross","qL50_gross","qL90_gross",
        "qS10_gross","qS50_gross","qS90_gross",
        "iqrL_gross","iqrS_gross",
        "open","open_next","high","low","close","atr",
    ]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Faltan columnas en pred_df: {miss}")

    cost_rt = _rt_cost(cfg) * leverage  # fricción escala con notional

    # arrays
    p_long  = df["p_long"].astype(float).to_numpy()
    p_short = df["p_short"].astype(float).to_numpy()

    pL_tp = df["pL_tp"].astype(float).to_numpy()
    pL_sl = df["pL_sl"].astype(float).to_numpy()
    pS_tp = df["pS_tp"].astype(float).to_numpy()
    pS_sl = df["pS_sl"].astype(float).to_numpy()

    qL10 = df["qL10_gross"].astype(float).to_numpy()
    qL50 = df["qL50_gross"].astype(float).to_numpy()
    qL90 = df["qL90_gross"].astype(float).to_numpy()

    qS10 = df["qS10_gross"].astype(float).to_numpy()
    qS50 = df["qS50_gross"].astype(float).to_numpy()
    qS90 = df["qS90_gross"].astype(float).to_numpy()

    iqrL = df["iqrL_gross"].astype(float).to_numpy()
    iqrS = df["iqrS_gross"].astype(float).to_numpy()

    risk = np.maximum(np.maximum(iqrL, iqrS), 1e-6)

    # EV net (sin leverage aquí; leverage se aplica a PnL final)
    evL_net = (pL_tp*np.maximum(qL90, 0.0)) + (pL_sl*np.minimum(qL10, 0.0)) - (cost_rt / leverage)
    evS_net = (pS_tp*np.maximum(qS90, 0.0)) + (pS_sl*np.minimum(qS10, 0.0)) - (cost_rt / leverage)

    best_ev = np.maximum(evL_net, evS_net)
    score = best_ev / risk

    # --- thresholds rolling vectorizados (sin lookahead) ---
    s_score = pd.Series(score, index=df.index)
    s_risk  = pd.Series(risk, index=df.index)

    score_thr = s_score.rolling(lb, min_periods=mp).quantile(score_q).shift(1)
    ood_thr   = s_risk.rolling(lb,  min_periods=mp).quantile(ood_q).shift(1)

    # inicial: no bloquees por falta de historia
    score_thr = score_thr.fillna(-np.inf).to_numpy()
    ood_thr   = ood_thr.fillna(np.inf).to_numpy()

    # topk streaming
    if use_topk:
        topk_mask = topk_streaming_by_day(df.index, score.astype(np.float64), top_k=top_k)
    else:
        topk_mask = np.ones(len(df), dtype=bool)

    open_ = df["open"].astype(float).to_numpy()
    open_next = df["open_next"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low  = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    atr = df["atr"].astype(float).to_numpy()

    n = len(df)
    if n < 10:
        return (
            {"n_trades": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "sortino": 0.0, "winrate": 0.0, "net_ret": 1.0},
            {"n_trades": 0, "mean_ret": 0.0, "cost_rt": cost_rt, "dbg": {}}
        )

    in_pos = False
    side = 0  # 1 long, -1 short
    entry_px = 0.0
    entry_i = 0
    tp_px = 0.0
    sl_px = 0.0
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
        "gate_score": 0,
        "gate_ood": 0,
        "gate_topk": 0,
        "gate_side": 0,
        "exit_tp": 0,
        "exit_sl": 0,
        "exit_time": 0,
        "exit_reeval": 0,
    }

    def _pnl_ret(exit_px: float, entry_px: float, side: int) -> float:
        # retorno subyacente
        if side == 1:
            r = (exit_px / (entry_px + 1e-12)) - 1.0
        else:
            r = (entry_px - exit_px) / (entry_px + 1e-12)
        # slippage ida+vuelta aproximado (en retorno)
        r = r - (2.0 * slippage)
        # leverage y fricción
        return (r * leverage) - cost_rt

    for i in range(1, n):
        if cooldown > 0:
            cooldown -= 1

        if not in_pos:
            dbg["signals"] += 1

            # SIDE gate (económico y probabilístico)
            is_long = (p_long[i] >= p_side_min) and (pL_tp[i] >= p_tp_min) and (pL_sl[i] <= p_sl_max) and (evL_net[i] >= ev_min)
            is_short = (p_short[i] >= p_side_min) and (pS_tp[i] >= p_tp_min) and (pS_sl[i] <= p_sl_max) and (evS_net[i] >= ev_min)

            if not (is_long or is_short):
                dbg["gate_side"] += 1
                continue

            # side choice
            if is_long and is_short:
                side_choice = 1 if evL_net[i] >= evS_net[i] else -1
            else:
                side_choice = 1 if is_long else -1

            # gates de calidad (sin lookahead)
            if not topk_mask[i]:
                dbg["gate_topk"] += 1
                continue
            if score[i] < score_thr[i]:
                dbg["gate_score"] += 1
                continue
            if risk[i] > ood_thr[i]:
                dbg["gate_ood"] += 1
                continue
            if cooldown != 0:
                continue

            dbg["can_enter"] += 1

            px = float(open_next[i])
            if (not np.isfinite(px)) or px <= 0:
                dbg["skip_open_next"] += 1
                continue
            a = float(atr[i])
            if (not np.isfinite(a)) or a <= 0:
                dbg["skip_atr"] += 1
                continue

            in_pos = True
            side = side_choice
            entry_px = px
            entry_i = i

            if side == 1:
                tp_px = entry_px + tp_atr_mult * a
                sl_px = entry_px - sl_atr_mult * a
                dbg["entered_long"] += 1
            else:
                tp_px = entry_px - tp_atr_mult * a
                sl_px = entry_px + sl_atr_mult * a
                dbg["entered_short"] += 1

            dbg["entered"] += 1
            iqr_entries.append(float(risk[i]))
            continue

        # --- Gestión de posición ---
        bars_in = i - entry_i

        # 1) Reevaluación SIN lookahead:
        # decide con predicciones de i-1, ejecuta salida en OPEN[i]
        if i - 1 >= entry_i:
            if side == 1:
                prev_ev = evL_net[i - 1]
                prev_psl = pL_sl[i - 1]
            else:
                prev_ev = evS_net[i - 1]
                prev_psl = pS_sl[i - 1]

            if (prev_ev < exit_ev_min) or (prev_psl > exit_psl_max):
                ex = float(open_[i])
                if np.isfinite(ex) and ex > 0:
                    tr = _pnl_ret(ex, entry_px, side)
                    returns.append(tr)
                    equity.append(equity[-1] * (1.0 + tr))
                    holds.append(bars_in)
                in_pos = False
                cooldown = cooldown_bars
                dbg["exit_reeval"] += 1
                continue

        # 2) TIME EXIT (OPEN[i])
        if bars_in >= max_hold_bars:
            ex = float(open_[i])
            if np.isfinite(ex) and ex > 0:
                tr = _pnl_ret(ex, entry_px, side)
                returns.append(tr)
                equity.append(equity[-1] * (1.0 + tr))
                holds.append(bars_in)
            in_pos = False
            cooldown = cooldown_bars
            dbg["exit_time"] += 1
            continue

        # 3) Intrabar TP/SL (conservador: SL primero)
        lo = float(low[i])
        hi = float(high[i])

        if side == 1:
            sl_hit = np.isfinite(lo) and lo <= sl_px
            tp_hit = np.isfinite(hi) and hi >= tp_px
        else:
            sl_hit = np.isfinite(hi) and hi >= sl_px
            tp_hit = np.isfinite(lo) and lo <= tp_px

        if sl_hit:
            ex = float(sl_px)
            tr = _pnl_ret(ex, entry_px, side)
            returns.append(tr)
            equity.append(equity[-1] * (1.0 + tr))
            holds.append(bars_in)
            in_pos = False
            cooldown = cooldown_bars
            dbg["exit_sl"] += 1
            continue

        if tp_hit:
            ex = float(tp_px)
            tr = _pnl_ret(ex, entry_px, side)
            returns.append(tr)
            equity.append(equity[-1] * (1.0 + tr))
            holds.append(bars_in)
            in_pos = False
            cooldown = cooldown_bars
            dbg["exit_tp"] += 1
            continue

    # cierre al final
    if in_pos:
        ex = float(close[-1])
        if np.isfinite(ex) and ex > 0:
            tr = _pnl_ret(ex, entry_px, side)
            returns.append(tr)
            equity.append(equity[-1] * (1.0 + tr))
            holds.append((n - 1) - entry_i)

    r = np.asarray(returns, dtype=np.float64)
    eq = np.asarray(equity, dtype=np.float64)

    n_trades = int(len(r))
    winrate = float((r > 0).mean()) if n_trades else 0.0

    metrics = {
        "n_trades": float(n_trades),
        "profit_factor": float(_profit_factor(r)) if n_trades else 0.0,
        "max_drawdown": float(_max_drawdown(eq)),
        "sortino": float(_sortino_proxy(r)) if n_trades else 0.0,
        "winrate": float(winrate),
        "net_ret": float(eq[-1]) if len(eq) else 1.0,
        "avg_ret_per_trade": float(r.mean()) if n_trades else 0.0,
        "median_ret_per_trade": float(np.median(r)) if n_trades else 0.0,
    }

    diag = {
        "n_trades": n_trades,
        "mean_ret": float(r.mean()) if n_trades else 0.0,
        "cost_rt_eff": float(cost_rt),
        "avg_hold_bars": float(np.mean(holds)) if holds else 0.0,
        "avg_iqr_entry": float(np.mean(iqr_entries)) if iqr_entries else 0.0,
        "dbg": dbg,
    }

    print(f"[sim_v7_fast_debug] {dbg}")
    return metrics, diag
