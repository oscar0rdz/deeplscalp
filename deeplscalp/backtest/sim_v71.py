import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-12

@dataclass(frozen=True)
class ExecConfig:
    exec_lag_bars: int = 1          # 1 = next bar
    fee_bps: float = 4.0            # por lado (entry y exit)
    spread_bps: float = 1.0         # costo implícito (half-spread por lado si market)
    slippage_bps: float = 2.0       # base
    slippage_atr_k: float = 0.0     # extra proporcional a ATR/price (si quieres)

def _bps_cost_to_ret(cost_bps: float) -> float:
    return float(cost_bps) * 1e-4

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
DEFAULT_PF_CAP = 20.0

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
        pf = float(pf_cap if pos > 0 else 0.0)
    else:
        pf = float(min(pf_cap, pos / max(neg, PF_EPS)))
    return ProfitFactorStats(gross_profit=pos, gross_loss=neg, pf=pf, zero_loss=zero_loss)


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


def _profit_factor(r: np.ndarray) -> float:
    # Mantén esta función si otras partes la usan, pero ahora es robusta y finita.
    s = profit_factor_stats(r)
    return float(s.pf)


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = 1.0 - (equity / (peak + EPS))
    return float(np.max(dd))


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
    cost_rt_notional = _round_trip_cost_rt(cfg, cost_mult=cost_mult)

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

    # gates base
    p_side_min = float(thresholds.get("p_side_min", 0.52))
    p_tp_min = float(thresholds.get("p_tp_min", 0.20))
    p_sl_max = float(thresholds.get("p_sl_max", 0.45))

    # COST BUFFER gate (CLAVE)
    ev_abs_min = float(thresholds.get("ev_abs_min", 0.0))
    ev_buffer_mult = float(thresholds.get("ev_buffer_mult", 1.0))  # net >= 1x costo
    q_width_mult = float(thresholds.get("q_width_mult", 2.5))      # q90-q10 >= 2.5x costo

    ev_buffer = ev_buffer_mult * cost_rt_notional if adaptive_gating else 0.0
    q_width_thr = q_width_mult * cost_rt_notional if adaptive_gating else 0.0

    # topk
    use_topk = bool(thresholds.get("use_topk", True))
    top_k = int(thresholds.get("top_k", 10))
    if use_topk:
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
    holds = []

    # Diagnóstico de edge
    entered_ev = []

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

            # filtros rolling sin futuro
            if score[i] < thr_score[i]:
                dbg["gate_score"] += 1
                continue
            if iqr[i] > thr_ood[i]:
                dbg["gate_ood"] += 1
                continue
            if not topk_mask[i]:
                dbg["gate_topk"] += 1
                continue

            # EV gate = max(roll, abs_min, buffer)
            ev_thr = max(float(thr_ev_roll[i]), ev_abs_min, ev_buffer)
            if best_ev[i] < ev_thr:
                dbg["gate_ev_buffer"] += 1
                continue

            # Quantile width gate: evita micro-movimientos
            if side_choice == 1:
                qwidth = float(max(qL90[i], 0.0) - min(qL10[i], 0.0))
            else:
                qwidth = float(max(qS90[i], 0.0) - min(qS10[i], 0.0))
            if adaptive_gating and (qwidth < q_width_thr):
                dbg["gate_q_width"] += 1
                continue

            # Regime/Event policy
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

            dbg["can_enter"] += 1
            dbg["entered"] += 1
            if side_choice == 1:
                dbg["entered_long"] += 1
            else:
                dbg["entered_short"] += 1

            in_pos = True
            side = side_choice
            entry_px = px
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
                if side == 1:
                    notional = (ex / (entry_px + EPS)) - 1.0
                else:
                    notional = (entry_px - ex) / (entry_px + EPS)
                trade_ret = (notional * leverage * risk_fraction) - (cost_rt_notional * leverage * risk_fraction)
                rets.append(trade_ret)
                equity.append(equity[-1] * (1.0 + trade_ret))
                holds.append(bars_in)
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
                trade_ret = (notional * leverage * risk_fraction) - (cost_rt_notional * leverage * risk_fraction)
                rets.append(trade_ret)
                equity.append(equity[-1] * (1.0 + trade_ret))
                holds.append(bars_in)
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
            if side == 1:
                notional = (sl_px / (entry_px + EPS)) - 1.0
            else:
                notional = (entry_px - sl_px) / (entry_px + EPS)
            trade_ret = (notional * leverage * risk_fraction) - (cost_rt_notional * leverage * risk_fraction)
            rets.append(trade_ret)
            equity.append(equity[-1] * (1.0 + trade_ret))
            holds.append(bars_in)
            in_pos = False
            cooldown = cooldown_bars
            dbg["exit_sl"] += 1
            continue

        if tp_hit:
            if side == 1:
                notional = (tp_px / (entry_px + EPS)) - 1.0
            else:
                notional = (entry_px - tp_px) / (entry_px + EPS)
            trade_ret = (notional * leverage * risk_fraction) - (cost_rt_notional * leverage * risk_fraction)
            rets.append(trade_ret)
            equity.append(equity[-1] * (1.0 + trade_ret))
            holds.append(bars_in)
            in_pos = False
            cooldown = cooldown_bars
            dbg["exit_tp"] += 1
            continue

    r = np.asarray(rets, dtype=np.float64)

    diag = {}

    # Aplicar costos detallados si exec_cfg está definido
    if exec_cfg is not None:
        atr_rel = np.mean(atr[entry_i] / entry_px) if entry_i < len(atr) else None
        r_net, (fee, spread, slip) = apply_costs(r, len(r), exec_cfg, atr_rel)
        # Guardar métricas de costos
        diag["fee_per_trade"] = float(fee)
        diag["spread_per_trade"] = float(spread)
        diag["slip_per_trade"] = float(slip)
        diag["trade_ret_raw"] = r.copy()
        diag["trade_ret_net"] = r_net.copy()
        r = r_net

    eq = np.asarray(equity, dtype=np.float64)

    n_trades = int(len(r))
    winrate = float((r > 0).mean()) if n_trades else 0.0

    metrics = {
        "n_trades": float(n_trades),
        "profit_factor": float(_profit_factor(r)) if n_trades else 0.0,
        "max_drawdown": float(_max_drawdown(eq)),
        "sortino": float(_sortino_proxy(r)) if n_trades else 0.0,
        "winrate": float(winrate),
        "equity_final": float(eq[-1]) if len(eq) else 1.0,
        "avg_ret_per_trade": float(r.mean()) if n_trades else 0.0,
        "median_ret_per_trade": float(np.median(r)) if n_trades else 0.0,
    }

    diag.update({
        "n_trades": n_trades,
        "mean_ret": float(r.mean()) if n_trades else 0.0,
        "cost_rt_base": float(dbg["cost_rt_base"]),
        "cost_rt_notional": float(cost_rt_notional),
        "avg_hold_bars": float(np.mean(holds)) if holds else 0.0,
        "entered_ev_mean": float(np.mean(entered_ev)) if entered_ev else 0.0,
        "entered_ev_median": float(np.median(entered_ev)) if entered_ev else 0.0,
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
