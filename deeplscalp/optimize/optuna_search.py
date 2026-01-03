import numpy as np
import optuna
from deeplscalp.utils.costs import get_cost_rt, get_min_edge_rt


def _as_dict(x):
    """Convierte None u otros tipos a dict vacío para evitar crashes por YAML null."""
    return x if isinstance(x, dict) else {}


def _thresholds_from_percentiles(pred_df, params, cfg):
    # V5 score EV
    cost_rt = float(get_cost_rt(cfg))
    p_tp = pred_df["p_tp"].astype(float).values
    p_sl = pred_df["p_sl"].astype(float).values
    p_time = pred_df["p_time"].astype(float).values
    q10 = pred_df["q10_net"].astype(float).values
    q50 = pred_df["q50_net"].astype(float).values
    q90 = pred_df["q90_net"].astype(float).values
    iqr = (pred_df["iqr_net"] if "iqr_net" in pred_df.columns else pred_df["iqr"]).astype(float).values
    risk = np.maximum(iqr, 1e-6)

    # retorno esperado "time"
    r_time = np.clip(q50, -0.02, 0.02)
    ev_gross = (p_tp * np.maximum(q90, 0.0)) + (p_sl * np.minimum(q10, 0.0)) + (p_time * r_time)
    ev_net = ev_gross - cost_rt
    score = ev_net / risk

    # iqr for ood
    iqr_series = iqr

    # Check for debug override in cfg
    ocfg = cfg.get("optuna", {})
    if "score_pct" in ocfg:
        score_min = float(ocfg["score_pct"])
        ood_thr = float(ocfg["ood_pct"])
        p_tp_min = float(ocfg["p_tp_min"])
        p_sl_max = float(ocfg["p_sl_max"])
        ev_min = float(ocfg["ev_min"])
        top_k = int(ocfg["top_k"])
        lambda_iqr = 0.0  # dummy
        cooldown_bars = 0  # dummy
    else:
        score_min = float(np.quantile(score, float(params["score_pct"])))
        ood_thr   = float(np.quantile(iqr_series, float(params["ood_pct"])))
        p_tp_min = float(params["p_tp_min"])
        p_sl_max = float(params["p_sl_max"])
        ev_min = float(params["ev_min"])
        top_k = int(params["top_k"])
        lambda_iqr = float(params["lambda_iqr"])
        cooldown_bars = int(params["cooldown_bars"])

    return {
        "lambda_iqr": lambda_iqr,
        "top_k": top_k,
        "score_min": score_min,
        "ood_thr": ood_thr,
        "p_tp_min": p_tp_min,
        "p_sl_max": p_sl_max,
        "ev_min": ev_min,
        "cooldown_bars": cooldown_bars,
    }


def _gate_audit(pred_df, thresholds):
    # V5 gate audit
    cost_rt = float(get_cost_rt({}))  # dummy cfg
    p_tp = pred_df["p_tp"].astype(float).values
    p_sl = pred_df["p_sl"].astype(float).values
    p_time = pred_df["p_time"].astype(float).values
    q10 = pred_df["q10_net"].astype(float).values
    q50 = pred_df["q50_net"].astype(float).values
    q90 = pred_df["q90_net"].astype(float).values
    iqr = (pred_df["iqr_net"] if "iqr_net" in pred_df.columns else pred_df["iqr"]).astype(float).values
    risk = np.maximum(iqr, 1e-6)

    r_time = np.clip(q50, -0.02, 0.02)
    ev_gross = (p_tp * np.maximum(q90, 0.0)) + (p_sl * np.minimum(q10, 0.0)) + (p_time * r_time)
    ev_net = ev_gross - cost_rt
    score = ev_net / risk

    score_min = float(thresholds["score_min"])
    ood_thr   = float(thresholds["ood_thr"])
    p_tp_min  = float(thresholds["p_tp_min"])
    p_sl_max  = float(thresholds["p_sl_max"])
    ev_min    = float(thresholds["ev_min"])

    pass_score = int(np.sum(score >= score_min))
    pass_ood   = int(np.sum(iqr <= ood_thr))
    pass_ptp   = int(np.sum(p_tp >= p_tp_min))
    pass_psl   = int(np.sum(p_sl <= p_sl_max))
    pass_ev    = int(np.sum(ev_net >= ev_min))
    pass_all   = int(np.sum((score >= score_min) & (iqr <= ood_thr) & (p_tp >= p_tp_min) & (p_sl <= p_sl_max) & (ev_net >= ev_min)))
    return {"pass_score": pass_score, "pass_ood": pass_ood, "pass_ptp": pass_ptp, "pass_psl": pass_psl, "pass_ev": pass_ev, "pass_all": pass_all}


def _profit_factor(returns: np.ndarray) -> float:
    """
    PF real (sin cap):
    PF = gross_profit / abs(gross_loss)
    - Si no hay pérdidas: PF = +inf
    - Si no hay ganancias: PF = 0
    """
    r = np.asarray(returns, dtype=np.float64)
    pos = r[r > 0].sum()
    neg = r[r < 0].sum()  # negativo
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / abs(neg))


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = 1.0 - (equity / (peak + 1e-12))
    return float(np.max(dd)) if len(dd) else 0.0


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except:
        return default


def score_objective(metrics, quick_mode=False):
    n = int(metrics.get("n_trades", 0))
    pf = _safe_float(metrics.get("profit_factor", 0.0), 0.0)
    mdd = _safe_float(metrics.get("max_drawdown", 1.0), 1.0)
    net = _safe_float(metrics.get("net_ret", 0.0), 0.0)

    # sanea PF degenerado
    if not np.isfinite(pf):
        pf = 0.0
    pf = max(0.0, min(pf, 5.0))

    # score base estable
    base = net - 0.6*mdd

    # penalización por pocos trades (MUCHO más fuerte)
    min_tr = 30 if not quick_mode else 12
    short = max(0, min_tr - n)
    penalty = 0.25 * short   # 0.25 por trade faltante

    return float(base - penalty)


def _safe_log(x: float, eps: float = 1e-12) -> float:
    return float(np.log(max(float(x), eps)))


def _score(metrics: dict, cfg: dict) -> float:
    """
    Score para Optuna orientado a scalping:
    - Penaliza fuerte 0 trades (no es estrategia).
    - Premia net, PF y estabilidad.
    - Fuerza un mínimo de actividad (trades y trades/día) sin volver todo -1000 constante.
    """
    bt_cfg = _as_dict(cfg.get("backtest"))
    c_cfg  = _as_dict(bt_cfg.get("constraints"))
    s_cfg  = _as_dict(bt_cfg.get("scoring"))

    n_trades = float(metrics.get("n_trades", 0.0))
    pf       = float(metrics.get("profit_factor", 0.0))
    mdd      = float(metrics.get("max_drawdown", 0.0))
    net      = float(metrics.get("net_ret", 1.0))
    wr       = float(metrics.get("winrate", 0.0))

    # Muchos backtests guardan tpd como trades_per_day o tpd; soportamos ambos
    tpd = float(metrics.get("trades_per_day", metrics.get("tpd", 0.0)))

    # Defaults (ajustables desde YAML)
    min_trades = float(c_cfg.get("min_trades", 30))
    max_mdd    = float(c_cfg.get("max_drawdown", 0.25))
    min_net    = float(c_cfg.get("min_net_ret", 0.995))  # exploración: no mates todo por poco

    pf_clamp   = float(s_cfg.get("pf_clamp_for_score", 5.0))

    # objetivos de actividad scalping
    tpd_min    = float(s_cfg.get("tpd_min", 2.0))
    tpd_target = float(s_cfg.get("tpd_target", 8.0))

    # pesos
    no_trade_penalty          = float(s_cfg.get("no_trade_penalty", 10000.0))
    trade_shortfall_penalty   = float(s_cfg.get("trade_shortfall_penalty", 120.0))
    tpd_penalty               = float(s_cfg.get("tpd_penalty", 250.0))
    tpd_bonus                 = float(s_cfg.get("tpd_bonus", 25.0))

    # 1) Nunca dejes que gane 0 trades
    if n_trades <= 0:
        return -no_trade_penalty

    # 2) PF clamped SOLO para score
    pf_for_score = min(pf, pf_clamp) if np.isfinite(pf) else pf_clamp

    # 3) Base score: net manda, pero con estabilidad
    score = 0.0
    score += (net - 1.0) * 2500.0          # net impacta fuerte
    score += pf_for_score * 120.0          # PF ayuda pero no domina
    score += wr * 50.0                     # winrate suma, con peso moderado
    score -= mdd * 900.0                   # drawdown penaliza fuerte

    # 4) Penaliza falta de actividad (sin matar todo)
    if n_trades < min_trades:
        score -= (min_trades - n_trades) * trade_shortfall_penalty

    # 5) Trades por día: para scalping necesitas flujo
    # premio por acercarte a target, castigo si estás por debajo del mínimo
    score += min(tpd, tpd_target) * tpd_bonus
    if tpd < tpd_min:
        score -= (tpd_min - tpd) * tpd_penalty

    # 6) Constraints "profesionales" como penalización dura (pero no todo a -1000)
    # Si quieres duro total, cambia a return -no_trade_penalty*2, etc.
    if mdd > max_mdd:
        score -= (mdd - max_mdd) * 20000.0
    if net < min_net:
        score -= (min_net - net) * 20000.0

    return float(score)





def optuna_find_thresholds(pred_df, cfg: dict, fold_id: int) -> dict:
    ocfg = cfg["optuna"]

    # Validación previa (fail-fast real)
    if bool(cfg["labels"].get("vol_scaled", False)) and "vol_scale" not in pred_df.columns:
        raise ValueError("vol_scaled=True pero pred_df no trae vol_scale. Debes adjuntar vol_scale a pred_df (VAL).")

    def objective(trial: optuna.Trial) -> float:
        thresholds = {}
        try:
            params = {
                "lambda_iqr": trial.suggest_float("lambda_iqr", 0.0, 1.0),
                "top_k": trial.suggest_int("top_k", 5, 20),
                "score_pct": trial.suggest_float("score_pct", 0.20, 0.80),
                "ood_pct": trial.suggest_float("ood_pct", 0.70, 0.98),
                "p_tp_min": trial.suggest_float("p_tp_min", 0.05, 0.30),
                "p_sl_max": trial.suggest_float("p_sl_max", 0.35, 0.70),
                "ev_min": trial.suggest_float("ev_min", 0.0, 0.0010),
                "cooldown_bars": trial.suggest_int("cooldown_bars", 0, 3),
            }
            thresholds = _thresholds_from_percentiles(pred_df, params, cfg)
            trial.set_user_attr("thresholds", thresholds)

            audit = _gate_audit(pred_df, thresholds)
            print(f"[gate_audit] {audit}")

            from deeplscalp.backtest.sim import backtest_from_predictions
            metrics, _ = backtest_from_predictions(pred_df, cfg, thresholds)

            n_tr = int(metrics.get("n_trades", 0))
            if n_tr == 0:
                return -5.0

            value = score_objective(metrics, quick_mode=False)

            trial.set_user_attr("val_metrics", metrics)

            print(f"[optuna] trades={n_tr} score={value:.3f}")
            return float(value)

        except Exception as e:
            # jamás dejar sin thresholds
            trial.set_user_attr("thresholds", thresholds)
            return -10000.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(ocfg["n_trials"]), gc_after_trial=True)

    valid_trials = [t for t in study.trials if (t.value is not None) and ("thresholds" in t.user_attrs)]
    if not valid_trials:
        raise RuntimeError("Optuna terminó sin trials válidos con thresholds. Revisa errores dentro de objective().")

    best_trial = max(valid_trials, key=lambda t: t.value)
    best_thresholds = best_trial.user_attrs["thresholds"]
    best_val_metrics = best_trial.user_attrs.get("val_metrics", None)

    return {"best_value": float(best_trial.value), "thresholds": best_thresholds, "val_metrics": best_val_metrics}
