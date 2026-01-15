import math

import numpy as np


def profit_factor_raw(returns):
    pos = returns[returns > 0].sum()
    neg = -returns[returns < 0].sum()
    if neg == 0:
        return np.inf
    return float(pos / neg)


def pf_score_for_objective(pf_raw, n_trades, min_trades=80, pf_cap_display=100.0):
    # Penaliza "soluciones" con pocos trades
    if n_trades < min_trades:
        return -1.0 * (min_trades - n_trades)  # penalización lineal simple

    # Para objective: función suave (evita saturación dura)
    if np.isinf(pf_raw):
        return np.log1p(pf_cap_display)  # si realmente no tuvo pérdidas y hay suficientes trades
    return float(np.log1p(pf_raw))


def robust_objective_v2(
    pf: float,
    net_return: float,
    mdd: float,
    ntr: int,
    turnover: float,
    avg_hold_bars: float,
    pf_cap: float = 50.0,  # kept for compatibility, but not used for capping
    ntr_min: int = 150,
    hold_min_bars: float = 1.0,
    lam_mdd: float = 2.0,
    lam_turn: float = 0.05,
    beta_ntr: float = 0.002,
    gamma_hold: float = 0.2,
) -> float:
    """
    PF por sí solo se satura. Aquí metemos:
    - pf_score_for_objective (sin cap duro) + net_return (ya con costos)
    - penalización por MDD
    - penalización por turnover (sobre-operar)
    - penalización por pocos trades
    - penalización por trades demasiado cortos (proxy de fills irreales)
    """
    pf_raw = float(pf)  # assume pf is already raw or handle externally
    mdd = max(0.0, float(mdd))
    ntr = int(ntr)
    net_return = float(net_return)
    turnover = max(0.0, float(turnover))
    avg_hold_bars = max(0.0, float(avg_hold_bars))

    score = pf_score_for_objective(pf_raw, ntr, min_trades=ntr_min)
    score += 5.0 * net_return           # aumenta el peso del retorno neto
    score -= lam_mdd * mdd
    score -= lam_turn * turnover

    if avg_hold_bars < hold_min_bars:
        score -= gamma_hold * float(hold_min_bars - avg_hold_bars)

    return float(score)
