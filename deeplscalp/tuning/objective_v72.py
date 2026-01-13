import math


def robust_objective_v2(
    pf: float,
    net_return: float,
    mdd: float,
    ntr: int,
    turnover: float,
    avg_hold_bars: float,
    pf_cap: float = 50.0,
    ntr_min: int = 150,
    hold_min_bars: float = 1.0,
    lam_mdd: float = 2.0,
    lam_turn: float = 0.05,
    beta_ntr: float = 0.002,
    gamma_hold: float = 0.2,
) -> float:
    """
    PF por sí solo se satura. Aquí metemos:
    - log1p(PF cap alto) + net_return (ya con costos)
    - penalización por MDD
    - penalización por turnover (sobre-operar)
    - penalización por pocos trades
    - penalización por trades demasiado cortos (proxy de fills irreales)
    """
    pf = max(0.0, float(pf))
    pf = min(pf, float(pf_cap))
    mdd = max(0.0, float(mdd))
    ntr = int(ntr)
    net_return = float(net_return)
    turnover = max(0.0, float(turnover))
    avg_hold_bars = max(0.0, float(avg_hold_bars))

    score = math.log1p(pf)
    score += 5.0 * net_return           # aumenta el peso del retorno neto
    score -= lam_mdd * mdd
    score -= lam_turn * turnover

    if ntr < ntr_min:
        score -= beta_ntr * float(ntr_min - ntr)

    if avg_hold_bars < hold_min_bars:
        score -= gamma_hold * float(hold_min_bars - avg_hold_bars)

    return float(score)
