from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PFResult:
    pf_raw: float
    pf_capped: float
    gross_profit: float
    gross_loss: float
    flag: str  # "ok", "no_losses", "no_trades"

def profit_factor(pnls: np.ndarray, cap: float = 20.0, eps: float = 1e-12) -> PFResult:
    pnls = np.asarray(pnls, dtype=float)
    if pnls.size == 0:
        return PFResult(0.0, 0.0, 0.0, 0.0, "no_trades")
    gp = float(np.sum(pnls[pnls > 0.0])) if np.any(pnls > 0.0) else 0.0
    gl = float(-np.sum(pnls[pnls < 0.0])) if np.any(pnls < 0.0) else 0.0

    if gl <= eps:
        # Sin pérdidas -> PF infinito en teoría.
        # Para evitar que el tuner "haga trampa", lo marcamos explícitamente.
        pf_raw = float("inf") if gp > 0 else 0.0
        pf_c = cap if gp > 0 else 0.0
        return PFResult(pf_raw, pf_c, gp, gl, "no_losses")

    pf_raw = gp / gl
    pf_c = min(pf_raw, cap)
    return PFResult(float(pf_raw), float(pf_c), gp, gl, "ok")

def max_drawdown(equity_curve: np.ndarray) -> float:
    e = np.asarray(equity_curve, dtype=float)
    if e.size == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    dd = (peak - e) / np.maximum(peak, 1e-12)
    return float(np.max(dd))
