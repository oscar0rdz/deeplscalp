from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class PFStats:
    pf_raw: float
    pf_capped: float
    gross_profit: float
    gross_loss: float
    pf_inflated: bool

def equity_curve_from_pnl(pnl: np.ndarray, mode: str = "compound", start: float = 1.0) -> np.ndarray:
    pnl = np.asarray(pnl, dtype=float)
    pnl = np.nan_to_num(pnl, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "additive":
        eq = start + np.cumsum(pnl)
        return eq
    elif mode == "compound":
        eq = np.empty_like(pnl, dtype=float)
        v = float(start)
        for i, r in enumerate(pnl):
            v *= (1.0 + float(r))
            eq[i] = v
        return eq
    else:
        raise ValueError(f"equity_mode inválido: {mode}. Usa 'compound' o 'additive'.")

def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    # Evitar division por cero si peak es 0
    dd = 1.0 - (equity / np.maximum(peak, 1e-12))
    return float(np.max(dd))

def profit_factor_from_pnl(
    pnl: np.ndarray,
    pf_cap: float = 10.0,
    gross_loss_warn: float = 1e-3
) -> PFStats:
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    gp = float(pnl[pnl > 0].sum())
    gl = float(abs(pnl[pnl < 0].sum())) # Ensure positive magnitude
    
    if gl <= 0.0:
        pf_raw = float("inf")
    else:
        pf_raw = gp / gl

    pf_capped = float(min(pf_raw, pf_cap)) if np.isfinite(pf_raw) else float(pf_cap)
    
    # Flag if generic inflation condition met (low loss, some profit)
    pf_inflated = bool(gl < gross_loss_warn and gp > 0.0)
    
    return PFStats(pf_raw=pf_raw, pf_capped=pf_capped, gross_profit=gp, gross_loss=gl, pf_inflated=pf_inflated)

def sortino_ratio(pnl: np.ndarray, eps: float = 1e-12) -> float:
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    if pnl.size == 0:
        return 0.0
    downside = pnl[pnl < 0]
    if downside.size == 0:
        return float("inf") # O un valor alto capado like 100.0? 'inf' es correcto matematicamente.
    dd = float(np.sqrt(np.mean(downside ** 2)) + eps)
    mu = float(np.mean(pnl))
    return float(mu / dd)

def compute_metrics_from_trades(
    trades: pd.DataFrame,
    pnl_col: str = "pnl_net",
    equity_mode: str = "compound",
    pf_cap: float = 10.0
) -> Dict[str, Any]:
    if trades is None or len(trades) == 0:
        return {
            "net": 0.0, "mdd": 0.0,
            "profit_factor_raw": 0.0, "profit_factor": 0.0, "pf": 0.0,
            "n_trades": 0, "winrate": 0.0, "sortino": 0.0,
            "equity_final": 1.0, "avg_ret_per_trade": 0.0, "median_ret_per_trade": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0,
            "flags": {}
        }

    if pnl_col not in trades.columns:
        raise KeyError(f"compute_metrics_from_trades: falta columna '{pnl_col}' en trades: {list(trades.columns)}")

    pnl = pd.to_numeric(trades[pnl_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    eq = equity_curve_from_pnl(pnl, mode=equity_mode, start=1.0)

    pf = profit_factor_from_pnl(pnl, pf_cap=pf_cap, gross_loss_warn=1e-3)
    mdd = max_drawdown(eq)
    eq_final = float(eq[-1]) if eq.size else 1.0
    net = float(eq_final - 1.0)

    pos = int(np.sum(pnl > 0))
    n = int(len(pnl))
    winrate = float(pos / max(1, n))

    flags = {}
    if pf.pf_inflated:
        flags["pf_inflated"] = f"pf_raw={pf.pf_raw:.2f} gross_loss={pf.gross_loss:.4g}"
    
    # Check for NaN consistency in critical metrics
    if not np.isfinite(net) or not np.isfinite(mdd):
        flags["nan_guard"] = f"net={net} mdd={mdd}"

    return {
        "net": net,
        "mdd": mdd,
        "profit_factor_raw": float(pf.pf_raw if np.isfinite(pf.pf_raw) else pf_cap),
        "profit_factor": float(pf.pf_capped),
        "pf": float(pf.pf_capped),
        "gross_profit": float(pf.gross_profit),
        "gross_loss": float(pf.gross_loss),
        "n_trades": n,
        "winrate": winrate,
        "sortino": float(sortino_ratio(pnl)),
        "equity_final": eq_final,
        "avg_ret_per_trade": float(np.mean(pnl)),
        "median_ret_per_trade": float(np.median(pnl)),
        "flags": flags,
    }
