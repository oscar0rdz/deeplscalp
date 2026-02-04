# deeplscalp/reporting/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd


PREFER_PNL_COLS = [
    "pnl_net",     # preferido (net real)
    "ret_net",     # net en returns
    "pnl",         # fallback
    "ret_raw",     # último recurso
]

PREFER_EQUITY_COLS = ["equity", "balance", "nav", "value", "curve"]


def pick_first_col(df: pd.DataFrame, cands) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None


def compute_profit_factor(pnl: np.ndarray) -> Tuple[float, float, float]:
    pnl = pnl[np.isfinite(pnl)]
    gp = float(pnl[pnl > 0].sum())
    gl = float((-pnl[pnl < 0]).sum())
    if gl <= 1e-12:
        return float("inf"), gp, gl
    return gp / gl, gp, gl


def compute_equity_from_returns(ret: np.ndarray, start: float = 1.0) -> np.ndarray:
    """
    Compounding equity: eq[t+1] = eq[t] * (1 + ret_t)
    """
    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    eq = np.empty(len(ret) + 1, dtype=float)
    eq[0] = start
    for i, r in enumerate(ret):
        eq[i + 1] = eq[i] * (1.0 + float(r))
    return eq


def compute_mdd(equity: np.ndarray) -> float:
    equity = equity[np.isfinite(equity)]
    if len(equity) < 2:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    dd = 1.0 - (equity / peak)
    return float(np.max(dd))


@dataclass
class FoldMetrics:
    n_trades: int
    winrate: float
    net: float
    equity_final: float
    mdd: float
    pf_raw: float
    gross_profit: float
    gross_loss: float
    pnl_col_used: str
    equity_col_used: Optional[str]
    flags: Dict[str, str]


def compute_fold_metrics(trades: pd.DataFrame, equity: Optional[pd.DataFrame] = None) -> FoldMetrics:
    flags: Dict[str, str] = {}

    if trades is None or len(trades) == 0:
        return FoldMetrics(
            n_trades=0, winrate=float("nan"), net=float("nan"), equity_final=float("nan"),
            mdd=float("nan"), pf_raw=float("nan"), gross_profit=float("nan"), gross_loss=float("nan"),
            pnl_col_used="(none)", equity_col_used=None, flags={"empty": "no trades"}
        )

    pnl_col = pick_first_col(trades, PREFER_PNL_COLS)
    if pnl_col is None:
        pnl_col = trades.columns[0]
        flags["pnl_col_fallback"] = f"using {pnl_col}"

    pnl = pd.to_numeric(trades[pnl_col], errors="coerce").fillna(0.0).astype(float).values
    n = int(len(pnl))
    winrate = float(np.mean(pnl > 0)) if n else float("nan")
    pf_raw, gp, gl = compute_profit_factor(pnl)

    # equity: si se pasa equity dataframe, úsala; si no, construye por compounding desde pnl como retorno
    equity_col = None
    if equity is not None and len(equity) > 0:
        equity_col = pick_first_col(equity, PREFER_EQUITY_COLS)
        if equity_col is None:
            flags["equity_missing_col"] = "no equity column found"
            eq_arr = compute_equity_from_returns(pnl, start=1.0)
        else:
            arr = pd.to_numeric(equity[equity_col], errors="coerce").astype(float).values
            arr = arr[np.isfinite(arr)]
            if len(arr) < 2:
                flags["equity_too_short"] = "equity has <2 finite points"
                eq_arr = compute_equity_from_returns(pnl, start=1.0)
            else:
                eq_arr = arr
    else:
        eq_arr = compute_equity_from_returns(pnl, start=1.0)

    equity_final = float(eq_arr[-1]) if len(eq_arr) else float("nan")
    net = equity_final - float(eq_arr[0]) if len(eq_arr) else float("nan")
    mdd = compute_mdd(eq_arr)

    # banderas de costos sospechosos
    if "fee" in trades.columns:
        fee_sum = float(pd.to_numeric(trades["fee"], errors="coerce").fillna(0).sum())
        if n >= 50 and fee_sum <= 1e-6:
            flags["fee_suspicious"] = f"fee_sum≈0 ({fee_sum:.3g})"

    if "slippage" in trades.columns:
        slip_sum = float(pd.to_numeric(trades["slippage"], errors="coerce").fillna(0).sum())
        if n >= 50 and slip_sum <= 1e-6:
            flags["slippage_suspicious"] = f"slippage_sum≈0 ({slip_sum:.3g})"

    if pf_raw > 50 and gl < 0.02:
        flags["pf_inflated"] = f"pf_raw={pf_raw:.2f} gross_loss={gl:.4f}"

    if mdd < 0.002 and n >= 80:
        flags["mdd_too_small"] = f"mdd={mdd:.4f} with n_trades={n}"

    return FoldMetrics(
        n_trades=n,
        winrate=winrate,
        net=net,
        equity_final=equity_final,
        mdd=mdd,
        pf_raw=pf_raw,
        gross_profit=gp,
        gross_loss=gl,
        pnl_col_used=pnl_col,
        equity_col_used=equity_col,
        flags=flags,
    )
