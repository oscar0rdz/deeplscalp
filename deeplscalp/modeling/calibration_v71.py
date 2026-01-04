# deeplscalp/modeling/calibration_v71.py
from __future__ import annotations

import numpy as np


def _safe_np(x, dtype=None):
    x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
    return x


def apply_temperature_multiclass(P: np.ndarray, T: float, eps: float = 1e-12) -> np.ndarray:
    """
    Aplica temperature scaling a probabilidades multiclass P (N,K).
    Usa log(P)/T y softmax estable.
    """
    P = _safe_np(P, np.float64)
    T = float(T)
    if not np.isfinite(T) or T <= 0:
        return P

    logits = np.log(np.clip(P, eps, 1.0))
    logits = logits / T
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / (ex.sum(axis=1, keepdims=True) + eps)


def fit_temperature_multiclass(
    P: np.ndarray,
    y: np.ndarray,
    t_grid: np.ndarray,
    eps: float = 1e-12,
):
    """
    Encuentra T que minimiza NLL en validación para un problema multiclass.
    Robusto a:
      - len(P) != len(y)
      - y fuera de rango
      - NaNs
    """
    P = _safe_np(P, np.float64)
    y = _safe_np(y, np.int64)
    t_grid = _safe_np(t_grid, np.float64)

    nP = int(P.shape[0])
    ny = int(y.shape[0])
    n = min(nP, ny)
    if n <= 0:
        return 1.0, float("inf")

    if nP != ny:
        print(f"[CAL] WARNING: len mismatch; trimming to n={n} (P={nP}, y={ny})")

    P = P[:n]
    y = y[:n]

    # limpia targets fuera de rango
    K = int(P.shape[1])
    y = np.clip(y, 0, K - 1)

    best_T = 1.0
    best_nll = float("inf")

    # si no hay grid, usa algo razonable
    if t_grid.size == 0:
        t_grid = np.linspace(0.5, 5.0, 19)

    for T in t_grid:
        PT = apply_temperature_multiclass(P, float(T), eps=eps)
        # NLL
        idx = np.arange(n, dtype=np.int64)
        p_true = np.clip(PT[idx, y], eps, 1.0)
        nll = -np.log(p_true).mean()
        if float(nll) < best_nll:
            best_nll = float(nll)
            best_T = float(T)

    return best_T, best_nll
