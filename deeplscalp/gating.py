# deeplscalp/gating.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

@dataclass(frozen=True)
class GatingConfig:
    p_side_min: float
    score_q: float
    topk_frac: float
    ev_buffer: float
    q_width_mult: float
    ood_enable: bool = True
    ood_soft: bool = True  # <-- CLAVE: no matar colas por default

def apply_gating(
    cfg: GatingConfig,
    p_side: np.ndarray,
    score: np.ndarray,
    ev_net: np.ndarray,
    q_width: np.ndarray,
    ood_mask: np.ndarray | None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Devuelve mask de trades permitidos + contador de razones de rechazo.
    Todo vectorizado y testeable.
    """
    n = len(p_side)
    ok = np.ones(n, dtype=bool)
    reasons = {
        "gate_side": 0,
        "gate_score": 0,
        "gate_topk": 0,
        "gate_ev_buffer": 0,
        "gate_q_width": 0,
        "gate_ood": 0,
    }

    # 1) side confidence
    m = (p_side >= cfg.p_side_min)
    reasons["gate_side"] += int((~m).sum())
    ok &= m

    # 2) score quantile gate
    # Si score_q <= 0, asumimos sin gate
    if cfg.score_q > 0:
        thr = np.quantile(score, cfg.score_q) if n > 0 else 1.0
        m = (score >= thr)
        reasons["gate_score"] += int((~m).sum())
        ok &= m

    # 3) top-k fraction gate
    if cfg.topk_frac > 0 and n > 0:
        k = max(1, int(np.ceil(cfg.topk_frac * n)))
        # argpartition es eficiente para topk
        # cuidado si k >= n
        if k < n:
            idx = np.argpartition(score, -k)[-k:]
            m_tk = np.zeros(n, dtype=bool)
            m_tk[idx] = True
            reasons["gate_topk"] += int((~m_tk).sum())
            ok &= m_tk

    # 4) expected value buffer (net)
    # Solo aplica si ev_buffer > 0
    if cfg.ev_buffer > 0:
        m = (ev_net >= cfg.ev_buffer)
        reasons["gate_ev_buffer"] += int((~m).sum())
        ok &= m

    # 5) q_width gate (control de incertidumbre)
    # q_width_mult es un multiplicador sobre mediana para evitar valores absurdos.
    # Si q_width_mult <= 0, desactivado
    if cfg.q_width_mult > 0 and n > 0:
        med = float(np.median(q_width)) if n > 0 else 0.0
        limit = (cfg.q_width_mult * med) if med > 0 else np.inf
        m = (q_width <= limit)
        reasons["gate_q_width"] += int((~m).sum())
        ok &= m

    # 6) OOD gate (soft por default)
    if cfg.ood_enable and ood_mask is not None:
        if cfg.ood_soft:
            # soft: solo bloquea OOD si además el EV net es malo (ev_net < 0)
            # Ojo: la logica "ev_net < 0" asume que ev_net esta bien escalado.
            m = ~(ood_mask & (ev_net < 0))
        else:
            # hard: bloquea todo OOD
            m = ~ood_mask
        reasons["gate_ood"] += int((~m).sum())
        ok &= m

    return ok, reasons
