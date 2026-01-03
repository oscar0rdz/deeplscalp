# evaluation/policy_gates.py
import numpy as np

def _finite(x):
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]

def qtile(x, p, default=0.0):
    x = _finite(x)
    if x.size == 0:
        return float(default)
    return float(np.quantile(x, float(p)))

def gate_report(tag, eligible_mask, ood_mask, entry_mask, trades_n=None):
    n = int(len(eligible_mask))
    elig = int(np.sum(eligible_mask))
    pass_ood = int(np.sum(eligible_mask & ood_mask))
    pass_entry = int(np.sum(eligible_mask & ood_mask & entry_mask))
    def pct(a, b):
        return 0.0 if b == 0 else (100.0 * a / b)
    msg = (
        f"[gates:{tag}] n={n} eligible={elig} ({pct(elig,n):.2f}%) | "
        f"pass_ood={pass_ood} ({pct(pass_ood,elig):.2f}%) | "
        f"pass_entry={pass_entry} ({pct(pass_entry,pass_ood):.2f}%)"
    )
    if trades_n is not None:
        msg += f" | n_trades={int(trades_n)}"
    print(msg)

def thresholds_from_val(val_pred, pct_cfg):
    """
    val_pred: dict con arrays: q50, q10, p_sl, ood (si existen)
    pct_cfg: dict con percentiles (0..1)
    """
    q50 = val_pred.get("q50", None)
    q10 = val_pred.get("q10", None)
    psl = val_pred.get("p_sl", None)
    ood = val_pred.get("ood", None)

    thr = {}

    # Entrada: q50 alto (top quantile) y q10 no tan malo (quantile más “alto” en q10)
    thr["enter_q50_min"] = qtile(q50, pct_cfg["enter_q50_pct"], default=0.0) if q50 is not None else 0.0
    thr["enter_q10_min"] = qtile(q10, pct_cfg["enter_q10_pct"], default=-1e9) if q10 is not None else -1e9

    # OOD: permitir hasta cierto percentil del score OOD (menos restrictivo)
    # Convención típica: ood_score mayor => más outlier => filtras si ood_score > umbral
    thr["ood_q"] = qtile(ood, pct_cfg["ood_pct"], default=np.inf) if ood is not None else np.inf

    # Salida: cuando q50 cae bajo cierto umbral (percentil bajo de q50)
    thr["exit_q50_below"] = qtile(q50, pct_cfg["exit_q50_pct"], default=-np.inf) if q50 is not None else -np.inf

    # Salida por riesgo: si p_sl se dispara (percentil alto de p_sl)
    thr["exit_p_sl_max"] = qtile(psl, pct_cfg["exit_p_sl_pct"], default=np.inf) if psl is not None else np.inf

    return thr
