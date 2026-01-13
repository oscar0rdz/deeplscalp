import math

def robust_objective(
    pf,
    mdd,
    ntr,
    zero_loss,
    pf_cap=20.0,
    ntr_min=150,
    lam_mdd=2.0,
    beta_ntr=0.002,
    gamma_zero_loss=0.25,
):
    """
    Objetivo robusto para evitar "soluciones tramposas":
    - log1p(PF) para evitar dominancia por PF grandes
    - penaliza MDD
    - penaliza pocos trades
    - penaliza zero_loss si hay pocos trades (típicamente degeneración)
    """
    pf = max(0.0, float(pf))
    pf = min(pf, float(pf_cap))
    mdd = max(0.0, float(mdd))
    ntr = int(ntr)

    score = math.log1p(pf)
    score -= lam_mdd * mdd

    if ntr < ntr_min:
        score -= beta_ntr * float(ntr_min - ntr)

    if bool(zero_loss) and ntr < (2 * ntr_min):
        score -= float(gamma_zero_loss)

    return float(score)
