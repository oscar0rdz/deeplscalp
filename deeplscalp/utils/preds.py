import pandas as pd


def ensure_net_quantiles(pred_df: pd.DataFrame, vol_scaled: bool) -> pd.DataFrame:
    """
    Convierte q10/q50/q90 e iqr a escala neta (RT) si vol_scaled=True.
    Produce:
      - q10_net/q50_net/q90_net
      - iqr_net
    Para V6: maneja qL* y qS* por separado
    """
    if not vol_scaled:
        for c in ("q10", "q50", "q90", "iqr"):
            if c in pred_df.columns and f"{c}_net" not in pred_df.columns:
                pred_df[f"{c}_net"] = pred_df[c].astype(float)
        return pred_df

    if "vol_scale" not in pred_df.columns:
        raise ValueError("vol_scaled=True pero falta vol_scale en pred_df")

    vs = pred_df["vol_scale"].astype(float)
    for c in ("q10", "q50", "q90"):
        if c in pred_df.columns:
            pred_df[f"{c}_net"] = pred_df[c].astype(float) * vs

    if "iqr" in pred_df.columns:
        pred_df["iqr_net"] = pred_df["iqr"].astype(float) * vs

    return pred_df


def add_score(pred_df: pd.DataFrame, score_iqr_lambda: float) -> pd.DataFrame:
    """
    Score V6: mejor EV neto entre long y short
      score = max(EV_L_net, EV_S_net) - lambda * max(iqr_L, iqr_S)
    """
    # V6: calcular EV neto por lado
    cost_rt = 0.0013  # aproximado, debería venir de cfg

    if "qL50_gross" in pred_df.columns and "qS50_gross" in pred_df.columns:
        # V6 predictions
        evL_net = (pred_df["pL_tp"] * pred_df["qL90_gross"] +
                  pred_df["pL_sl"] * pred_df["qL10_gross"]) - cost_rt
        evS_net = (pred_df["pS_tp"] * pred_df["qS90_gross"] +
                  pred_df["pS_sl"] * pred_df["qS10_gross"]) - cost_rt

        best_ev = pd.concat([evL_net, evS_net], axis=1).max(axis=1)

        # Risk: max IQR entre lados
        if "iqrL_gross" in pred_df.columns and "iqrS_gross" in pred_df.columns:
            risk = pd.concat([pred_df["iqrL_gross"], pred_df["iqrS_gross"]], axis=1).max(axis=1)
        else:
            risk = pd.Series([1e-6] * len(pred_df), index=pred_df.index)

        pred_df["score"] = best_ev - 0.0 * risk  # TEMP: set lambda to 0 for testing

        # DEBUG: print some stats
        print(f"[DEBUG] EV_L_net > 0: {(evL_net > 0).sum()}")
        print(f"[DEBUG] EV_S_net > 0: {(evS_net > 0).sum()}")
        print(f"[DEBUG] Best EV > 0: {(best_ev > 0).sum()}")
        print(f"[DEBUG] Score > 0: {(pred_df['score'] > 0).sum()}")
        print(f"[DEBUG] Score mean: {pred_df['score'].mean():.6f}, std: {pred_df['score'].std():.6f}")

    elif "q50_net" in pred_df.columns:
        # V5 fallback
        iqr_col = "iqr_net" if "iqr_net" in pred_df.columns else ("iqr" if "iqr" in pred_df.columns else None)
        if iqr_col is None:
            pred_df["score"] = pred_df["q50_net"].astype(float)
        else:
            pred_df["score"] = pred_df["q50_net"].astype(float) - float(score_iqr_lambda) * pred_df[iqr_col].astype(float)
    else:
        raise ValueError("Faltan quantiles para calcular score (q50_net para V5 o qL*/qS* para V6)")

    return pred_df


def normalize_preds_for_trading(pred_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    vol_scaled = bool(cfg["labels"].get("vol_scaled", False))
    lam = float(cfg.get("execution", {}).get("score_iqr_lambda", 0.5))
    pred_df = ensure_net_quantiles(pred_df, vol_scaled=vol_scaled)
    pred_df = add_score(pred_df, score_iqr_lambda=lam)
    return pred_df
