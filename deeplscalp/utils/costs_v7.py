def round_trip_cost_rt(cfg: dict) -> float:
    """
    Costo round-trip en NOTIONAL (sin leverage):
      - fee_rate: comisión por lado (taker) como fracción (0.0004 = 0.04%)
      - slippage: slippage por lado como fracción (0.0002 = 0.02%)

    Nota:
      - Este costo se usa en labels (para EV net) y en backtest.
      - En backtest se convierte a costo sobre equity multiplicando por leverage.
    """
    risk = cfg.get("risk") if isinstance(cfg.get("risk"), dict) else {}
    fee_rate = float(risk.get("fee_rate", 0.0004))
    slippage = float(risk.get("slippage", 0.0002))
    return 2.0 * fee_rate + 2.0 * slippage
