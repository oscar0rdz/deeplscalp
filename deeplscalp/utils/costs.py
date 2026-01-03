def get_cost_rt(cfg: dict) -> float:
    """
    Devuelve costo round-trip (RT).
    Soporta:
      - backtest.cost_rt
      - backtest.costs.{fee_per_side, slippage_per_side, spread_per_side}
      - backtest.costs.{fee_rt, slippage_rt, spread_rt}  (ya en RT)
    """
    bt = cfg.get("backtest") or {}

    if bt.get("cost_rt") is not None:
        return float(bt["cost_rt"])

    costs = bt.get("costs") or {}

    # Esquema por-lado (lo convertimos a RT)
    if any(k in costs for k in ("fee_per_side", "slippage_per_side", "spread_per_side")):
        fee = float(costs.get("fee_per_side", 0.0))
        slip = float(costs.get("slippage_per_side", 0.0))
        spr = float(costs.get("spread_per_side", 0.0))
        return 2.0 * (fee + slip + spr)

    # Esquema RT (ya viene sumable)
    fee = float(costs.get("fee_rt", 0.0))
    slip = float(costs.get("slippage_rt", 0.0))
    spr = float(costs.get("spread_rt", 0.0))
    return fee + slip + spr


def get_min_edge_rt(cfg: dict) -> float:
    """
    Umbral mínimo de edge neto (RT) para no tradear ruido.
    """
    bt = cfg.get("backtest") or {}
    costs = bt.get("costs") or {}
    return float(costs.get("min_edge_rt", 0.0))
