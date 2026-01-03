def round_trip_cost_rt(cfg: dict, hold_minutes: float = 0.0) -> float:
    v = cfg.get("venue", {}) or {}
    mode = str(v.get("mode", "perp")).lower()
    use_maker = bool(v.get("use_maker", False))

    fee_maker = float(v.get("fee_maker", 0.0002))
    fee_taker = float(v.get("fee_taker", 0.0006))
    slip_bps = float(v.get("slip_bps", 1.0))
    funding_bps_8h = float(v.get("funding_bps_8h", 0.0))

    fee = fee_maker if use_maker else fee_taker
    slip = slip_bps / 10000.0
    rt = 2.0 * (fee + slip)

    if mode == "perp" and funding_bps_8h != 0.0 and hold_minutes > 0:
        rt += (funding_bps_8h / 10000.0) * (hold_minutes / (8.0 * 60.0))

    return float(rt)
