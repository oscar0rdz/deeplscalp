# deeplscalp/sim/cost_model.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class CostModel:
    fee_bps: float = 4.0        # taker ~4 bps por lado (ajustable)
    slippage_bps: float = 1.5   # por lado
    spread_bps: float = 0.8     # roundtrip aproximado o por lado según modelo
    latency_bars: int = 1       # ejecución en la siguiente barra (live-like)

    def fee_roundtrip_ret(self) -> float:
        return 2.0 * self.fee_bps * 1e-4

    def slippage_roundtrip_ret(self) -> float:
        return 2.0 * self.slippage_bps * 1e-4

    def spread_roundtrip_ret(self) -> float:
        # si lo modelas como costo directo roundtrip
        return 1.0 * self.spread_bps * 1e-4

    def total_cost_ret(self) -> float:
        return self.fee_roundtrip_ret() + self.slippage_roundtrip_ret() + self.spread_roundtrip_ret()
