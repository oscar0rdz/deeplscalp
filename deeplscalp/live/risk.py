# deeplscalp/live/risk.py
from dataclasses import dataclass

@dataclass(frozen=True)
class RiskLimits:
    max_pos_size_usd: float
    max_daily_loss_pct: float
    max_drawdown_pct: float
    max_leverage: float
    min_equity_usd: float

@dataclass
class RiskState:
    equity_usd: float
    day_start_equity_usd: float
    peak_equity_usd: float
    halted: bool = False
    reason: str = ""

def check_kill_switch(state: RiskState, limits: RiskLimits) -> None:
    if state.equity_usd < limits.min_equity_usd:
        state.halted = True
        state.reason = "min_equity"
        return
    dd = 1.0 - (state.equity_usd / max(1e-9, state.peak_equity_usd))
    day_loss = 1.0 - (state.equity_usd / max(1e-9, state.day_start_equity_usd))
    if dd >= limits.max_drawdown_pct:
        state.halted = True
        state.reason = "max_drawdown"
    if day_loss >= limits.max_daily_loss_pct:
        state.halted = True
        state.reason = "max_daily_loss"
