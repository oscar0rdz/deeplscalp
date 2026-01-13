import numpy as np

from deeplscalp.backtest.sim_v71 import profit_factor_stats
from deeplscalp.tuning.objective_v71 import robust_objective


def test_profit_factor_finite_and_capped():
    r = np.array([0.01, 0.02, 0.03])  # sin pérdidas
    s = profit_factor_stats(r, pf_cap=20.0)
    assert np.isfinite(s.pf)
    assert s.pf <= 20.0
    assert s.zero_loss is True

def test_objective_penalizes_low_trades():
    # PF alto pero pocos trades => penaliza
    score_low = robust_objective(pf=20, mdd=0.0, ntr=20, zero_loss=True, ntr_min=150)
    score_ok  = robust_objective(pf=2.0, mdd=0.1, ntr=200, zero_loss=False, ntr_min=150)
    assert score_ok > score_low
