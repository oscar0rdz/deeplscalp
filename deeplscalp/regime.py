# deeplscalp/regime.py
import numpy as np

def multi_scale_regime_features(logret: np.ndarray, horizons=(16, 64, 256)) -> np.ndarray:
    """
    Devuelve matriz [n, len(horizons)*2]:
    - vol_ewm por horizonte
    - trend (slope proxy) por horizonte
    """
    x = np.asarray(logret, dtype=float)
    n = len(x)
    feats = []

    for h in horizons:
        a = 2.0 / (h + 1.0)
        v = 0.0
        m = 0.0
        vol = np.empty(n, dtype=float)
        mom = np.empty(n, dtype=float)
        for i in range(n):
            m = (1-a)*m + a*x[i]
            v = (1-a)*v + a*(x[i]**2)
            vol[i] = np.sqrt(max(v, 1e-12))
            mom[i] = m
        feats.append(vol)
        feats.append(mom)

    return np.vstack(feats).T
