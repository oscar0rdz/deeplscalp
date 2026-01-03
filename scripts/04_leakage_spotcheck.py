import pandas as pd
import numpy as np

PATH = "artifacts/datasets/nf_ETH_USDT_5m.parquet"

df = pd.read_parquet(PATH, engine="pyarrow").sort_values(["unique_id", "ds"]).reset_index(drop=True)

# Elegimos una feature típica y vemos si coincide exactamente con una versión desplazada hacia atrás (sospechoso).
# Esto no prueba todo, pero detecta fugas obvias.
candidates = [c for c in df.columns if c not in ("unique_id","ds","y","tbm_hit")]
sample_cols = candidates[:25]

uid = df["unique_id"].iloc[0]
g = df[df["unique_id"]==uid].copy()

print("series:", uid, "rows:", len(g))

for c in sample_cols:
    x = g[c].to_numpy()
    # correlación con el futuro inmediato (si es ~1 sin razón, sospechoso)
    if len(x) > 10:
        corr = np.corrcoef(x[:-1], x[1:])[0,1]
        if np.isfinite(corr) and abs(corr) > 0.9999:
            print("SUSPICIOUS near-perfect lag corr:", c, corr)

print("[OK] spotcheck complete")
