import argparse
import numpy as np
import pandas as pd

def edge_series(fwd: np.ndarray, cost_rt: float, mode: str) -> np.ndarray:
    if mode == "abs":
        return np.abs(fwd) - cost_rt
    if mode == "long":
        return np.maximum(fwd, 0.0) - cost_rt
    # short
    return np.maximum(-fwd, 0.0) - cost_rt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--horizons", default="6,12,18,24")
    ap.add_argument("--cost_rt", type=float, default=0.0013)
    ap.add_argument("--use_logret", action="store_true")
    ap.add_argument(
        "--mode",
        type=str,
        default="abs",
        choices=["abs", "long", "short"],
        help="Cómo medir edge: abs=|fwd|, long=max(fwd,0), short=max(-fwd,0)"
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet, columns=["close"])
    c = df["close"].astype("float64").values
    if args.use_logret:
        r1 = np.diff(np.log(c), prepend=np.log(c[0]))
    else:
        r1 = np.diff(c, prepend=c[0]) / (c + 1e-12)

    print(f"rows={len(df):,} close_nonnull={np.isfinite(c).mean()*100:.2f}%")
    print(f"cost_rt={args.cost_rt:.6f} (round-trip)")

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    for h in horizons:
        # forward return (log or simple)
        if args.use_logret:
            fwd = np.log(c[h:] / c[:-h])
        else:
            fwd = (c[h:] / c[:-h]) - 1.0

        # oracle: si supieras el signo perfecto (upper bound)
        e = edge_series(fwd, args.cost_rt, args.mode)
        mean_edge = float(np.mean(e))
        p_pos = float(np.mean(e > 0.0))

        # stats
        q = np.quantile(np.abs(fwd), [0.5, 0.8, 0.9, 0.95, 0.99])
        print(f"\n[h={h}] abs(fwd) quantiles: "
              f"p50={q[0]:.6f} p80={q[1]:.6f} p90={q[2]:.6f} p95={q[3]:.6f} p99={q[4]:.6f}")
        print(f"[h={h}] mode={args.mode} oracle mean(edge)={mean_edge:.6f} | P(edge>0)={p_pos*100:.2f}%")

        if mean_edge <= 0:
            print(f"[h={h}] WARNING: ni con 'predicción perfecta de signo' ganas en promedio con ese costo.")

    print("\nInterpretación:")
    print("- Si oracle_mean <= 0 en casi todos los h: el problema es costo/horizonte/ruido, no el modelo.")
    print("- Si oracle_mean > 0 para algún h: hay espacio; entonces labels/features/exits deben alinearse a ese h.")

if __name__ == "__main__":
    main()
