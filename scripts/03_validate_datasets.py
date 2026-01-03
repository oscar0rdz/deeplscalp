import glob
import numpy as np
import pandas as pd

def check_file(path: str) -> None:
    df = pd.read_parquet(path, engine="pyarrow")

    # básicos
    assert "unique_id" in df.columns and "ds" in df.columns and "y" in df.columns and "tbm_hit" in df.columns
    assert df["ds"].notna().all(), "ds has NaN"
    assert df["unique_id"].notna().all(), "unique_id has NaN"

    # orden temporal por serie
    for uid, g in df.groupby("unique_id", sort=False):
        if not g["ds"].is_monotonic_increasing:
            raise ValueError(f"{path}: ds not monotonic for {uid}")

    # inf/nan en numéricas
    num = df.select_dtypes(include=[np.number])
    inf_count = np.isinf(num.to_numpy()).sum()
    nan_count = num.isna().sum().sum()

    # NaNs esperables: al inicio por rolling; pero no deberían ser masivos
    print(f"\n{path}")
    print("rows:", len(df), "cols:", len(df.columns))
    print("nan_count:", int(nan_count), "inf_count:", int(inf_count))

    # etiqueta
    vc = df["tbm_hit"].value_counts(dropna=False)
    print("tbm_hit:", vc.to_dict())

    # sanity: y debe ser finito
    if not np.isfinite(df["y"].to_numpy()).all():
        raise ValueError(f"{path}: y has non-finite values")

if __name__ == "__main__":
    files = sorted(glob.glob("artifacts/datasets/*.parquet"))
    if not files:
        raise SystemExit("No parquet files found in artifacts/datasets")
    for f in files:
        check_file(f)
    print("\n[OK] Validation finished.")
