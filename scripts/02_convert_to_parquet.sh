#!/usr/bin/env bash
set -euo pipefail

DATADIR="./data/freqtrade"

freqtrade convert-data \
  --format-from feather \
  --format-to parquet \
  --datadir "${DATADIR}"

echo "[OK] Convertido a Parquet en ${DATADIR}"
