#!/usr/bin/env bash
set -euo pipefail

DATADIR="./data/freqtrade/binance"

freqtrade convert-data \
  --format-from json \
  --format-to parquet \
  --datadir "${DATADIR}"

echo "[OK] Convertido a Parquet en ${DATADIR}"

