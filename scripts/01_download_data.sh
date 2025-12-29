#!/usr/bin/env bash
set -euo pipefail

EXCHANGE="binance"
DATADIR="./data/freqtrade/binance"

# Pares (incluye BTC como anchor)
PAIRS=("BTC/USDT" "ETH/USDT" "SOL/USDT" "DOGE/USDT" "XRP/USDT")

# Timeframes (1m para ejecución + 5m base + 1h/4h contexto)
TFS=("1m" "5m" "1h" "4h")

# Rango: formato YYYYMMDD-YYYYMMDD (ajústalo)
TIMERANGE="20210101-20251227"

mkdir -p "${DATADIR}"

# Nota: si pasas -c config.json, toma pares/exchange desde ahí.
# Aquí lo hacemos directo por CLI para evitar dependencia de config.
freqtrade download-data \
  --exchange "${EXCHANGE}" \
  --datadir "${DATADIR}" \
  --pairs "${PAIRS[@]}" \
  --timeframes "${TFS[@]}" \
  --timerange "${TIMERANGE}"

echo "[OK] Datos descargados en ${DATADIR}"
