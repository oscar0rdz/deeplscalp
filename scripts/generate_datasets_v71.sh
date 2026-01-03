#!/bin/bash

# Script para generar datasets V7.1 para múltiples pares
# Uso: bash scripts/generate_datasets_v71.sh

PAIRS=("XRP/USDT" "SOL/USDT" "ETH/USDT" "DOGE/USDT" "BTC/USDT")
CONFIG="configs/v71_cpu_pro_stress.yaml"

echo "Generando datasets V7.1 para múltiples pares..."

for PAIR in "${PAIRS[@]}"; do
    echo "=== Procesando $PAIR ==="

    # Crear config temporal para este par
    TEMP_CONFIG="/tmp/config_${PAIR//[\/:]/_}.yaml"
    cp "$CONFIG" "$TEMP_CONFIG"

    # Actualizar el par en la config temporal
    sed -i.bak "s|XRP/USDT|$PAIR|g" "$TEMP_CONFIG"

    # Ejecutar pipeline
    caffeinate -dimsu bash -lc "
    PYTHONPATH=\$(pwd) python pipeline.py --config $TEMP_CONFIG build 2>&1 | tee artifacts/log_build_${PAIR//[\/:]/_}.txt
    "

    # Limpiar config temporal
    rm -f "$TEMP_CONFIG" "$TEMP_CONFIG.bak"

    echo "=== $PAIR completado ==="
done

echo "Todos los datasets generados!"
