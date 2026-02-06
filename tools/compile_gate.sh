#!/usr/bin/env bash
set -euo pipefail
python3 -m py_compile \
  deeplscalp/modeling/train_v71.py \
  scripts/run_v71_walkforward.py
echo "[OK] compile gate ✅"
