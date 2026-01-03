#!/usr/bin/env bash
set -euo pipefail

cd /Users/oscarr/Desarrollo/Python0/DeepLScalp/DeepLScalp
source .venv/bin/activate

python -m evaluation.run_full_pipeline --config configs/full/full_pipeline.yaml
