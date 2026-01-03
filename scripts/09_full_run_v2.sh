#!/usr/bin/env bash
set -euo pipefail

cd /Users/oscarr/Desarrollo/Python0/DeepLScalp/DeepLScalp
source .venv/bin/activate

caffeinate -dimsu python -u -m evaluation.run_full_pipeline_event_v2 --config configs/full/full_pipeline_event_v2.yaml 2>&1 | tee reports/full_run_v2/run.log
