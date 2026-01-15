#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/.."

# Debe existir carpeta project/
if [ ! -d "project" ]; then
  echo "No existe carpeta project/ en $(pwd)"
  exit 1
fi

rm -f project.zip

# Excluye basura típica
zip -r project.zip project \
  -x "project/.git/*" \
  -x "project/**/__pycache__/*" \
  -x "project/**/.ipynb_checkpoints/*" \
  -x "project/artifacts/**" \
  -x "project/.venv/*" \
  -x "project/.mypy_cache/*" \
  -x "project/.pytest_cache/*"

echo "OK -> $(pwd)/project.zip"
