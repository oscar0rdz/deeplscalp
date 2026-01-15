#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

rm -f ../project.zip

# Crea temp dir con project/
TEMP_DIR="$(mktemp -d)"
mkdir -p "$TEMP_DIR/project"

# Copia excluyendo basura
rsync -a --exclude='.git' --exclude='__pycache__' --exclude='.ipynb_checkpoints' --exclude='artifacts' --exclude='.venv' --exclude='.mypy_cache' --exclude='.pytest_cache' . "$TEMP_DIR/project/"

# Zip
cd "$TEMP_DIR"
zip -r ../project.zip .

# Limpia
cd ..
rm -rf "$TEMP_DIR"

echo "OK -> $(pwd)/project.zip"
