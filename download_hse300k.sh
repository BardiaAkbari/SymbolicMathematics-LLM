#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-hse_dif_equations_fine_tune}"
if [[ -d "$DEST/.git" ]]; then
  echo "Already cloned: $DEST"
else
  git clone --depth 1 https://github.com/hse-scila/dif_equations_fine_tune.git "$DEST"
fi
echo "Dataset repository: $DEST"
find "$DEST/data" -maxdepth 2 -type f 2>/dev/null | head -30 || true
