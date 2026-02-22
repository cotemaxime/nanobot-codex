#!/usr/bin/env bash
set -euo pipefail

# Nightly qmd refresh for Obsidian vault.
# - Uses nvm Node so qmd is on PATH in non-interactive shells.
# - Runs qmd update + qmd embed for the paan-vault collection.

export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [[ -s "$NVM_DIR/nvm.sh" ]]; then
  # shellcheck disable=SC1090
  source "$NVM_DIR/nvm.sh"
  nvm use 24 --silent >/dev/null 2>&1 || nvm use 22 --silent >/dev/null 2>&1 || true
fi

QMD_BIN="$(command -v qmd || true)"
if [[ -z "$QMD_BIN" ]]; then
  echo "ERROR: qmd not found on PATH (nvm may not be configured correctly)." >&2
  exit 127
fi

COLLECTION="${QMD_COLLECTION:-paan-vault}"

# Optional: keep logs
LOG_DIR="${QMD_REFRESH_LOG_DIR:-$HOME/.cache/qmd}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/qmd-nightly-refresh.log"

echo "[$(date -Is)] Starting qmd refresh (collection=$COLLECTION, qmd=$QMD_BIN)" | tee -a "$LOG_FILE"

"$QMD_BIN" update -c "$COLLECTION" 2>&1 | tee -a "$LOG_FILE"
"$QMD_BIN" embed -c "$COLLECTION" 2>&1 | tee -a "$LOG_FILE"

echo "[$(date -Is)] Done" | tee -a "$LOG_FILE"
