#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "[INFO] Cleaning Python caches..."
find "$ROOT" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true

echo "[INFO] Cleaning backups..."
find "$ROOT" -maxdepth 3 -type f -name "*.bak*" -delete 2>/dev/null || true

echo "[INFO] Cleaning logs..."
if [ -d "$ROOT/logs" ]; then
  find "$ROOT/logs" -type f -name "*.log" -delete 2>/dev/null || true
fi

echo "[INFO] Cleaning temp files..."
find "$ROOT" -maxdepth 1 -type f -name "tmp_*" -delete 2>/dev/null || true

echo "[OK] Cleanup complete."
