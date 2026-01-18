#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
PORT="${PORT:-8090}"
HOST="${HOST:-0.0.0.0}"

cd "$ROOT"
if [ ! -d "$VENV" ]; then
  echo "[ERR] venv missing at $VENV"
  echo "Run: ./scripts/install.sh"
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

LAN_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}' || true)"
echo
echo "[INFO] Time-Lapse Web Pro starting..."
echo "[INFO] Local: http://127.0.0.1:${PORT}"
if [ -n "${LAN_IP:-}" ]; then
  echo "[INFO] LAN:   http://${LAN_IP}:${PORT}"
fi
echo

exec python -m uvicorn "src.timelapse_web_pro:app" --host "$HOST" --port "$PORT"
