#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[INFO] Installing system deps (ffmpeg, v4l) ..."
if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y ffmpeg v4l-utils python3-venv python3-pip
else
  echo "[WARN] sudo not found. Install deps manually: ffmpeg v4l-utils python3-venv python3-pip"
fi

echo "[INFO] Creating venv..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel setuptools

echo "[INFO] Installing Python deps (numpy<2 pinned for OpenCV compatibility)..."
python -m pip install -r requirements.txt

echo
echo "[INFO] Done."
echo "[INFO] Start: ./start.sh"
echo "[INFO] Camera list: v4l2-ctl --list-devices"
