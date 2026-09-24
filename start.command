#!/bin/zsh
set -eu
cd -- "${0:A:h}"
if [ ! -x .venv/bin/python ]; then
  echo "First-time setup: install uv, then run uv sync --python 3.12 in this folder."
  exit 1
fi
echo "EEG Explorer is available at http://127.0.0.1:8765"
echo "Keep this window open. Press Control-C to stop."
exec .venv/bin/python -m uvicorn eeg_explorer.app:app --host 127.0.0.1 --port 8765
