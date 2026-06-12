#!/usr/bin/env bash
# ============================================================
#  VaultRAG v2.0 - Linux / WSL Launcher
# ============================================================
set -e
cd "$(dirname "$0")"

if [ -f ".venv-wsl/bin/activate" ]; then
    source .venv-wsl/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "[INFO] No venv found. Creating..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "[INFO] Installing dependencies..."
    pip install -r requirements.txt
fi

export PYTHONPATH="$(pwd)"

echo ""
echo " ============================================"
echo "  VaultRAG v2.0 - Secure Offline RAG Server"
echo "  http://127.0.0.1:8000"
echo " ============================================"
echo ""

python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
