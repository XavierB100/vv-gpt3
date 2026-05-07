#!/bin/bash
# ============================================================
# VV-GPT3 — macOS Launcher
# Double-click this file in Finder to start the app.
# ============================================================

# Change into the script's own directory (so paths are always correct)
cd "$(dirname "$0")"

echo ""
echo "=================================================="
echo "  🤖  VV-GPT3 — macOS Launcher"
echo "=================================================="
echo ""

# --- Python: prefer venv, then system python3 ---
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    echo "✅  Using virtual environment (.venv)"
elif [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
    echo "✅  Using virtual environment (venv)"
else
    PYTHON=$(which python3)
    echo "ℹ️   Using system Python: $PYTHON"
fi

# --- Check Python is available ---
if [ -z "$PYTHON" ]; then
    echo "❌  Python 3 not found. Please install it from https://www.python.org"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "🐍  Python: $($PYTHON --version)"
echo ""

# --- Install dependencies if not yet installed ---
if ! $PYTHON -c "import flask" &>/dev/null; then
    echo "📦  Installing required packages (first run only)..."
    $PYTHON -m pip install -r requirements.txt
    echo ""
fi

# --- Launch the app ---
echo "🚀  Starting VV-GPT3..."
echo "--------------------------------------------------"
$PYTHON desktop_launcher.py

echo ""
echo "=================================================="
echo "  VV-GPT3 has stopped."
echo "=================================================="
read -p "Press Enter to close this window..."
