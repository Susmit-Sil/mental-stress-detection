#!/usr/bin/env bash
set -e

echo "========================================"
echo " Mental Stress Detection AI"
echo " Auto-Setup & Launch Script"
echo "========================================"
echo ""

# Check Python is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed or not in PATH."
    echo "        Please install Python 3.10+ first."
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv venv
    echo "      Done."
else
    echo "[1/3] Virtual environment already exists. Skipping creation."
fi

# Activate venv
echo "[2/3] Activating virtual environment..."
source venv/bin/activate

# Install / upgrade dependencies
echo "[3/3] Installing dependencies (this may take a few minutes on first run)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "      All dependencies installed successfully."
echo ""

# Launch Streamlit
echo "========================================"
echo " Launching Streamlit app..."
echo " Press Ctrl+C to stop."
echo "========================================"
echo ""
streamlit run chatbot_mega.py
