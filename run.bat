@echo off
title Mental Stress Detection AI - Setup ^& Launch
echo ========================================
echo  Mental Stress Detection AI
echo  Auto-Setup ^& Launch Script
echo ========================================
echo.

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

:: Create venv if it doesn't exist
if not exist "venv\" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo       Done.
) else (
    echo [1/3] Virtual environment already exists. Skipping creation.
)

:: Use explicit venv paths (activate.bat is unreliable in scripts)
set VENV_PYTHON=venv\Scripts\python.exe
set VENV_PIP=venv\Scripts\python.exe -m pip
set VENV_STREAMLIT=venv\Scripts\streamlit.exe

:: Install / upgrade dependencies
echo [2/3] Installing dependencies (this may take several minutes on first run)...
%VENV_PIP% install --upgrade pip --quiet

:: Install PyTorch with CUDA 12.4 support for GPU (RTX 4060)
echo       [Step A] Installing PyTorch with CUDA GPU support...
%VENV_PIP% install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet

:: Install remaining dependencies
echo       [Step B] Installing other dependencies...
%VENV_PIP% install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check requirements.txt for issues.
    pause
    exit /b 1
)
echo       All dependencies installed successfully.
echo.

:: Launch Streamlit
echo ========================================
echo  Launching Streamlit app...
echo  Press Ctrl+C in this window to stop.
echo ========================================
echo.
%VENV_STREAMLIT% run chatbot_mega.py
