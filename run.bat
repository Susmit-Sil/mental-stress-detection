@echo off
title Mental Stress Detection AI
echo  Mental Stress Detection AI
echo  Start
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

if not exist "venv\" (
    echo [1/5] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo       Done.
) else (
    echo [1/5] Virtual environment already exists. Skipping creation.
)

set VENV_PYTHON=venv\Scripts\python.exe
set VENV_PIP=venv\Scripts\python.exe -m pip
set VENV_STREAMLIT=venv\Scripts\streamlit.exe

set PYTHONPATH=

echo [2/5] Installing dependencies...
%VENV_PIP% install --upgrade pip --quiet

echo       [Step A] Installing PyTorch with CUDA 12.4 GPU support...
%VENV_PIP% install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet

echo       [Step B] Installing other dependencies...
%VENV_PIP% install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check requirements.txt.
    pause
    exit /b 1
)
echo       All dependencies installed.
echo.

if not exist "data\raw\" (
    mkdir "data\raw"
    echo       Created data\raw\  — drop your CSVs here!
)

echo [3/5] Checking datasets...
if exist "data\auto_balanced_dataset.csv" (
    echo       Datasets already prepared. Skipping preparation.
    echo       [TIP] Delete data\auto_balanced_dataset.csv to force rebuilding
    echo             if you dropped new raw datasets into data\raw\.
) else (
    echo       No prepared dataset found — starting dataset preparation...
    %VENV_PYTHON% scripts\prepare_auto_dataset.py
    if errorlevel 1 (
        echo [ERROR] Dataset preparation failed.
        echo         Make sure at least one CSV is in data\raw\  OR
        echo         that you have an internet connection for auto-download.
        pause
        exit /b 1
    )
    echo       Dataset preparation complete!
)
echo.

echo [4/5] Checking text emotion model...

if exist "models\emotion_model_auto\" (
    echo       Model already trained. Skipping training.
    echo       [TIP] Delete the models\emotion_model_auto\ folder to force retraining
    echo             after dropping new datasets into data\raw\.
) else (
    echo       No trained model found — starting training...
    echo       This may take 10-30 minutes depending on your GPU/CPU.
    %VENV_PYTHON% scripts\train_auto_model.py
    if errorlevel 1 (
        echo [ERROR] Training failed. Check the output above for details.
        pause
        exit /b 1
    )
    echo       Training complete!
)
echo.

echo [5/5] Launching Streamlit app...
echo  App is starting. Press Ctrl+C to stop.
echo.
%VENV_STREAMLIT% run chatbot_mega.py
