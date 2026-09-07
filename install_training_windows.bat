@echo off
setlocal
cd /d "%~dp0"
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
if errorlevel 1 (
  echo.
  echo Training dependencies failed to install.
  echo For NVIDIA CUDA, install the appropriate PyTorch build from pytorch.org first, then rerun this file.
  pause
  exit /b 1
)
echo.
echo LoRA training dependencies installed.
pause
