@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "VENV_PY=%CD%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [ERROR] .venv was not found. Run start_windows.bat once first.
  if /I not "%~1"=="/nopause" pause
  exit /b 1
)

"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
"%VENV_PY%" -m pip install -r requirements-training.txt
if errorlevel 1 goto :fail

echo.
echo [OK] LoRA training dependencies installed in .venv.
if /I not "%~1"=="/nopause" pause
exit /b 0

:fail
echo.
echo [ERROR] Training dependencies failed to install.
echo For NVIDIA CUDA, install a compatible PyTorch build first if needed,
echo then run this file again.
if /I not "%~1"=="/nopause" pause
exit /b 1
