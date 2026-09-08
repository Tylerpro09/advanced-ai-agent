@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Advanced AI Agent - Windows Launcher

set "PY_LAUNCH="
set "VENV_PY=%CD%\.venv\Scripts\python.exe"

echo ============================================================
echo   Advanced AI Agent - Windows launcher
echo ============================================================
echo.

rem ------------------------------------------------------------
rem Find a usable Python 3.11+ installation.
rem ------------------------------------------------------------
py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if not errorlevel 1 set "PY_LAUNCH=py -3.12"

if not defined PY_LAUNCH (
  py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
  if not errorlevel 1 set "PY_LAUNCH=py -3"
)

if not defined PY_LAUNCH (
  python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
  if not errorlevel 1 set "PY_LAUNCH=python"
)

if not defined PY_LAUNCH (
  echo [ERROR] Python 3.11 or newer was not found.
  echo Install Python 3.12 from https://www.python.org/downloads/windows/
  echo During installation enable: Add Python to PATH.
  goto :fail
)

echo [OK] Python launcher: %PY_LAUNCH%

rem ------------------------------------------------------------
rem Create/recover the virtual environment.
rem ------------------------------------------------------------
if not exist "%VENV_PY%" (
  echo [INFO] Creating .venv ...
  if exist ".venv" rmdir /s /q ".venv" >nul 2>&1
  %PY_LAUNCH% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Could not create the Python virtual environment.
    goto :fail
  )
)

"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if errorlevel 1 (
  echo [WARN] Existing .venv is invalid or uses an old Python. Recreating it ...
  rmdir /s /q ".venv" >nul 2>&1
  %PY_LAUNCH% -m venv .venv
  if errorlevel 1 goto :fail
)

echo [OK] Virtual environment ready.

rem ------------------------------------------------------------
rem Install only missing REQUIRED dependencies.
rem ------------------------------------------------------------
"%VENV_PY%" -c "import fastapi,uvicorn,httpx,pydantic,pydantic_settings,multipart,pypdf" >nul 2>&1
if errorlevel 1 (
  echo [INFO] Installing required dependencies ...
  "%VENV_PY%" -m pip install --disable-pip-version-check -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Failed to install requirements.txt.
    echo Check your Internet connection and the pip output above.
    goto :fail
  )
) else (
  echo [OK] Required dependencies already installed.
)

rem Neural memory is optional because the program has a lexical fallback.
"%VENV_PY%" -c "import numpy,sentence_transformers" >nul 2>&1
if errorlevel 1 (
  echo [INFO] Installing optional neural-memory dependencies ...
  "%VENV_PY%" -m pip install --disable-pip-version-check -r requirements-neural.txt
  if errorlevel 1 (
    echo [WARN] Neural dependencies could not be installed.
    echo        The agent will still start using lexical-memory fallback.
  )
)

rem ------------------------------------------------------------
rem Print configuration and warnings, but DO NOT block web startup
rem just because a model file/runtime is not ready yet. The GGUF
rem provider is lazy-loaded, so /health, /ready and the UI can run.
rem ------------------------------------------------------------
echo.
echo [INFO] Configuration diagnostics:
"%VENV_PY%" doctor.py
if errorlevel 1 (
  echo [ERROR] Base diagnostics failed.
  goto :fail
)

"%VENV_PY%" -c "from app.config import settings; from pathlib import Path; import importlib.util; b=settings.model_backend.strip().lower(); print('[INFO] Backend:', settings.model_backend); print('[INFO] URL: http://127.0.0.1:%s' % settings.app_port); print('[WARN] GGUF model is missing:', settings.local_model_path) if b in ('embedded','embedded_gguf','gguf','llama_cpp','llama.cpp') and not Path(settings.local_model_path).is_file() else None; print('[WARN] llama-cpp-python is not installed. Run install_local_model_windows.bat before chatting with embedded GGUF.') if b in ('embedded','embedded_gguf','gguf','llama_cpp','llama.cpp') and importlib.util.find_spec('llama_cpp') is None else None"

for /f %%P in ('"%VENV_PY%" -c "from app.config import settings; print(settings.app_port)"') do set "AAA_PORT=%%P"
if not defined AAA_PORT set "AAA_PORT=8000"

echo.
echo ============================================================
echo [START] Server starting at http://127.0.0.1:%AAA_PORT%
echo [INFO] Press Ctrl+C to stop it.
echo ============================================================
echo.

rem Open the UI after a short delay without blocking the server process.
start "" /b cmd /c "timeout /t 3 /nobreak ^>nul ^& start "" http://127.0.0.1:%AAA_PORT%"

"%VENV_PY%" run.py
set "SERVER_EXIT=%ERRORLEVEL%"

echo.
if "%SERVER_EXIT%"=="0" (
  echo [INFO] Server stopped normally.
) else (
  echo [ERROR] Server stopped with exit code %SERVER_EXIT%.
)
echo.
pause
exit /b %SERVER_EXIT%

:fail
echo.
echo ============================================================
echo [FAILED] Advanced AI Agent could not start.
echo The window will remain open so you can read the error above.
echo ============================================================
echo.
pause
exit /b 1
