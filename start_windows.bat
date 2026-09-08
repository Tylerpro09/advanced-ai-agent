@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Advanced AI Agent - Windows Launcher

set "PY_LAUNCH="
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ============================================================
echo   Advanced AI Agent - Windows launcher
echo ============================================================
echo.

rem Find Python 3.11+.
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
  echo Install Python 3.12 and enable Add Python to PATH.
  goto :fail
)
echo [OK] Python launcher: %PY_LAUNCH%

rem Create/recover venv.
if not exist "%VENV_PY%" (
  echo [INFO] Creating .venv ...
  if exist ".venv" rmdir /s /q ".venv" >nul 2>&1
  %PY_LAUNCH% -m venv .venv
  if errorlevel 1 goto :venv_fail
)
"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if errorlevel 1 (
  echo [WARN] Recreating invalid .venv ...
  rmdir /s /q ".venv" >nul 2>&1
  %PY_LAUNCH% -m venv .venv
  if errorlevel 1 goto :venv_fail
)
echo [OK] Virtual environment ready.

rem Required packages.
"%VENV_PY%" -c "import fastapi,uvicorn,httpx,pydantic,pydantic_settings,multipart,pypdf" >nul 2>&1
if errorlevel 1 (
  echo [INFO] Installing required dependencies ...
  "%VENV_PY%" -m pip install --disable-pip-version-check -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Failed to install requirements.txt.
    goto :fail
  )
) else (
  echo [OK] Required dependencies already installed.
)

rem Neural memory is optional; lexical fallback remains available.
"%VENV_PY%" -c "import numpy,sentence_transformers" >nul 2>&1
if errorlevel 1 (
  echo [INFO] Installing optional neural-memory dependencies ...
  "%VENV_PY%" -m pip install --disable-pip-version-check -r requirements-neural.txt
  if errorlevel 1 echo [WARN] Neural package install failed; lexical fallback will be used.
)

rem Read configured backend before applying the LM Studio compatibility fallback.
for /f %%B in ('"%VENV_PY%" -c "from app.config import settings; print(settings.model_backend.strip().lower())"') do set "AAA_BACKEND=%%B"
for /f %%M in ('"%VENV_PY%" -c "from app.config import settings; from pathlib import Path; print(1 if Path(settings.local_model_path).is_file() else 0)"') do set "AAA_GGUF_EXISTS=%%M"

rem Detect LM Studio. Its OpenAI-compatible server normally listens on port 1234.
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r=Invoke-RestMethod -Uri 'http://127.0.0.1:1234/v1/models' -Method Get -TimeoutSec 3; if ($null -ne $r.data -and $r.data.Count -gt 0) { exit 0 } else { exit 2 } } catch { exit 1 }" >nul 2>&1
set "AAA_LM_STATUS=%ERRORLEVEL%"

rem Old clones may still have MODEL_BACKEND=embedded_gguf in .env. If the GGUF is
rem absent and LM Studio is running, transparently use LM Studio for this launch.
if /I "%AAA_BACKEND%"=="embedded_gguf" if "%AAA_GGUF_EXISTS%"=="0" if "%AAA_LM_STATUS%"=="0" goto :use_lmstudio
if /I "%AAA_BACKEND%"=="embedded" if "%AAA_GGUF_EXISTS%"=="0" if "%AAA_LM_STATUS%"=="0" goto :use_lmstudio
if /I "%AAA_BACKEND%"=="gguf" if "%AAA_GGUF_EXISTS%"=="0" if "%AAA_LM_STATUS%"=="0" goto :use_lmstudio
goto :after_lmstudio

:use_lmstudio
set "MODEL_BACKEND=openai_compatible"
set "AI_BASE_URL=http://127.0.0.1:1234/v1"
set "AI_API_KEY=lm-studio"
set "AI_MODEL=auto"
echo [OK] LM Studio detected. Using its loaded model automatically.

:after_lmstudio
if /I "%AAA_BACKEND%"=="openai_compatible" if not "%AAA_LM_STATUS%"=="0" (
  echo [WARN] LM Studio was not detected at http://127.0.0.1:1234/v1
  echo        Open LM Studio, load a model, then start Local Server in Developer.
)
if /I "%MODEL_BACKEND%"=="openai_compatible" if not "%AAA_LM_STATUS%"=="0" (
  echo [WARN] LM Studio was not detected at http://127.0.0.1:1234/v1
)

rem Non-blocking diagnostics.
echo.
echo [INFO] Configuration diagnostics:
"%VENV_PY%" doctor.py
if errorlevel 1 goto :fail

"%VENV_PY%" -c "from app.config import settings; print('[INFO] Backend:', settings.model_backend); print('[INFO] AI endpoint:', settings.ai_base_url); print('[INFO] Model:', settings.ai_model)"
for /f %%P in ('"%VENV_PY%" -c "from app.config import settings; print(settings.app_port)"') do set "AAA_PORT=%%P"
if not defined AAA_PORT set "AAA_PORT=8000"

echo.
echo ============================================================
echo [START] Server starting at http://127.0.0.1:%AAA_PORT%
echo [INFO] Open that address in your browser.
echo [INFO] Press Ctrl+C to stop the server.
echo ============================================================
echo.

"%VENV_PY%" run.py
set "SERVER_EXIT=%ERRORLEVEL%"
echo.
if "%SERVER_EXIT%"=="0" (
  echo [INFO] Server stopped normally.
) else (
  echo [ERROR] Server stopped with exit code %SERVER_EXIT%.
)
pause
exit /b %SERVER_EXIT%

:venv_fail
echo [ERROR] Could not create the Python virtual environment.
:fail
echo.
echo ============================================================
echo [FAILED] Advanced AI Agent could not start.
echo The window will remain open so you can read the error above.
echo ============================================================
pause
exit /b 1
