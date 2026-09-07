@echo off
setlocal
if not exist .venv (py -3.12 -m venv .venv 2>nul || py -m venv .venv)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-neural.txt
python -m pip install -r requirements-local-model.txt
echo.
echo Local-model runtime installed.
echo Put a GGUF file at models\model.gguf, then run start_windows.bat
pause
