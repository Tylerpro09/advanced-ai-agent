@echo off
setlocal
cd /d "%~dp0"
if not exist .venv (
  py -3.12 -m venv .venv 2>nul || py -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m pip install -r requirements-neural.txt
python -c "from app.config import settings; import importlib.util,sys; b=settings.model_backend.lower(); need=(b in ('embedded','embedded_gguf','gguf','llama_cpp','llama.cpp') and importlib.util.find_spec('llama_cpp') is None); sys.exit(2 if need else 0)"
if errorlevel 2 (
  echo llama-cpp-python is not installed for embedded GGUF mode.
  echo Run install_local_model_windows.bat once.
  exit /b 1
)
python -c "from app.config import settings; import importlib.util,sys; b=settings.model_backend.lower(); need=(b in ('transformers','transformers_peft','hf','huggingface') and (importlib.util.find_spec('transformers') is None or importlib.util.find_spec('peft') is None)); sys.exit(3 if need else 0)"
if errorlevel 3 (
  echo transformers/peft are not installed for MODEL_BACKEND=transformers_peft.
  echo Run install_training_windows.bat once.
  exit /b 1
)
python doctor.py --runtime
if errorlevel 1 (
  echo Runtime diagnostics failed. Fix the FAIL items above before starting the server.
  exit /b 1
)
python run.py
