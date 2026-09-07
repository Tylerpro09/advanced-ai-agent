#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-neural.txt
python - <<'PY'
from app.config import settings
import importlib.util
b = settings.model_backend.lower()
if b in {'embedded','embedded_gguf','gguf','llama_cpp','llama.cpp'} and importlib.util.find_spec('llama_cpp') is None:
    raise SystemExit("llama-cpp-python is missing for embedded GGUF mode. Run ./install_local_model_linux.sh once.")
if b in {'transformers','transformers_peft','hf','huggingface'} and (
    importlib.util.find_spec('transformers') is None or importlib.util.find_spec('peft') is None
):
    raise SystemExit("transformers/peft are missing. Run ./install_training_linux.sh once.")
PY
python doctor.py --runtime
python run.py
