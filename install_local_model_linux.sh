#!/usr/bin/env bash
set -e
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-neural.txt
python -m pip install -r requirements-local-model.txt
echo "Local-model runtime installed. Put a GGUF at models/model.gguf, then run ./start_linux.sh"
