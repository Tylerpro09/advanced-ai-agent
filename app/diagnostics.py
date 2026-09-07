from __future__ import annotations

import importlib.util
import platform
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.config import settings
from app.version import APP_VERSION


@dataclass
class Check:
    name: str
    ok: bool
    level: str = "error"  # error | warning | info
    detail: str = ""


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def collect_diagnostics(*, runtime: bool = False) -> dict[str, Any]:
    checks: list[Check] = []
    checks.append(Check("python", sys.version_info >= (3, 11), detail=platform.python_version()))

    try:
        conn = sqlite3.connect(settings.database_path, timeout=2)
        result = conn.execute("PRAGMA quick_check").fetchone()
        conn.close()
        checks.append(Check("sqlite", bool(result and result[0] == "ok"), detail=str(result[0] if result else "no result")))
    except Exception as exc:
        checks.append(Check("sqlite", False, detail=f"{type(exc).__name__}: {exc}"))

    static_ok = (settings.project_root / "static" / "index.html").is_file()
    checks.append(Check("web_ui", static_ok, level="warning" if not static_ok else "info", detail="static/index.html"))

    backend = settings.model_backend.strip().lower()
    supported = backend in {
        "embedded", "embedded_gguf", "gguf", "llama_cpp", "llama.cpp",
        "transformers", "transformers_peft", "hf", "huggingface",
        "openai", "openai_compatible", "remote", "server",
    }
    checks.append(Check("model_backend", supported, detail=settings.model_backend))

    if backend in {"embedded", "embedded_gguf", "gguf", "llama_cpp", "llama.cpp"}:
        model_exists = bool(settings.local_model_path and Path(settings.local_model_path).is_file())
        checks.append(Check("gguf_model", model_exists if runtime else True, level="error" if runtime else "warning", detail="present" if model_exists else "model file not bundled"))
        llama_ok = _has("llama_cpp")
        checks.append(Check("llama_cpp", llama_ok if runtime else True, level="error" if runtime else "warning", detail="installed" if llama_ok else "install requirements-local-model.txt"))
    elif backend in {"transformers", "transformers_peft", "hf", "huggingface"}:
        deps = _has("transformers") and _has("peft") and _has("torch")
        checks.append(Check("transformers_peft", deps if runtime else True, level="error" if runtime else "warning", detail="installed" if deps else "install requirements-training.txt"))
        configured = bool(settings.hf_model_path or settings.lora_base_model)
        checks.append(Check("hf_model", configured if runtime else True, level="error" if runtime else "warning", detail="configured" if configured else "HF_MODEL_PATH/LORA_BASE_MODEL empty"))
    elif backend in {"openai", "openai_compatible", "remote", "server"}:
        checks.append(Check("openai_endpoint", bool(settings.ai_base_url), detail="configured" if settings.ai_base_url else "AI_BASE_URL empty"))

    if settings.neural_memory_enabled and settings.embedding_provider.lower() == "local":
        neural_ok = _has("sentence_transformers")
        checks.append(Check("neural_memory_encoder", neural_ok, level="warning", detail="installed" if neural_ok else "lexical fallback remains available"))

    if settings.experience_policy_enabled:
        policy_ok = _has("numpy") or _has("torch")
        checks.append(Check("experience_policy", policy_ok, level="warning" if not policy_ok else "info", detail="NumPy/PyTorch available" if policy_ok else "install requirements-neural.txt"))

    if settings.continual_learning_enabled:
        deps = _has("torch") and _has("transformers") and _has("peft") and _has("accelerate")
        checks.append(Check("lora_dependencies", deps, detail="installed" if deps else "install requirements-training.txt"))
        checks.append(Check("lora_base_model", bool(settings.lora_base_model.strip()), detail="configured" if settings.lora_base_model.strip() else "LORA_BASE_MODEL empty"))

    errors = [c for c in checks if not c.ok and c.level == "error"]
    warnings = [c for c in checks if not c.ok and c.level == "warning"]
    return {
        "ok": not errors,
        "version": APP_VERSION,
        "runtime_mode": runtime,
        "errors": len(errors),
        "warnings": len(warnings),
        "checks": [asdict(c) for c in checks],
    }
