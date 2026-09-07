from __future__ import annotations

from app.config import settings
from app.providers.embedded_gguf import EmbeddedGGUFProvider
from app.providers.openai_compatible import OpenAICompatibleProvider
from app.providers.transformers_peft import TransformersPEFTProvider


def create_chat_provider():
    backend = settings.model_backend.strip().lower()
    if backend in {"embedded", "embedded_gguf", "gguf", "llama_cpp", "llama.cpp"}:
        return EmbeddedGGUFProvider(
            model_path=settings.local_model_path,
            model_name=settings.local_model_name or settings.ai_model,
            timeout=settings.ai_timeout_seconds,
            n_ctx=settings.local_model_context,
            n_gpu_layers=settings.local_model_gpu_layers,
            n_threads=settings.local_model_threads,
            chat_format=settings.local_model_chat_format,
            verbose=settings.local_model_verbose,
            lora_path=settings.local_lora_path,
            lora_scale=settings.local_lora_scale,
        )
    if backend in {"transformers", "transformers_peft", "hf", "huggingface"}:
        adapter_path = settings.hf_adapter_path
        if not adapter_path and settings.hf_adapter_user:
            try:
                from app.core.continual_learning import AdapterRegistry
                active = AdapterRegistry().active(settings.hf_adapter_user)
                adapter_path = active.get("path", "") if active else ""
            except Exception:
                adapter_path = ""
        return TransformersPEFTProvider(
            model_path=settings.hf_model_path or settings.lora_base_model,
            adapter_path=adapter_path,
            timeout=settings.ai_timeout_seconds,
            max_new_tokens=settings.hf_max_new_tokens,
            local_files_only=settings.hf_local_files_only,
        )
    if backend in {"openai", "openai_compatible", "remote", "server"}:
        return OpenAICompatibleProvider(
            settings.ai_base_url,
            settings.ai_api_key,
            settings.ai_model,
            settings.ai_timeout_seconds,
        )
    raise ValueError(f"Unsupported MODEL_BACKEND: {settings.model_backend!r}")
