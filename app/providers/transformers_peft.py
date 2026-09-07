from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Lock
from typing import Any


class TransformersPEFTProvider:
    """In-process Hugging Face causal LM with an optional PEFT/LoRA adapter.

    This backend is intended for users who want the trained LoRA weights to affect generation
    directly. It is heavier than GGUF and currently provides text generation only (no native
    OpenAI-style tool calls).
    """

    def __init__(self, model_path: str, adapter_path: str = "", timeout: float = 300.0, max_new_tokens: int = 768, local_files_only: bool = False):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self.model = Path(model_path).name or "transformers-peft"
        self._model = None
        self._tokenizer = None
        self._lock = Lock()
        self._inference_lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def status(self) -> dict[str, Any]:
        return {
            "backend": "transformers_peft",
            "model": self.model,
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "adapter_exists": bool(self.adapter_path and Path(self.adapter_path).is_dir()),
            "loaded": self.loaded,
            "tool_calls": False,
        }

    def set_adapter(self, adapter_path: str) -> None:
        with self._lock:
            self.adapter_path = adapter_path
            # Reload lazily so a new adapter is guaranteed to be applied cleanly.
            self._model = None
            self._tokenizer = None

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return self._model, self._tokenizer
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except Exception as exc:
                raise RuntimeError("Install requirements-training.txt for transformers_peft backend") from exc
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, local_files_only=self.local_files_only)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            dtype = None
            if torch.cuda.is_available():
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            base = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                local_files_only=self.local_files_only,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if self.adapter_path:
                try:
                    from peft import PeftModel
                except Exception as exc:
                    raise RuntimeError("PEFT is required to load a LoRA adapter") from exc
                if not Path(self.adapter_path).is_dir():
                    raise FileNotFoundError(f"LoRA adapter not found: {self.adapter_path}")
                base = PeftModel.from_pretrained(base, self.adapter_path)
            base.eval()
            self._model, self._tokenizer = base, tokenizer
            return base, tokenizer

    def _chat_sync(self, messages: list[dict[str, Any]], temperature: float) -> dict[str, Any]:
        import torch
        model, tokenizer = self._load()
        clean = [{"role": x.get("role", "user"), "content": str(x.get("content", ""))} for x in messages if x.get("role") in {"system", "user", "assistant"}]
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(clean, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join(f"{x['role'].title()}: {x['content']}" for x in clean) + "\nAssistant:"
        encoded = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        do_sample = temperature > 0.01
        with self._inference_lock, torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=max(0.05, float(temperature)) if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0, encoded["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {"role": "assistant", "content": text}

    async def chat(self, messages: list[dict[str, Any]], temperature: float = 0.4, tools=None, tool_choice=None) -> dict[str, Any]:
        task = asyncio.to_thread(self._chat_sync, messages, temperature)
        return await asyncio.wait_for(task, timeout=self.timeout)
