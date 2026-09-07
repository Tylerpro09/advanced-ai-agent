from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.config import settings
from app.core.experience import ExperienceStore


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)[:100] or "user"


@dataclass
class AdapterVersion:
    id: str
    user_id: str
    path: str
    base_model: str
    created_at: float
    examples: int
    min_reward: float
    epochs: float
    learning_rate: float
    active: bool = False
    quality_passed: bool = True


class AdapterRegistry:
    """Versioned LoRA adapter registry with activation and rollback pointers."""

    def __init__(self, root: str | None = None):
        self.root = Path(root or settings.lora_output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.json"
        self._lock = threading.RLock()

    def _load(self) -> dict[str, Any]:
        if not self.registry_path.is_file():
            return {"versions": [], "active": {}, "history": {}}
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            data.setdefault("versions", [])
            data.setdefault("active", {})
            data.setdefault("history", {})
            return data
        except Exception:
            return {"versions": [], "active": {}, "history": {}}

    def _save(self, data: dict[str, Any]) -> None:
        tmp = self.registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.registry_path)

    def register(self, version: AdapterVersion) -> None:
        with self._lock:
            data = self._load()
            data["versions"] = [x for x in data["versions"] if x.get("id") != version.id]
            data["versions"].append(asdict(version))
            self._save(data)

    def list(self, user_id: str | None = None) -> list[dict[str, Any]]:
        data = self._load()
        active = data.get("active", {})
        items = []
        for item in data.get("versions", []):
            if user_id and item.get("user_id") != user_id:
                continue
            row = dict(item)
            row["active"] = active.get(item.get("user_id", "")) == item.get("id")
            items.append(row)
        items.sort(key=lambda x: float(x.get("created_at", 0.0)), reverse=True)
        return items

    def get(self, version_id: str) -> dict[str, Any] | None:
        for item in self.list():
            if item.get("id") == version_id:
                return item
        return None

    def active(self, user_id: str) -> dict[str, Any] | None:
        data = self._load()
        version_id = data.get("active", {}).get(user_id)
        return self.get(version_id) if version_id else None

    def activate(self, user_id: str, version_id: str) -> dict[str, Any]:
        with self._lock:
            version = self.get(version_id)
            if not version or version.get("user_id") != user_id:
                raise ValueError("Adapter version not found for this user")
            if not Path(version["path"]).is_dir():
                raise FileNotFoundError(f"Adapter directory missing: {version['path']}")
            data = self._load()
            old = data.setdefault("active", {}).get(user_id)
            history = data.setdefault("history", {}).setdefault(user_id, [])
            if old and old != version_id:
                history.append(old)
                del history[:-20]
            data["active"][user_id] = version_id
            self._save(data)
            return self.get(version_id) or version

    def rollback(self, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._load()
            history = data.setdefault("history", {}).setdefault(user_id, [])
            while history:
                previous = history.pop()
                version = self.get(previous)
                if version and Path(version["path"]).is_dir():
                    data.setdefault("active", {})[user_id] = previous
                    self._save(data)
                    return self.get(previous)
            data.setdefault("active", {}).pop(user_id, None)
            self._save(data)
            return None


class LoRAContinualTrainer:
    """Train versioned PEFT/LoRA adapters from positively-rated experiences.

    Important: GGUF is an inference format. Training uses the original Hugging Face model
    (local path or repo). The resulting adapter can be used by the transformers_peft backend,
    or converted separately for compatible llama.cpp workflows.
    """

    def __init__(self, experiences: ExperienceStore, registry: AdapterRegistry | None = None):
        self.experiences = experiences
        self.registry = registry or AdapterRegistry()
        self._train_lock = threading.Lock()

    def status(self, user_id: str) -> dict[str, Any]:
        return {
            "enabled": bool(settings.continual_learning_enabled),
            "base_model": settings.lora_base_model,
            "active": self.registry.active(user_id),
            "versions": self.registry.list(user_id),
            "min_examples": settings.lora_min_examples,
            "min_reward": settings.lora_min_reward,
        }

    def _examples(self, user_id: str, min_reward: float, max_examples: int) -> list[dict[str, str]]:
        rows = self.experiences.training_examples(user_id, min_reward=min_reward, limit=max_examples)
        # De-duplicate exact situation/action pairs so repeated thumbs-up does not dominate.
        seen: set[tuple[str, str]] = set()
        out: list[dict[str, str]] = []
        sensitive = re.compile(r"(?i)(password|passwd|api[_ -]?key|secret[_ -]?key|bearer\s+[a-z0-9._-]{12,}|BEGIN [A-Z ]*PRIVATE KEY|credit card|cvv)")
        for row in rows:
            key = (row["situation"].strip(), row["action"].strip())
            if not key[0] or not key[1] or key in seen:
                continue
            if settings.lora_filter_sensitive and sensitive.search(key[0] + "\n" + key[1]):
                continue
            seen.add(key)
            out.append(row)
        return out

    def train(self, user_id: str, activate: bool = True) -> dict[str, Any]:
        if not self._train_lock.acquire(blocking=False):
            raise RuntimeError("A LoRA training run is already in progress in this process")
        try:
            return self._train_locked(user_id, activate)
        finally:
            self._train_lock.release()

    def _train_locked(self, user_id: str, activate: bool = True) -> dict[str, Any]:
        if not settings.continual_learning_enabled:
            raise RuntimeError("CONTINUAL_LEARNING_ENABLED=false")
        if not settings.lora_base_model.strip():
            raise RuntimeError("LORA_BASE_MODEL is empty. Set it to the original Hugging Face model path/repo.")

        examples = self._examples(user_id, float(settings.lora_min_reward), int(settings.lora_max_examples))
        if len(examples) < int(settings.lora_min_examples):
            raise RuntimeError(
                f"Need at least {settings.lora_min_examples} positively-rated experiences; found {len(examples)}."
            )

        try:
            import torch
            from torch.utils.data import Dataset
            from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as exc:
            raise RuntimeError(
                "LoRA training dependencies are missing. Install requirements-training.txt."
            ) from exc

        base_model = settings.lora_base_model.strip()
        local_only = bool(settings.lora_local_files_only)
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, local_files_only=local_only)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = None
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            local_files_only=local_only,
            low_cpu_mem_usage=True,
        )
        if settings.lora_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.config.use_cache = False

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(settings.lora_rank),
            lora_alpha=int(settings.lora_alpha),
            lora_dropout=float(settings.lora_dropout),
            target_modules=settings.lora_target_modules.strip() or "all-linear",
            bias="none",
        )
        model = get_peft_model(model, config)

        max_len = int(settings.lora_max_seq_length)

        def render_prompt(user: str) -> str:
            messages = [{"role": "user", "content": user}]
            if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return f"User: {user}\nAssistant:"

        def render_full(user: str, assistant: str) -> str:
            messages = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
            if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return f"User: {user}\nAssistant: {assistant}"

        items: list[dict[str, list[int]]] = []
        for ex in examples:
            prompt_ids = tokenizer(render_prompt(ex["situation"]), add_special_tokens=False)["input_ids"]
            full = tokenizer(
                render_full(ex["situation"], ex["action"]),
                add_special_tokens=False,
                truncation=True,
                max_length=max_len,
            )["input_ids"]
            prompt_len = min(len(prompt_ids), len(full))
            labels = [-100] * prompt_len + full[prompt_len:]
            if all(x == -100 for x in labels):
                continue
            items.append({"input_ids": full, "attention_mask": [1] * len(full), "labels": labels})

        if len(items) < int(settings.lora_min_examples):
            raise RuntimeError("Too few usable tokenized examples after truncation.")

        # Deterministic holdout. This is not a benchmark, but it provides a simple
        # anti-regression gate before a new adapter becomes active.
        min_val = max(0, int(settings.lora_min_validation_examples))
        val_count = int(round(len(items) * max(0.0, min(0.5, float(settings.lora_validation_fraction)))))
        if len(items) >= int(settings.lora_min_examples) + min_val:
            val_count = max(min_val, val_count)
        else:
            val_count = 0
        val_count = min(val_count, max(0, len(items) - int(settings.lora_min_examples)))
        eval_items = items[-val_count:] if val_count else []
        train_items = items[:-val_count] if val_count else items

        class ExperienceDataset(Dataset):
            def __init__(self, rows):
                self.rows = rows

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, idx):
                return self.rows[idx]

        def collate(batch):
            width = max(len(x["input_ids"]) for x in batch)
            pad = tokenizer.pad_token_id
            input_ids, masks, labels = [], [], []
            for row in batch:
                n = width - len(row["input_ids"])
                input_ids.append(row["input_ids"] + [pad] * n)
                masks.append(row["attention_mask"] + [0] * n)
                labels.append(row["labels"] + [-100] * n)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        version_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        out_dir = self.registry.root / _safe(user_id) / version_id
        out_dir.mkdir(parents=True, exist_ok=False)

        args = TrainingArguments(
            output_dir=str(out_dir / "checkpoints"),
            per_device_train_batch_size=int(settings.lora_batch_size),
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=int(settings.lora_gradient_accumulation),
            num_train_epochs=float(settings.lora_epochs),
            learning_rate=float(settings.lora_learning_rate),
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            fp16=bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
            bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            dataloader_pin_memory=bool(torch.cuda.is_available()),
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ExperienceDataset(train_items),
            eval_dataset=ExperienceDataset(eval_items) if eval_items else None,
            data_collator=collate,
        )
        pre_eval_loss = None
        if eval_items:
            pre_eval_loss = float(trainer.evaluate().get("eval_loss", 0.0))
        result = trainer.train()
        post_eval_loss = None
        if eval_items:
            post_eval_loss = float(trainer.evaluate().get("eval_loss", 0.0))

        quality_passed = True
        if pre_eval_loss is not None and post_eval_loss is not None and pre_eval_loss > 0:
            quality_passed = post_eval_loss <= pre_eval_loss * float(settings.lora_max_eval_regression)

        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)

        metrics = dict(getattr(result, "metrics", {}) or {})
        metrics.update({
            "train_examples": len(train_items),
            "validation_examples": len(eval_items),
            "pre_eval_loss": pre_eval_loss,
            "post_eval_loss": post_eval_loss,
            "quality_passed": quality_passed,
        })
        manifest = {
            "id": version_id,
            "user_id": user_id,
            "base_model": base_model,
            "created_at": time.time(),
            "examples": len(items),
            "min_reward": float(settings.lora_min_reward),
            "epochs": float(settings.lora_epochs),
            "learning_rate": float(settings.lora_learning_rate),
            "quality_passed": quality_passed,
            "metrics": metrics,
        }
        (out_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        version = AdapterVersion(
            id=version_id,
            user_id=user_id,
            path=str(out_dir),
            base_model=base_model,
            created_at=manifest["created_at"],
            examples=len(items),
            min_reward=float(settings.lora_min_reward),
            epochs=float(settings.lora_epochs),
            learning_rate=float(settings.lora_learning_rate),
            quality_passed=quality_passed,
        )
        self.registry.register(version)
        activated = False
        if activate and quality_passed:
            self.registry.activate(user_id, version_id)
            activated = True
        return {
            "ok": True,
            "adapter": self.registry.get(version_id),
            "metrics": metrics,
            "activated": activated,
            "quality_passed": quality_passed,
        }
