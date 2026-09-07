from __future__ import annotations

import io
import json
import math
import os
import re
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from app.config import settings
from app.core.experience import ExperienceStore


def _safe_user(user_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", user_id)[:100] or "user"


@dataclass
class PolicyStatus:
    enabled: bool
    available: bool
    user_id: str
    trained: bool
    samples: int
    input_dim: int
    model_path: str
    last_loss: float | None = None
    reason: str = ""


class ExperiencePolicyNetwork:
    """Small trainable neural network that learns expected outcome from episodic experiences.

    It is intentionally separate from the LLM. Ratings update this network quickly, even when
    the main model is a read-only GGUF. Its prediction is injected into the next prompt as a
    learned risk/success prior. This is real weight updating, but not a replacement for LoRA.
    """

    def __init__(self, experiences: ExperienceStore):
        self.experiences = experiences
        self.root = Path(settings.experience_policy_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._train_lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return bool(settings.experience_policy_enabled)

    def _paths(self, user_id: str) -> tuple[Path, Path]:
        stem = _safe_user(user_id)
        return self.root / f"{stem}.pt", self.root / f"{stem}.json"

    def _numpy_path(self, user_id: str) -> Path:
        return self.root / f"{_safe_user(user_id)}.npz"

    @staticmethod
    def _atomic_json(path: Path, data: dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _torch():
        try:
            import torch
            import torch.nn as nn
            return torch, nn
        except Exception:
            return None, None

    def _build_model(self, nn, dim: int):
        hidden = max(16, int(settings.experience_policy_hidden))
        return nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(float(settings.experience_policy_dropout)),
            nn.Linear(hidden, max(8, hidden // 2)),
            nn.ReLU(),
            nn.Linear(max(8, hidden // 2), 1),
            nn.Tanh(),
        )

    def status(self, user_id: str) -> dict[str, Any]:
        model_path, meta_path = self._paths(user_id)
        numpy_path = self._numpy_path(user_id)
        torch, _ = self._torch()
        try:
            import numpy  # noqa: F401
            numpy_available = True
        except Exception:
            numpy_available = False
        meta: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        backend = str(meta.get("backend") or ("torch" if model_path.is_file() else ("numpy" if numpy_path.is_file() else "")))
        available = torch is not None or numpy_available
        reason = "" if available else "Neither PyTorch nor NumPy is installed"
        result = PolicyStatus(
            enabled=self.enabled,
            available=available,
            user_id=user_id,
            trained=model_path.is_file() or numpy_path.is_file(),
            samples=int(meta.get("samples", 0)),
            input_dim=int(meta.get("input_dim", 0)),
            model_path=str(model_path if backend == "torch" else numpy_path),
            last_loss=meta.get("last_loss"),
            reason=reason,
        )
        out = asdict(result)
        out["backend"] = backend or ("torch" if torch is not None else ("numpy" if numpy_available else "unavailable"))
        return out

    def _prepare_samples(self, user_id: str) -> tuple[list[tuple[list[float], float]], int]:
        samples = self.experiences.rated_vector_samples(user_id, int(settings.experience_policy_max_samples))
        min_samples = max(2, int(settings.experience_policy_min_samples))
        if len(samples) < min_samples:
            raise ValueError(f"need at least {min_samples} rated experiences")
        dim = len(samples[0][0])
        samples = [(v, r) for v, r in samples if len(v) == dim and math.isfinite(r)]
        if len(samples) < min_samples:
            raise ValueError("not enough compatible vectors")
        return samples, dim

    def _train_numpy(self, user_id: str, samples: list[tuple[list[float], float]], dim: int) -> dict[str, Any]:
        try:
            import numpy as np
        except Exception:
            return {**self.status(user_id), "ok": False, "reason": "NumPy is not installed"}
        hidden = max(16, int(settings.experience_policy_hidden))
        path = self._numpy_path(user_id)
        _, meta_path = self._paths(user_id)
        rng = np.random.default_rng(42)
        scale1 = 1.0 / max(1.0, dim ** 0.5)
        W1 = rng.normal(0.0, scale1, size=(dim, hidden)).astype(np.float32)
        b1 = np.zeros((1, hidden), dtype=np.float32)
        W2 = rng.normal(0.0, 1.0 / hidden ** 0.5, size=(hidden, 1)).astype(np.float32)
        b2 = np.zeros((1, 1), dtype=np.float32)
        if path.is_file():
            try:
                old = np.load(path)
                if int(old["input_dim"]) == dim and int(old["hidden"]) == hidden:
                    W1, b1, W2, b2 = old["W1"], old["b1"], old["W2"], old["b2"]
            except Exception:
                pass
        x = np.asarray([v for v, _ in samples], dtype=np.float32)
        y = np.asarray([[max(-1.0, min(1.0, float(r)))] for _, r in samples], dtype=np.float32)
        lr = max(1e-5, float(settings.experience_policy_lr))
        epochs = max(1, int(settings.experience_policy_epochs))
        last_loss = 0.0
        for _ in range(epochs):
            h = np.tanh(x @ W1 + b1)
            pred = np.tanh(h @ W2 + b2)
            err = pred - y
            last_loss = float(np.mean(err * err))
            # MSE + tanh derivatives; full-batch SGD is stable for this tiny personal network.
            d2 = (2.0 / len(x)) * err * (1.0 - pred * pred)
            gW2 = h.T @ d2
            gb2 = np.sum(d2, axis=0, keepdims=True)
            dh = d2 @ W2.T
            d1 = dh * (1.0 - h * h)
            gW1 = x.T @ d1
            gb1 = np.sum(d1, axis=0, keepdims=True)
            # Conservative clipping to avoid exploding updates after unusual feedback.
            for grad in (gW1, gb1, gW2, gb2):
                np.clip(grad, -1.0, 1.0, out=grad)
            W1 -= lr * gW1
            b1 -= lr * gb1
            W2 -= lr * gW2
            b2 -= lr * gb2
        buf = io.BytesIO()
        np.savez_compressed(buf, W1=W1, b1=b1, W2=W2, b2=b2, input_dim=np.asarray(dim), hidden=np.asarray(hidden))
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, path)
        meta = {"samples": len(samples), "input_dim": dim, "last_loss": last_loss, "backend": "numpy"}
        self._atomic_json(meta_path, meta)
        return {"ok": True, **self.status(user_id)}

    def train_user(self, user_id: str) -> dict[str, Any]:
        if not self.enabled:
            return {**self.status(user_id), "ok": False, "reason": "experience policy disabled"}
        with self._train_lock:
            try:
                samples, dim = self._prepare_samples(user_id)
            except ValueError as exc:
                count = len(self.experiences.rated_vector_samples(user_id, int(settings.experience_policy_max_samples)))
                return {**self.status(user_id), "ok": False, "reason": str(exc), "samples": count}

            torch, nn = self._torch()
            if torch is None or nn is None:
                return self._train_numpy(user_id, samples, dim)

            torch.manual_seed(42)
            device = torch.device("cuda" if torch.cuda.is_available() and settings.experience_policy_use_gpu else "cpu")
            model = self._build_model(nn, dim).to(device)
            model_path, meta_path = self._paths(user_id)
            if model_path.is_file():
                try:
                    payload = torch.load(model_path, map_location=device)
                    if int(payload.get("input_dim", -1)) == dim:
                        model.load_state_dict(payload["state_dict"])
                except Exception:
                    pass

            x = torch.tensor([v for v, _ in samples], dtype=torch.float32, device=device)
            y = torch.tensor([[max(-1.0, min(1.0, float(r)))] for _, r in samples], dtype=torch.float32, device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings.experience_policy_lr), weight_decay=1e-4)
            loss_fn = nn.SmoothL1Loss()
            epochs = max(1, int(settings.experience_policy_epochs))
            model.train()
            last_loss = 0.0
            for _ in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                last_loss = float(loss.detach().cpu().item())

            payload = {"state_dict": model.cpu().state_dict(), "input_dim": dim}
            tmp = model_path.with_suffix(".tmp")
            torch.save(payload, tmp)
            os.replace(tmp, model_path)
            meta = {"samples": len(samples), "input_dim": dim, "last_loss": last_loss, "backend": "torch"}
            self._atomic_json(meta_path, meta)
            return {"ok": True, **self.status(user_id)}

    async def predict(self, user_id: str, situation: str) -> dict[str, Any] | None:
        if not self.enabled or not situation.strip():
            return None
        model_path, meta_path = self._paths(user_id)
        numpy_path = self._numpy_path(user_id)
        if not meta_path.is_file():
            return None
        try:
            vectors = await self.experiences.embedder.embed([situation])
            if not vectors:
                return None
            vector = vectors[0]
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            dim = int(meta.get("input_dim", 0))
            if dim != len(vector):
                return None
            backend = str(meta.get("backend", ""))
            if backend == "numpy" or (numpy_path.is_file() and not model_path.is_file()):
                import numpy as np
                data = np.load(numpy_path)
                x = np.asarray([vector], dtype=np.float32)
                h = np.tanh(x @ data["W1"] + data["b1"])
                score = float(np.tanh(h @ data["W2"] + data["b2"])[0, 0])
            else:
                torch, nn = self._torch()
                if torch is None or nn is None or not model_path.is_file():
                    return None
                model = self._build_model(nn, dim)
                payload = torch.load(model_path, map_location="cpu")
                model.load_state_dict(payload["state_dict"])
                model.eval()
                with torch.no_grad():
                    score = float(model(torch.tensor([vector], dtype=torch.float32)).item())
            return {
                "predicted_reward": max(-1.0, min(1.0, score)),
                "samples": int(meta.get("samples", 0)),
                "last_loss": meta.get("last_loss"),
                "backend": backend or "torch",
            }
        except Exception:
            return None

