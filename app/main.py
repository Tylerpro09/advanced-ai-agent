from __future__ import annotations

import os
import asyncio
import logging
import re
import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.version import APP_NAME, APP_VERSION
from app.diagnostics import collect_diagnostics
from app.core.agent import Agent
from app.core.media import MediaClient
from app.core.memory import MemoryStore
from app.core.experience import ExperienceStore
from app.core.experience_policy import ExperiencePolicyNetwork
from app.core.continual_learning import AdapterRegistry, LoRAContinualTrainer
from app.core.neural_memory import NeuralMemoryStore
from app.core.rag import RAGStore

logger = logging.getLogger("advanced_ai_agent")

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Local-first AI agent with embedded GGUF, trainable experience network, versioned LoRA continual learning, neural/episodic memory, RAG and tools.",
)

memory = MemoryStore(settings.database_path)
neural_memory = NeuralMemoryStore(memory)
rag = RAGStore(settings.database_path)
experiences = ExperienceStore(memory, neural_memory.embedder)
agent = Agent(memory, rag, neural_memory, experiences)
experience_policy = agent.experience_policy
adapter_registry = AdapterRegistry()
lora_trainer = LoRAContinualTrainer(experiences, adapter_registry)
auto_lora_users: set[str] = set()


async def _auto_train_lora(user_id: str) -> None:
    try:
        result = await asyncio.to_thread(lora_trainer.train, user_id, True)
        adapter = result.get("adapter") or {}
        if result.get("activated") and hasattr(agent.provider, "set_adapter") and adapter.get("path"):
            agent.provider.set_adapter(adapter["path"])
    except Exception:
        # Auto-training failures must not break chat/feedback, but they must be observable.
        logger.exception("Automatic LoRA training failed for user %s", user_id)
    finally:
        auto_lora_users.discard(user_id)


@app.middleware("http")
async def optional_bearer_auth(request: Request, call_next):
    if settings.api_token and request.url.path.startswith("/v1/"):
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {settings.api_token}":
            return Response(content='{"detail":"Unauthorized"}', status_code=401, media_type="application/json")
    return await call_next(request)


vision = MediaClient(
    settings.vision_base_url or settings.ai_base_url,
    settings.vision_api_key or settings.ai_api_key,
    settings.vision_model or settings.ai_model,
    settings.ai_timeout_seconds,
)
stt = MediaClient(
    settings.stt_base_url or settings.ai_base_url,
    settings.stt_api_key or settings.ai_api_key,
    settings.stt_model,
    settings.ai_timeout_seconds,
)
tts = MediaClient(
    settings.tts_base_url or settings.ai_base_url,
    settings.tts_api_key or settings.ai_api_key,
    settings.tts_model,
    settings.ai_timeout_seconds,
)


def _main_backend_has_http_api() -> bool:
    return settings.model_backend.strip().lower() in {"openai", "openai_compatible", "remote", "server"}


def _vision_configured() -> bool:
    return bool((settings.vision_base_url and settings.vision_model) or (_main_backend_has_http_api() and settings.ai_base_url and (settings.vision_model or settings.ai_model)))


def _stt_configured() -> bool:
    return bool(settings.stt_base_url or (_main_backend_has_http_api() and settings.ai_base_url)) and bool(settings.stt_model)


def _tts_configured() -> bool:
    return bool(settings.tts_base_url or (_main_backend_has_http_api() and settings.ai_base_url)) and bool(settings.tts_model)


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    conversation_id: str = Field(default="main", min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=30000)


class MemoryCreate(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=2000)
    kind: str = Field(default="fact", max_length=32)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class RAGTextCreate(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    source: str = Field(min_length=1, max_length=240)
    text: str = Field(min_length=1, max_length=2_000_000)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    voice: str = Field(default=settings.tts_voice, min_length=1, max_length=64)


class ExperienceFeedback(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    reward: float = Field(ge=-1.0, le=1.0)
    outcome: str = Field(default="", max_length=1000)
    lesson: str = Field(default="", max_length=2000)


class ExperienceExport(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    min_reward: float = Field(default=0.5, ge=-1.0, le=1.0)


class LoRATrainRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    activate: bool = True


class AdapterActionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    force: bool = False



def _safe_name(name: str) -> str:
    name = os.path.basename(name or "upload.bin")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name[:180] or "upload.bin"


async def _save_upload(upload: UploadFile) -> str:
    Path(settings.uploads_dir).mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}_{_safe_name(upload.filename or 'upload.bin')}"
    path = Path(settings.uploads_dir) / name
    total = 0
    max_bytes = max(1, int(settings.max_upload_mb)) * 1024 * 1024
    try:
        with path.open("wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
                f.write(chunk)
        return str(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def _remove_temp(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove temporary upload %s", path, exc_info=True)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "version": APP_VERSION,
        "model": getattr(agent.provider, "model", settings.ai_model),
        "model_backend": settings.model_backend,
        "provider": settings.ai_base_url if settings.model_backend.lower().startswith("openai") else "in-process",
        "local_model": agent.provider.status() if hasattr(agent.provider, "status") else None,
        "features": {
            "memory": True,
            "neural_memory": neural_memory.stats(),
            "experience_memory": experiences.stats(),
            "experience_policy": {"enabled": settings.experience_policy_enabled},
            "continual_learning": {"enabled": settings.continual_learning_enabled, "base_model": settings.lora_base_model},
            "rag": True,
            "tools": settings.enable_tools,
            "plugins": settings.enable_plugins,
            "pc_tools": settings.enable_pc_tools,
            "web_search": bool(settings.searxng_url),
            "vision": _vision_configured(),
            "stt": _stt_configured(),
            "tts": _tts_configured(),
        },
    }


@app.get("/ready")
def readiness() -> dict:
    return collect_diagnostics(runtime=True)


@app.get("/v1/model/status")
def model_status() -> dict:
    provider = agent.provider
    if hasattr(provider, "status"):
        return provider.status()
    return {
        "backend": settings.model_backend,
        "model": getattr(provider, "model", settings.ai_model),
        "base_url": settings.ai_base_url,
        "loaded": True,
    }


@app.post("/v1/agent/chat")
async def chat(req: ChatRequest) -> dict:
    try:
        return await agent.chat(req.user_id, req.conversation_id, req.message)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI provider error: {exc}") from exc


@app.websocket("/v1/ws/chat")
async def ws_chat(ws: WebSocket):
    if settings.api_token and ws.headers.get("authorization", "") != f"Bearer {settings.api_token}" and ws.query_params.get("token") != settings.api_token:
        await ws.close(code=1008)
        return
    await ws.accept()
    try:
        while True:
            req = ChatRequest(**(await ws.receive_json()))
            await ws.send_json(await agent.chat(req.user_id, req.conversation_id, req.message))
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await ws.send_json({"error": str(exc)})
        await ws.close(code=1011)


@app.post("/v1/memory")
async def create_memory(req: MemoryCreate) -> dict:
    return {"id": await neural_memory.add_memory(req.user_id, req.text, req.kind, req.importance), "ok": True}


@app.get("/v1/memory/search")
async def search_memory(user_id: str, q: str = Query(default=""), limit: int = Query(default=8, ge=1, le=50)) -> dict:
    if q.strip():
        return {"items": [asdict(x) for x in await neural_memory.hybrid_search(user_id, q, limit)]}
    return {"items": [asdict(x) for x in memory.search_memories(user_id, q, limit)]}


@app.get("/v1/neural-memory/status")
def neural_memory_status(user_id: str | None = None) -> dict:
    return neural_memory.stats(user_id)


@app.post("/v1/neural-memory/reindex")
async def neural_memory_reindex(user_id: str) -> dict:
    try:
        return {"ok": True, **(await neural_memory.sync_user(user_id, limit=100000))}
    except Exception as exc:
        raise HTTPException(500, f"Neural memory indexing failed: {exc}") from exc


@app.get("/v1/neural-memory/search")
async def neural_memory_search(user_id: str, q: str, limit: int = Query(default=8, ge=1, le=50)) -> dict:
    try:
        return {"items": [asdict(x) for x in await neural_memory.search(user_id, q, limit)]}
    except Exception as exc:
        raise HTTPException(500, f"Neural memory search failed: {exc}") from exc


@app.get("/v1/memory")
def list_memory(user_id: str, limit: int = Query(default=100, ge=1, le=500)) -> dict:
    return {"items": [asdict(x) for x in memory.list_memories(user_id, limit)]}


@app.delete("/v1/memory/{memory_id}")
def delete_memory(memory_id: int, user_id: str) -> dict:
    if not memory.delete_memory(user_id, memory_id):
        raise HTTPException(404, "Memory not found")
    return {"ok": True}


@app.get("/v1/experiences")
def list_experiences(user_id: str, limit: int = Query(default=50, ge=1, le=500)) -> dict:
    return {"items": [asdict(x) for x in experiences.list(user_id, limit)]}


@app.get("/v1/experiences/search")
async def search_experiences(user_id: str, q: str, limit: int = Query(default=6, ge=1, le=50)) -> dict:
    return {"items": [asdict(x) for x in await experiences.search(user_id, q, limit)]}


@app.get("/v1/experiences/status")
def experience_status(user_id: str | None = None) -> dict:
    return experiences.stats(user_id)


@app.post("/v1/experiences/{experience_id}/feedback")
async def experience_feedback(experience_id: int, req: ExperienceFeedback) -> dict:
    ok = await experiences.feedback(req.user_id, experience_id, req.reward, req.outcome, req.lesson)
    if not ok:
        raise HTTPException(404, "Experience not found")
    policy = None
    if settings.experience_policy_enabled:
        # Small local update is intentionally performed after explicit feedback.
        policy = await asyncio.to_thread(experience_policy.train_user, req.user_id)

    auto_lora_queued = False
    every = max(0, int(settings.lora_auto_train_every))
    if (
        settings.continual_learning_enabled
        and every > 0
        and req.reward >= float(settings.lora_min_reward)
        and req.user_id not in auto_lora_users
    ):
        positive_count = len(experiences.training_examples(req.user_id, settings.lora_min_reward, 1_000_000))
        if positive_count >= int(settings.lora_min_examples) and positive_count % every == 0:
            auto_lora_users.add(req.user_id)
            asyncio.create_task(_auto_train_lora(req.user_id))
            auto_lora_queued = True
    return {
        "ok": True,
        "experience_id": experience_id,
        "experience_policy": policy,
        "auto_lora_queued": auto_lora_queued,
    }


@app.get("/v1/learning/policy/status")
def learning_policy_status(user_id: str) -> dict:
    return experience_policy.status(user_id)


@app.post("/v1/learning/policy/train")
async def learning_policy_train(req: AdapterActionRequest) -> dict:
    return await asyncio.to_thread(experience_policy.train_user, req.user_id)


@app.get("/v1/learning/status")
def learning_status(user_id: str) -> dict:
    return {
        "policy": experience_policy.status(user_id),
        "lora": lora_trainer.status(user_id),
        "model_backend": settings.model_backend,
    }


@app.post("/v1/learning/lora/train")
async def learning_lora_train(req: LoRATrainRequest) -> dict:
    try:
        result = await asyncio.to_thread(lora_trainer.train, req.user_id, req.activate)
        adapter = result.get("adapter") or {}
        # If this process is already running the direct PEFT backend, apply the freshly
        # activated adapter without changing the base model. Reload remains lazy.
        if result.get("activated") and hasattr(agent.provider, "set_adapter") and adapter.get("path"):
            agent.provider.set_adapter(adapter["path"])
        return result
    except Exception as exc:
        raise HTTPException(400, f"LoRA training failed: {exc}") from exc


@app.get("/v1/learning/lora/adapters")
def learning_lora_adapters(user_id: str) -> dict:
    return {"items": adapter_registry.list(user_id), "active": adapter_registry.active(user_id)}


@app.post("/v1/learning/lora/adapters/{version_id}/activate")
def learning_lora_activate(version_id: str, req: AdapterActionRequest) -> dict:
    try:
        candidate = adapter_registry.get(version_id)
        if candidate and not candidate.get("quality_passed", True) and not req.force:
            raise ValueError("This adapter failed the validation gate. Pass force=true only if you intentionally want to test it.")
        adapter = adapter_registry.activate(req.user_id, version_id)
        if hasattr(agent.provider, "set_adapter"):
            agent.provider.set_adapter(adapter["path"])
        return {"ok": True, "adapter": adapter}
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/v1/learning/lora/rollback")
def learning_lora_rollback(req: AdapterActionRequest) -> dict:
    try:
        adapter = adapter_registry.rollback(req.user_id)
        if hasattr(agent.provider, "set_adapter"):
            agent.provider.set_adapter(adapter["path"] if adapter else "")
        return {"ok": True, "adapter": adapter}
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/v1/experiences/export")
def export_experiences(req: ExperienceExport) -> dict:
    safe_user = re.sub(r"[^A-Za-z0-9._-]", "_", req.user_id)[:100]
    safe_user = safe_user or "user"
    path = settings.project_root / "data" / "experience_exports" / f"{safe_user}.jsonl"
    count = experiences.export_training_jsonl(req.user_id, str(path), req.min_reward)
    return {"ok": True, "items": count, "path": str(path)}


@app.post("/v1/knowledge/text")
def add_knowledge_text(req: RAGTextCreate) -> dict:
    return {"ok": True, "chunks": rag.ingest_text(req.user_id, req.source, req.text)}


@app.post("/v1/knowledge/upload")
async def upload_knowledge(user_id: str = Form(...), file: UploadFile = File(...)) -> dict:
    source_name = _safe_name(file.filename or "upload.bin")
    path = await _save_upload(file)
    try:
        chunks = rag.ingest_file(user_id, path, source=source_name)
        return {"ok": True, "source": source_name, "chunks": chunks}
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        _remove_temp(path)


@app.get("/v1/knowledge/search")
def knowledge_search(user_id: str, q: str, limit: int = Query(default=6, ge=1, le=30)) -> dict:
    return {"items": [asdict(x) for x in rag.search(user_id, q, limit)]}


@app.get("/v1/knowledge/sources")
def knowledge_sources(user_id: str) -> dict:
    return {"items": rag.list_sources(user_id)}


@app.delete("/v1/knowledge/source")
def delete_knowledge_source(user_id: str, source: str) -> dict:
    return {"ok": rag.delete_source(user_id, source)}


@app.post("/v1/vision/analyze")
async def analyze_image(
    prompt: str = Form(default="Describe and analyze this image in detail."),
    file: UploadFile = File(...),
) -> dict:
    if not _vision_configured():
        raise HTTPException(503, "Vision is not configured. Set VISION_BASE_URL and VISION_MODEL, or use an OpenAI-compatible main backend.")
    path = await _save_upload(file)
    try:
        return {"answer": await vision.vision(path, prompt)}
    except Exception as exc:
        raise HTTPException(502, f"Vision endpoint failed: {exc}") from exc
    finally:
        _remove_temp(path)


@app.post("/v1/audio/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> dict:
    if not _stt_configured():
        raise HTTPException(503, "STT is not configured. Set STT_BASE_URL/STT_MODEL, or use an OpenAI-compatible main backend that provides transcription.")
    path = await _save_upload(file)
    try:
        return {"text": await stt.transcribe(path)}
    except Exception as exc:
        raise HTTPException(502, f"STT endpoint failed: {exc}") from exc
    finally:
        _remove_temp(path)


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest) -> Response:
    if not _tts_configured():
        raise HTTPException(503, "TTS is not configured. Set TTS_BASE_URL/TTS_MODEL, or use an OpenAI-compatible main backend that provides speech.")
    try:
        data = await tts.speech(req.text, req.voice)
        return Response(content=data, media_type="audio/mpeg")
    except Exception as exc:
        raise HTTPException(502, f"TTS endpoint failed: {exc}") from exc


@app.post("/v1/plugins/reload")
def reload_plugins() -> dict:
    if not agent.plugins:
        raise HTTPException(400, "Plugins are disabled")
    agent.plugins.reload()
    return {"ok": True, "tools": list(agent.plugins.plugins)}


static_dir = settings.project_root / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def home():
    if (static_dir / "index.html").exists():
        return FileResponse(static_dir / "index.html")
    return {"name": APP_NAME, "version": APP_VERSION, "docs": "/docs"}


@app.get("/manifest.webmanifest")
def manifest():
    path = static_dir / "manifest.webmanifest"
    if not path.is_file():
        raise HTTPException(404, "Manifest not found")
    return FileResponse(path, media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    path = static_dir / "sw.js"
    if not path.is_file():
        raise HTTPException(404, "Service worker not found")
    return FileResponse(path, media_type="application/javascript")
