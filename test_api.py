from __future__ import annotations

import os
import tempfile
from pathlib import Path


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        os.environ["DATABASE_PATH"] = str(root / "api.db")
        os.environ["UPLOADS_DIR"] = str(root / "uploads")
        os.environ["EXPERIENCE_POLICY_DIR"] = str(root / "policy")
        os.environ["LORA_OUTPUT_DIR"] = str(root / "adapters")
        os.environ["API_TOKEN"] = "test-token"
        os.environ["EMBEDDING_PROVIDER"] = "off"
        os.environ["NEURAL_MEMORY_ENABLED"] = "false"
        os.environ["MODEL_BACKEND"] = "embedded_gguf"
        os.environ["LOCAL_MODEL_PATH"] = str(root / "missing.gguf")
        os.environ["VISION_BASE_URL"] = ""
        os.environ["VISION_MODEL"] = ""
        os.environ["STT_BASE_URL"] = ""
        os.environ["TTS_BASE_URL"] = ""

        from fastapi.testclient import TestClient
        from app.main import app
        from app.version import APP_VERSION

        client = TestClient(app)
        health = client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json()["version"] == APP_VERSION
        assert client.get("/").status_code == 200
        assert client.get("/manifest.webmanifest").status_code == 200

        assert client.get("/v1/model/status").status_code == 401
        auth = {"Authorization": "Bearer test-token"}
        status = client.get("/v1/model/status", headers=auth)
        assert status.status_code == 200, status.text
        assert status.json()["model_exists"] is False

        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["runtime_mode"] is True
        assert ready.json()["ok"] is False  # no GGUF/runtime dependency in this isolated test

        files = {"file": ("note.txt", b"La nave Aurora usa un reactor de fusion.", "text/plain")}
        data = {"user_id": "tester"}
        upload = client.post("/v1/knowledge/upload", headers=auth, data=data, files=files)
        assert upload.status_code == 200, upload.text
        assert upload.json()["chunks"] >= 1
        upload_dir = root / "uploads"
        assert upload_dir.exists()
        assert list(upload_dir.iterdir()) == [], "temporary RAG upload was not cleaned"

        vision = client.post(
            "/v1/vision/analyze",
            headers=auth,
            data={"prompt": "describe"},
            files={"file": ("x.png", b"not-a-real-image", "image/png")},
        )
        assert vision.status_code == 503, vision.text

    print("API tests passed")


if __name__ == "__main__":
    main()
