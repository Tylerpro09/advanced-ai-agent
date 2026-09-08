from __future__ import annotations

import os
import tempfile
from pathlib import Path

_tmp = tempfile.TemporaryDirectory(prefix="aaa_v290_test_")
root = Path(_tmp.name)
os.environ["DATABASE_PATH"] = str(root / "agent.db")
os.environ["UPLOADS_DIR"] = str(root / "uploads")
os.environ["EXPERIENCE_POLICY_DIR"] = str(root / "policy")
os.environ["LORA_OUTPUT_DIR"] = str(root / "adapters")
os.environ["MODEL_BACKEND"] = "openai_compatible"
os.environ["AI_BASE_URL"] = "http://127.0.0.1:9/v1"
os.environ["AI_MODEL"] = "auto"
os.environ["NEURAL_MEMORY_ENABLED"] = "false"
os.environ["HUMAN_REFLECTION_ENABLED"] = "false"
os.environ["CONTINUAL_LEARNING_ENABLED"] = "false"

from fastapi.testclient import TestClient
from app.main import app, experiences, memory
from app.version import APP_VERSION


client = TestClient(app)
user = "v290-user"
conv = "main"

# Seed one answer exactly as the chat path would.
conn = experiences._conn()
cur = conn.execute(
    """
    INSERT INTO experiences(user_id,conversation_id,situation,action,tool_log,outcome,reward,lesson,created_at,updated_at)
    VALUES(?,?,?,?,?,?,?,?,strftime('%s','now'),strftime('%s','now'))
    """,
    (user, conv, "¿Cuál es la capital de Ecuador?", "Guayaquil", "[]", "unrated", 0.0, ""),
)
conn.commit()
original_id = int(cur.lastrowid)

r = client.post(
    f"/v1/experiences/{original_id}/correction",
    json={
        "user_id": user,
        "corrected_answer": "La capital de Ecuador es Quito.",
        "reason": "La respuesta anterior confundió la capital con otra ciudad.",
    },
)
assert r.status_code == 200, r.text
payload = r.json()
assert payload["saved"] is True
correction_id = int(payload["correction_experience_id"])

old = conn.execute("SELECT reward,outcome FROM experiences WHERE id=?", (original_id,)).fetchone()
new = conn.execute("SELECT reward,outcome,action FROM experiences WHERE id=?", (correction_id,)).fetchone()
assert float(old["reward"]) == -1.0
assert old["outcome"] == "corrected_by_user"
assert float(new["reward"]) == 1.0
assert new["outcome"] == "explicit_user_correction"
assert "Quito" in new["action"]

r = client.get("/v1/brain/dashboard", params={"user_id": user})
assert r.status_code == 200, r.text
dashboard = r.json()
assert dashboard["version"] == APP_VERSION == "2.9.0"
assert dashboard["experiences"]["rated"] >= 2
assert "developmental" in dashboard
assert "model" in dashboard

# New chat clears only short conversation messages.
memory.add_memory(user, "Me gusta que las respuestas sean directas.", "preference", 0.9)
memory.add_message(user, conv, "user", "mensaje temporal")
memory.add_message(user, conv, "assistant", "respuesta temporal")
r = client.post(f"/v1/conversations/{conv}/clear", json={"user_id": user})
assert r.status_code == 200, r.text
assert r.json()["deleted_messages"] >= 2
assert memory.recent_messages(user, conv, 10) == []
assert any("directas" in x.text for x in memory.list_memories(user, 10))

# Consolidation is allowed even with too few samples: it should return an inspectable policy result.
r = client.post("/v1/brain/consolidate", json={"user_id": user})
assert r.status_code == 200, r.text
consolidated = r.json()
assert consolidated["ok"] is True
assert consolidated["exported_examples"] >= 1
assert "policy" in consolidated

print("v2.9 feature tests passed")
