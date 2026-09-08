from __future__ import annotations

import time
import uuid

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app, experiences


def main() -> None:
    # The v2.8.1 fast route must be registered before the legacy compatibility route.
    routes = [
        r for r in app.routes
        if getattr(r, "path", "") == "/v1/experiences/{experience_id}/feedback"
        and "POST" in getattr(r, "methods", set())
    ]
    assert routes, "feedback route is missing"
    assert routes[0].endpoint.__module__ == "app.v281_feedback", routes[0].endpoint.__module__

    user_id = "feedback-test-" + uuid.uuid4().hex
    now = time.time()
    conn = experiences._conn()  # intentional white-box regression test
    cur = conn.execute(
        """
        INSERT INTO experiences(user_id,conversation_id,situation,action,tool_log,outcome,reward,lesson,created_at,updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,?)
        """,
        (user_id, "main", "hola", "hey", "[]", "unrated", 0.0, "", now, now),
    )
    experience_id = int(cur.lastrowid)
    conn.commit()

    old_human = settings.human_like_learning_enabled
    old_dev = settings.developmental_learning_enabled
    old_policy = settings.experience_policy_enabled
    old_continual = settings.continual_learning_enabled
    settings.human_like_learning_enabled = False
    settings.developmental_learning_enabled = False
    settings.experience_policy_enabled = False
    settings.continual_learning_enabled = False

    try:
        with TestClient(app) as client:
            response = client.post(
                f"/v1/experiences/{experience_id}/feedback",
                json={
                    "user_id": user_id,
                    "reward": -1.0,
                    "outcome": "No encajó",
                    "lesson": "Cambiar el enfoque",
                },
            )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["ok"] is True
        assert payload["saved"] is True
        assert payload["learning_queued"] is True

        row = conn.execute(
            "SELECT reward,outcome,lesson FROM experiences WHERE user_id=? AND id=?",
            (user_id, experience_id),
        ).fetchone()
        assert row is not None
        assert float(row["reward"]) == -1.0
        assert str(row["outcome"]) == "No encajó"
        assert str(row["lesson"]) == "Cambiar el enfoque"
    finally:
        settings.human_like_learning_enabled = old_human
        settings.developmental_learning_enabled = old_dev
        settings.experience_policy_enabled = old_policy
        settings.continual_learning_enabled = old_continual
        conn.execute("DELETE FROM experiences WHERE user_id=?", (user_id,))
        conn.commit()

    print("v2.8.1 feedback tests passed")


if __name__ == "__main__":
    main()
