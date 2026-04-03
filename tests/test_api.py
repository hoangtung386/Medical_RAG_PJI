"""Smoke tests for the FastAPI endpoints (no LLM required)."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def _make_client() -> TestClient:
    """Create a TestClient with mocked RAG/PJI dependencies."""
    # Patch heavy dependencies so tests run without API keys
    with (
        patch("app.main.SharedResources"),
        patch("app.main.AdaptiveRAG"),
        patch("app.main.PJIRecommendationEngine"),
    ):
        from app.main import app
        return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        client = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestProcessSnapshot:
    def test_returns_503_when_engine_not_ready(self):
        """When pji_engine is None the endpoint should return 503."""
        with (
            patch("app.main.SharedResources"),
            patch("app.main.AdaptiveRAG"),
            patch("app.main.PJIRecommendationEngine"),
        ):
            from app.main import app
            # Explicitly set engine to None
            app.state.pji_engine = None
            client = TestClient(app)

        resp = client.post(
            "/api/v1/process-snapshot",
            json={
                "request_id": "test-1",
                "episode_id": 1,
                "snapshot_id": 1,
                "snapshot_data_json": {},
            },
        )
        assert resp.status_code == 503


class TestChat:
    def test_empty_question_returns_400(self):
        """An empty question should be rejected."""
        with (
            patch("app.main.SharedResources"),
            patch("app.main.AdaptiveRAG"),
            patch("app.main.PJIRecommendationEngine"),
        ):
            from app.main import app
            # Need a non-None engine to pass dependency check
            app.state.pji_engine = MagicMock()
            client = TestClient(app)

        resp = client.post(
            "/api/v1/chat",
            json={"question": "   "},
        )
        assert resp.status_code == 400
