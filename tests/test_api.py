"""API smoke tests via FastAPI TestClient (no real GPU/model needed for /health)."""

import pytest


@pytest.fixture
def client():
    # Importing api triggers torch import; that's OK for /health.
    from fastapi.testclient import TestClient

    from api.app import app

    return TestClient(app)


class TestHealth:
    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert "device" in body

    def test_gpu_status_endpoint(self, client):
        r = client.get("/gpu-status")
        assert r.status_code == 200
        body = r.json()
        # Pool may not be initialized in unit tests; the route still returns a dict.
        assert "total_gpus" in body
        assert "gpus" in body


class TestAnalyzeFurniture:
    def test_invalid_base64_returns_400(self, client):
        r = client.post("/analyze-furniture-base64", json={"image": "not-base64!@#"})
        assert r.status_code == 400
        assert "Invalid" in r.json().get("error", "")

    def test_empty_image_urls_rejected(self, client):
        # Pydantic min_length=1 → 422
        r = client.post("/analyze-furniture", json={"estimate_id": 1, "image_urls": []})
        assert r.status_code == 422

    def test_too_many_image_urls_rejected(self, client):
        urls = [{"id": i, "url": f"https://x/{i}.jpg"} for i in range(21)]
        r = client.post(
            "/analyze-furniture", json={"estimate_id": 1, "image_urls": urls}
        )
        assert r.status_code == 422
