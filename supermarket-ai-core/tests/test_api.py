"""Tests for the API layer: /health works, other routes are stubs (M0/M10)."""

import pytest


def test_health_endpoint():
    testclient = pytest.importorskip("fastapi.testclient")
    from src.api.main import app

    client = testclient.TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.skip(reason="M10: API endpoints not implemented yet")
def test_events_endpoint_returns_events():
    """GET /events should return persisted reviewable events."""


@pytest.mark.skip(reason="M10: API endpoints not implemented yet")
def test_person_journey_endpoint():
    """GET /persons/{global_id}/journey should return the camera history."""


@pytest.mark.skip(reason="M10: API endpoints not implemented yet")
def test_system_status_endpoint():
    """GET /system/status should report camera health and processing rates."""
