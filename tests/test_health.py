import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_healthz() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz() -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_contains_app_info() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "app_info" in response.text
