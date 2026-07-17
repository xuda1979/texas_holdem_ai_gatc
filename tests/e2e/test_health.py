"""End-to-end smoke tests for the FastAPI service health endpoints.

These run against a live container started by `.github/workflows/e2e-tests.yml`
or any service exposing `E2E_BASE_URL`. They are skipped locally when
`E2E_BASE_URL` is not set.
"""

from __future__ import annotations

import os

import pytest

requests = pytest.importorskip("requests")

BASE_URL = os.environ.get("E2E_BASE_URL")


@pytest.mark.skipif(BASE_URL is None, reason="E2E_BASE_URL not set")
def test_healthz_returns_ok() -> None:
    """`/healthz` should report liveness."""
    resp = requests.get(f"{BASE_URL}/healthz", timeout=5)
    resp.raise_for_status()
    assert resp.json() == {"status": "ok"}


@pytest.mark.skipif(BASE_URL is None, reason="E2E_BASE_URL not set")
def test_readyz_returns_ok() -> None:
    """`/readyz` should report readiness."""
    resp = requests.get(f"{BASE_URL}/readyz", timeout=5)
    resp.raise_for_status()
    assert resp.json() == {"status": "ok"}


@pytest.mark.skipif(BASE_URL is None, reason="E2E_BASE_URL not set")
def test_metrics_exposes_prometheus() -> None:
    """`/metrics` should return Prometheus-formatted text."""
    resp = requests.get(f"{BASE_URL}/metrics", timeout=5)
    resp.raise_for_status()
    assert "text/plain" in resp.headers.get("content-type", "")
    # Prometheus exposition always contains these HELP/TYPE lines from
    # prometheus_client's default registry.
    body = resp.text
    assert "# HELP" in body
    assert "# TYPE" in body
