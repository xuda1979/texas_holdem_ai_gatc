"""Minimal FastAPI application exposing health and metrics endpoints."""

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Info, generate_latest

app = FastAPI()


# Expose application metadata via Prometheus so tests can assert on it.
APP_INFO = Info("app_info", "Application info")
APP_INFO.info({"app": "texas_holdem_ai_gatc"})


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Return a simple liveness indicator."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    """Return a simple readiness indicator."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics for the application."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
