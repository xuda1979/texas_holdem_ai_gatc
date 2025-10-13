"""Minimal FastAPI application exposing health and metrics endpoints."""

# Ensure src/ imports and optional deterministic seeding for local runs.
import poker_ai_bootstrap as _pab

_pab.seed_all()  # no-op unless RUN_DETERMINISTIC=1 or SEED is set
del _pab

from pathlib import Path
import json
import random

from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Info, generate_latest

from poker_ai.engine.texas_holdem_simple import TexasHoldem

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


class MoveRequest(BaseModel):
    """Request model for the /v1/move endpoint."""

    state: dict | None = None


@app.post("/v1/move")
def v1_move(_: MoveRequest) -> dict[str, str]:
    """Return a dummy move for benchmarking purposes."""
    return {"action": "check"}


def _simulate_hand(seed: int = 0) -> dict:
    """Play a deterministic two-player hand and return a transcript."""

    random.seed(seed)
    game = TexasHoldem(2)
    transcript: dict[str, list] = {
        "players_hands": game.players_hands.copy(),
        "actions": [],
    }
    for _ in range(4):
        for player in range(2):
            game.apply_action(player, "check")
            transcript["actions"].append({"player": player, "action": "check"})
        game.next_betting_round()

    winners = game.get_winner()
    transcript["community_cards"] = game.community_cards
    transcript["winners"] = winners

    starting_stack = 100
    final_stacks = [starting_stack - bet for bet in game.bets]
    transcript["final_stacks"] = final_stacks
    return transcript


@app.get("/play")
def play() -> dict:
    """Simulate a short hand and persist its transcript to disk."""

    transcript = _simulate_hand()
    Path("transcript.json").write_text(json.dumps(transcript))
    return transcript
