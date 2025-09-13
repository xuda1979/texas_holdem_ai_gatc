import json
import os

import pytest

requests = pytest.importorskip("requests")

BASE_URL = os.environ.get("E2E_BASE_URL")

@pytest.mark.skipif(BASE_URL is None, reason="E2E_BASE_URL not set")
def test_play_endpoint_creates_transcript():
    resp = requests.get(f"{BASE_URL}/play", timeout=5)
    resp.raise_for_status()
    data = resp.json()
    assert sum(data["final_stacks"]) == 200
