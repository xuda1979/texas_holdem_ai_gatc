from __future__ import annotations

import io
import json
import logging
import unittest

from poker_ai.systems.logging import LoggingSubsystem


class LoggingSubsystemTests(unittest.TestCase):
    def test_event_emits_structured_payload(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger("poker_ai.systems.logging")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        subsystem = LoggingSubsystem.create()
        subsystem.event("training.iteration.completed", iteration=3, generated_samples=2)

        handler.flush()
        logger.handlers.clear()
        payload = stream.getvalue().strip()
        event_name, encoded = payload.split(" | ", maxsplit=1)
        self.assertEqual(event_name, "training.iteration.completed")
        self.assertEqual(json.loads(encoded), {"generated_samples": 2, "iteration": 3})


if __name__ == "__main__":
    unittest.main()
