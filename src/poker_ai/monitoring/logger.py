import csv
import json
import os
import threading
import time
from typing import Dict, Optional


class MetricsLogger:
    """
    Minimal, dependency-free metric logger.
    - Writes JSONL to <log_dir>/metrics.jsonl
    - Optionally writes a rolling CSV (<log_dir>/metrics.csv) with the last seen value per metric.
    - Offers simple health-check hooks in poker_ai.monitoring.health
    """

    def __init__(self, log_dir: str, write_csv: bool = True) -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self._jsonl_path = os.path.join(self.log_dir, "metrics.jsonl")
        self._csv_path = os.path.join(self.log_dir, "metrics.csv")
        self._csv_enabled = write_csv
        self._csv_headers_written = False
        self._csv_state: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._t0 = time.time()

    def log(self, name: str, value: float, step: Optional[int] = None, extra: Optional[Dict] = None) -> None:
        rec = {
            "time": time.time(),
            "rel_time_sec": time.time() - self._t0,
            "metric": name,
            "value": float(value),
        }
        if step is not None:
            rec["step"] = int(step)
        if extra:
            rec.update(extra)
        line = json.dumps(rec, separators=(",", ":"))
        with self._lock:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            if self._csv_enabled:
                self._csv_state["time"] = rec["time"]
                self._csv_state["rel_time_sec"] = rec["rel_time_sec"]
                if step is not None:
                    self._csv_state["step"] = int(step)
                self._csv_state[name] = float(value)
                self._write_csv_row()

    def log_many(self, values: Dict[str, float], step: Optional[int] = None) -> None:
        for k, v in values.items():
            self.log(k, v, step=step)

    def _write_csv_row(self) -> None:
        if not self._csv_enabled:
            return
        # deterministic column order
        headers = sorted(self._csv_state.keys())
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not self._csv_headers_written:
                writer.writeheader()
                self._csv_headers_written = True
            writer.writerow({h: self._csv_state.get(h, "") for h in headers})

    def mark(self, tag: str, step: Optional[int] = None) -> None:
        """Convenience event marker."""
        self.log("event", 1.0, step=step, extra={"tag": tag})

    def path_jsonl(self) -> str:
        return self._jsonl_path

    def path_csv(self) -> str:
        return self._csv_path

