#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _spec(name: str) -> str:
    return repr(importlib.util.find_spec(name))


def probe_env() -> int:
    payload = {
        "python": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "env": {
            key: os.environ.get(key, "")
            for key in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "PATH")
        },
        "specs": {name: _spec(name) for name in ("tbe", "op_impl", "te", "torch", "torch_npu")},
        "sys_path": sys.path,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def probe_repro() -> int:
    result: dict[str, object] = {}
    try:
        from poker_ai.ai.models.transformer import AdvantageNetwork  # noqa: F401

        result["project_import"] = "ok"
    except Exception:
        result["project_import"] = traceback.format_exc()

    try:
        import torch

        result["torch"] = getattr(torch, "__version__", "unknown")
    except Exception:
        result["torch"] = traceback.format_exc()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1

    try:
        import torch_npu  # noqa: F401

        result["torch_npu"] = "ok"
    except Exception:
        result["torch_npu"] = traceback.format_exc()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1

    try:
        x = torch.zeros(1).to("npu")
        result["npu_tensor"] = str(x.device)
    except Exception:
        result["npu_tensor"] = traceback.format_exc()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("env", "repro"))
    args = parser.parse_args()
    if args.mode == "env":
        return probe_env()
    return probe_repro()


if __name__ == "__main__":
    raise SystemExit(main())
