"""Provision real CPU tool components with the platform's own installer (for tests/realtools).

Usage: REALTOOLS_WS=/path/ws python tests/realtools/provision_cpu.py [heretic] [fp8]
Uses environment.torch_backend=cpu so no GPU is required. Network access is required.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llm_surgeon.config import build_config  # noqa: E402
from llm_surgeon.environment.components import ComponentManager, component_specs, find_uv  # noqa: E402
from llm_surgeon.environment.detect import HostProbe  # noqa: E402
from llm_surgeon.subprocesses import CommandRunner  # noqa: E402
from llm_surgeon.telemetry import EventLogger, Redactor  # noqa: E402


def main() -> int:
    workspace = Path(os.environ["REALTOOLS_WS"])
    names = sys.argv[1:] or ["heretic"]
    config = build_config(None, {"model.url": "tiny-random/qwen3.8", "environment.torch_backend": "cpu", "storage.root": str(workspace)})
    probe = HostProbe(allow_exec=True)
    specs = component_specs(config.data, {"source": "pip", "backend": "cpu"}, probe)
    runner = CommandRunner(Redactor(), threading.Event())
    manager = ComponentManager(
        workspace / "tools",
        runner,
        EventLogger(None, Redactor(), "info"),
        workspace / "logs",
        find_uv(None),
        ROOT / "scripts" / "runtime",
        workspace / "cache",
        probe,
    )
    for name in names:
        record, action = manager.ensure(name, specs[name], allow_install=True, gpu_smoke=False)
        print(name, action, json.dumps(record.versions)[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())
