"""Fake vLLM smoke test.

FAKE_VLLM_FAIL=1 mimics the real A100 run (2026-10-29, a100-evidence): the weight-only FP8 fallback's
JIT build failed, so vLLM wrote no outputs.
"""

import json
import os
import sys
from pathlib import Path

out = sys.argv[sys.argv.index("--out") + 1]
if os.environ.get("FAKE_VLLM_FAIL"):
    Path(out).write_text(json.dumps({"ok": False, "error": "RuntimeError: ninja is required to load C++ extensions"}))
else:
    Path(out).write_text(json.dumps({"ok": True, "outputs": ["Paris.", "def square(x):\n    return x * x", "391"], "load_seconds": 1.0}))
