"""Fake vLLM smoke test.

FAKE_VLLM_FAIL=1 mimics the real A100 run (2026-10-29, a100-evidence): the weight-only FP8 fallback's
JIT build failed, so vLLM wrote no outputs. ``--suite`` answers the rendered comparison prompts like
the fake baseline (FAKE_DEGRADE degrades it). The result fields mirror scripts/runtime/vllm_smoke.py,
which reads vLLM's RequestOutput; that path has not run with real vLLM here (no GPU).
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "fake_bin"))
from _fakeutil import degraded, fake_answer  # noqa: E402

out = sys.argv[sys.argv.index("--out") + 1]
model = sys.argv[sys.argv.index("--model") + 1]
if os.environ.get("FAKE_VLLM_FAIL"):
    Path(out).write_text(json.dumps({"ok": False, "error": "RuntimeError: ninja is required to load C++ extensions"}))
else:
    result = {"ok": True, "outputs": ["Paris.", "def square(x):\n    return x * x", "391"], "load_seconds": 1.0}
    if "--suite" in sys.argv:
        suite = json.loads(Path(sys.argv[sys.argv.index("--suite") + 1]).read_text(encoding="utf-8"))
        result["suite"] = [
            {"id": s["id"], "response": fake_answer(s["rendered"], degraded(model)), "finish_reason": "stop", "new_tokens": 9, "prompt_tokens": 20}
            for s in suite
        ]
    Path(out).write_text(json.dumps(result))
