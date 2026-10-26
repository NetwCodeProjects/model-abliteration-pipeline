"""Fake vLLM smoke test."""

import json
import sys
from pathlib import Path

out = sys.argv[sys.argv.index("--out") + 1]
Path(out).write_text(json.dumps({"ok": True, "outputs": ["Paris.", "def square(x):\n    return x * x", "391"], "load_seconds": 1.0}))
