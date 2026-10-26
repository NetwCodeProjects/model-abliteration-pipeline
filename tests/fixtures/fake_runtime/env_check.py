"""Fake env check: always healthy (component validation path is exercised, not the GPU)."""

import json
import sys

print(json.dumps({"ok": True, "component": sys.argv[sys.argv.index("--component") + 1], "fake": True}))
