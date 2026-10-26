"""Regenerate schemas/pipeline.schema.json from model_pipeline.configspec (the single source of truth).

Usage: python scripts/generate_schema.py [--check]
--check exits 1 if the committed schema is out of date (used by the unit tests and CI).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_pipeline.configspec import to_json_schema  # noqa: E402

TARGET = ROOT / "schemas" / "pipeline.schema.json"


def render() -> str:
    return json.dumps(to_json_schema(), indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            print(f"{TARGET} is out of date; run python scripts/generate_schema.py")
            return 1
        return 0
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
