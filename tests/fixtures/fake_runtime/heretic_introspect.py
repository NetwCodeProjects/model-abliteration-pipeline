"""Fake introspection returning the REAL Settings fields extracted from Heretic 171bb98b."""

import json
import sys
from pathlib import Path

fixture = json.loads((Path(__file__).parents[1] / "heretic_settings_171bb98b.json").read_text())
out = sys.argv[sys.argv.index("--out") + 1]
Path(out).write_text(
    json.dumps(
        {
            "ok": True,
            "heretic_version": "2.0.0.dev0",
            "direct_url": {"raw": None, "commit": None},
            "settings_fields": fixture["settings_fields"],
            "excluded_fields": fixture["excluded_fields"],
            "modifiers": {"heretic.modifiers.ara.ARA": True, "heretic.modifiers.abliteration.Abliteration": True},
            "scorers": {"heretic.scorers.keyword_rate.KeywordRate": True, "heretic.scorers.kl_divergence.KLDivergence": True},
        }
    )
)
