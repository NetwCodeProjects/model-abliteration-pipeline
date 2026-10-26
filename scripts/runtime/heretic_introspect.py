"""Introspect the installed Heretic (runs inside the heretic environment).

Reports the Settings fields of the *installed* version so stage 04 can reject unknown native keys,
plus which modifier/scorer plugins are importable. Never imports heretic.main (it parses argv).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import sys
import traceback

PLUGINS = {
    "modifiers": ["heretic.modifiers.ara.ARA", "heretic.modifiers.abliteration.Abliteration"],
    "scorers": [
        "heretic.scorers.keyword_rate.KeywordRate",
        "heretic.scorers.kl_divergence.KLDivergence",
        "heretic.scorers.benchmark_score.BenchmarkScore",
    ],
}


def version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def importable(path: str) -> bool:
    module, _, cls = path.rpartition(".")
    try:
        return hasattr(importlib.import_module(module), cls)
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result: dict = {"ok": False}
    try:
        from heretic.config import Settings

        dist = md.distribution("heretic-llm")
        direct = dist.read_text("direct_url.json")
        direct_url = json.loads(direct) if direct else None
        commit = ((direct_url or {}).get("vcs_info") or {}).get("commit_id")
        result.update(
            ok=True,
            heretic_version=dist.version,
            direct_url={"raw": direct_url, "commit": commit},
            settings_fields=sorted(Settings.model_fields),
            excluded_fields=sorted(n for n, f in Settings.model_fields.items() if f.exclude),
            modifiers={p: importable(p) for p in PLUGINS["modifiers"]},
            scorers={p: importable(p) for p in PLUGINS["scorers"]},
            versions={d: version(d) for d in ("torch", "transformers", "optuna", "peft", "accelerate", "datasets")},
        )
    except Exception as exc:  # noqa: BLE001
        result.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-3000:])
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
