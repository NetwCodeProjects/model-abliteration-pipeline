"""Stage 03 - published_baselines: ingest model-card metrics with provenance. Zero inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import load_yaml
from ..evaluation.baselines import ingest
from ..fsutil import atomic_write_json, sha256_file
from .base import Stage, StageContext


class BaselinesStage(Stage):
    id = "03"
    name = "published_baselines"
    title = "Published baselines (no local inference)"
    config_keys = ("evaluation.baseline_mode", "evaluation.published_baselines")

    def dependencies(self, config) -> tuple[str, ...]:
        return ("02",)

    def applicable(self, config) -> tuple[bool, str]:
        if not config.get("abliteration.enabled"):
            return False, "abliteration disabled (quantize-only)"
        if not config.get("evaluation.published_baselines"):
            return False, "evaluation.published_baselines not set"
        return True, ""

    def extra_fingerprint(self, run) -> dict[str, Any]:
        path = run.config.get("evaluation.published_baselines")
        return {"baselines_sha256": sha256_file(path) if path and Path(path).is_file() else None}

    def target_outputs(self, config) -> list[str]:
        return ["runs/<run>/metrics/published_baselines.json (provenance: published, 0 inference runs)"]

    def execute(self, ctx: StageContext) -> None:
        path = Path(ctx.config.get("evaluation.published_baselines"))
        data = load_yaml(path)
        source = ctx.run.load_receipt("02") or {}
        diag = source.get("diagnostics", {})
        result = ingest(data, diag.get("source_repo"), diag.get("source_revision"))
        result["file"] = {"path": str(path), "sha256": sha256_file(path)}
        for warning in result["comparability_warnings"]:
            ctx.warn(warning)
        out = ctx.metrics_path("published_baselines.json")
        atomic_write_json(out, result)
        ctx.add_output(out, "published_baselines")
        ctx.diagnostics.update(references=[r["id"] for r in result["references"]], local_inference_runs=0, reference_checkpoints_downloaded=0)
