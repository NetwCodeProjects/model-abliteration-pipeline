"""Published-baseline ingestion. No model is downloaded or run here.

A baseline file records numbers *as published* (model cards) together with their source URL,
retrieval date, evaluator description and a comparability caveat. They are never presented as
local measurements.
"""

from __future__ import annotations

from typing import Any

from ..errors import ExitCode, PipelineError

REQUIRED_REFERENCE_KEYS = ("id", "repo", "revision", "source_url", "retrieved_utc", "evaluator", "comparability_caveat", "metrics")
REQUIRED_METRIC_KEYS = ("name", "value", "model_role")


def validate_baselines(data: dict[str, Any]) -> list[str]:
    errors = []
    if str(data.get("schema_version")) != "1.0":
        errors.append("schema_version must be '1.0'")
    base = data.get("base_model") or {}
    for key in ("repo", "revision"):
        if not base.get(key):
            errors.append(f"base_model.{key} is required")
    refs = data.get("references")
    if not isinstance(refs, list) or not refs:
        errors.append("references must be a non-empty list")
        return errors
    for i, ref in enumerate(refs):
        for key in REQUIRED_REFERENCE_KEYS:
            if ref.get(key) in (None, "", []):
                errors.append(f"references[{i}].{key} is required (provenance is mandatory)")
        for j, metric in enumerate(ref.get("metrics") or []):
            for key in REQUIRED_METRIC_KEYS:
                if key not in metric:
                    errors.append(f"references[{i}].metrics[{j}].{key} is required")
            if metric.get("model_role") not in ("reference", "original"):
                errors.append(f"references[{i}].metrics[{j}].model_role must be 'reference' or 'original'")
    return errors


def ingest(data: dict[str, Any], source_repo: str | None, source_revision: str | None) -> dict[str, Any]:
    errors = validate_baselines(data)
    if errors:
        raise PipelineError(ExitCode.CONFIG, "E_BASELINES_INVALID", "Published baseline file is invalid: " + "; ".join(errors))
    base = data["base_model"]
    comparability = []
    if source_repo and base["repo"] != source_repo:
        comparability.append(f"Baselines describe {base['repo']} but this run uses {source_repo}: numbers are NOT comparable.")
    if source_revision and base["revision"] != source_revision:
        comparability.append(f"Baselines were published for revision {base['revision']}, this run resolved {source_revision}.")
    if source_repo is None:
        comparability.append("Run uses a locally supplied model; relation to the published base model is unknown.")
    return {
        "provenance": "published (copied from cited sources; not measured by this platform)",
        "local_inference_runs": 0,
        "reference_checkpoints_downloaded": 0,
        "base_model": base,
        "references": data["references"],
        "comparability_warnings": comparability,
    }


def published_method(data: dict[str, Any], reference_repo: str | None) -> dict[str, Any] | None:
    for ref in data.get("references", []):
        if reference_repo is None or ref.get("repo") == reference_repo:
            method = ref.get("published_method")
            if method:
                return {**method, "source_url": ref.get("source_url"), "reference_repo": ref.get("repo"), "reference_revision": ref.get("revision")}
    return None
