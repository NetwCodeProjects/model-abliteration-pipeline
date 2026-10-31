"""Append-only assessment records and the manifest status derived from them (docs/assessments.md).

A record is written once (exclusive create) to ``workspace/assessments/<artifact_id>/`` and copied
into the run. Manifests never hold measurements of their own: ``validation``/``inference``/
``evaluation`` are a fold of the records' ``derives`` over the manifest's current values, and
``derived_status`` names the record behind each part.
"""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any

from ..errors import ExitCode, PipelineError
from ..fsutil import atomic_write_json, canonical_json, read_json, sha256_obj
from ..telemetry import utc_now
from .schema_lite import load_schema, validate

SCHEMA_VERSION = "1.0"
KINDS = ("promotion", "verification", "comparison")
# derived_status part -> manifest fields it covers
PARTS = ("validation", "inference", "evaluation")
REDACTED = "unavailable (redacted)"


def store_dir(workspace: Path, artifact_id: str) -> Path:
    return workspace / "assessments" / artifact_id


def check(name: str, result: str, value: Any = None, threshold: Any = None, reason: str | None = None) -> dict[str, Any]:
    row = {"check": name, "result": result, "value": value, "threshold": threshold}
    if reason:
        row["reason"] = reason
    return row


def outcome_of(checks: list[dict[str, Any]]) -> str:
    """Weakest claim wins (Step 1): failed > unmeasured > passed; disabled checks are ignored."""
    results = {c["result"] for c in checks}
    if "failed" in results:
        return "failed"
    if "unmeasured" in results or not results - {"disabled"}:
        return "unmeasured"
    return "passed"


def environment_fingerprint(versions: dict[str, Any] | None, gpus: list[str] | None = None) -> str | None:
    if not versions:
        return None
    return sha256_obj({"versions": versions, "gpus": sorted(gpus or [])})


def new_record(
    *,
    artifact_id: str,
    role: str,
    run_id: str,
    stage_id: str,
    kind: str,
    results: dict[str, Any],
    checks: list[dict[str, Any]],
    outcome: str | None = None,
    suite: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
    gaps: list[dict[str, Any]] | None = None,
    redacted: bool = False,
    derives: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "role": role,
        "run_id": run_id,
        "stage_id": stage_id,
        "kind": kind,
        "created_utc": utc_now(),
        "suite": suite,
        "runtime": runtime or {"backend": None, "version": None, "environment_fingerprint": None},
        "generation_contract": contract,
        "contract_gaps": gaps or [],
        "redaction": {
            "redacted": redacted,
            "exact_text_replay": "not applicable" if contract is None else (REDACTED if redacted else "available"),
        },
        "results": results,
        "checks": checks,
        "outcome": outcome or outcome_of(checks),
        "derives": {part: (derives or {}).get(part) for part in PARTS},
    }
    record["assessment_id"] = "asm-" + sha256_obj(record)[:24]
    return record


def check_record(record: dict[str, Any]) -> None:
    errors = validate(record, load_schema("assessment.schema.json"))
    if errors:
        raise PipelineError(ExitCode.INTERNAL, "E_ASSESSMENT_INVALID", "Assessment record failed schema validation: " + "; ".join(errors[:10]))


def _create_exclusive(path: Path, data: bytes) -> None:
    """Write ``path`` completely or not at all, and never replace an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)  # atomic and fails if the target exists
        except FileExistsError:
            raise PipelineError(ExitCode.INTERNAL, "E_ASSESSMENT_EXISTS", f"Assessment record {path} already exists; records are never replaced") from None
        except OSError:  # no hard links on this filesystem: exclusive create, then write
            try:
                out = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                raise PipelineError(ExitCode.INTERNAL, "E_ASSESSMENT_EXISTS", f"Assessment record {path} already exists") from None
            with os.fdopen(out, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        Path(tmp).unlink(missing_ok=True)


def write_record(workspace: Path, run_dir: Path, record: dict[str, Any]) -> Path:
    check_record(record)
    data = (canonical_json(record) + "\n").encode("utf-8")
    path = store_dir(workspace, record["artifact_id"]) / f"{record['assessment_id']}.json"
    _create_exclusive(path, data)
    atomic_write_json(run_dir / "assessments" / path.name, record)
    return path


def records_for(workspace: Path, artifact_id: str) -> list[dict[str, Any]]:
    directory = store_dir(workspace, artifact_id)
    if not directory.is_dir():
        return []
    records = [read_json(p) for p in directory.glob("asm-*.json")]
    return sorted(records, key=lambda r: (r["created_utc"], r["assessment_id"]))


def ref(record: dict[str, Any]) -> dict[str, Any]:
    return {k: record[k] for k in ("assessment_id", "kind", "run_id", "stage_id", "created_utc", "outcome")}


def derive(manifest: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply every record's ``derives`` in time order to the manifest's current status.

    Idempotent: the current values either come from these records (re-applied in the same order) or
    from registration / a schema 1.0 manifest (kept until a record measures that part).
    """
    out = copy.deepcopy(manifest)
    status = dict(out.get("derived_status") or {part: None for part in PARTS})
    for record in records:
        derives = record.get("derives") or {}
        if derives.get("validation") is not None:
            out["validation"]["integrity_passed"] = derives["validation"]["integrity_passed"]
            status["validation"] = record["assessment_id"]
        if derives.get("inference") is not None:
            d = derives["inference"]
            out["validation"]["inference_passed"] = d["inference_passed"]
            out["validation"]["modalities_tested"] = list(d["modalities_tested"])
            out["inference"] = {k: d[k] for k in ("backend", "backend_revision", "tested_context")}
            status["inference"] = record["assessment_id"]
        if derives.get("evaluation") is not None:
            out["evaluation"] = {k: derives["evaluation"][k] for k in ("qualification_run_id", "metrics_provenance")}
            status["evaluation"] = record["assessment_id"]
    out["assessments"] = [ref(r) for r in records]
    out["derived_status"] = status
    return out
