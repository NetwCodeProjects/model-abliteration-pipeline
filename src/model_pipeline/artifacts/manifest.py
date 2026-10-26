"""Artifact manifests (schema 1.0) and the per-run artifact registry.

Manifests are written next to the artifact (``<dir>.manifest.json``), into the global
``<workspace>/manifests/<artifact_id>.json`` index and into the run's reports. They contain only
real values: unknown optional fields are ``null``; a manifest that fails schema validation is never
written.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..errors import ExitCode, PipelineError
from ..fsutil import InventoryEntry, atomic_write_json, inventory_digest, read_json, sha256_obj
from .schema_lite import load_schema, validate

SCHEMA_VERSION = "1.0"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "added_tokens.json",
)
PROCESSOR_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "image_processor_config.json",
    "feature_extractor_config.json",
)


def artifact_id_for(files: Iterable[InventoryEntry | dict[str, Any]]) -> str:
    return "art-" + inventory_digest(files)[:24]


def group_hash(files: list[dict[str, Any]], names: tuple[str, ...]) -> str | None:
    picked = sorted((f["relative_path"], f["sha256"]) for f in files if Path(f["relative_path"]).name in names and f.get("sha256"))
    return sha256_obj(picked) if picked else None


def build_manifest(
    *,
    run_id: str,
    model_family: str,
    source_model: str | None,
    source_revision: str | None,
    parent_artifact_id: str | None,
    kind: str,
    method: str,
    config_hash: str | None,
    tool_revision: str | None,
    precision: str,
    fmt: str,
    uri: str,
    files: list[dict[str, Any]],
    tokenizer: dict[str, Any] | None = None,
    processor: dict[str, Any] | None = None,
    quantization: dict[str, Any] | None = None,
    inference: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    extensions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if any(f.get("sha256") is None for f in files):
        raise PipelineError(ExitCode.VERIFY, "E_MANIFEST_INCOMPLETE", f"Refusing to build a manifest with unhashed files for {uri}")
    files = sorted(files, key=lambda f: f["relative_path"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": artifact_id_for(files),
        "parent_artifact_id": parent_artifact_id,
        "run_id": run_id,
        "model_family": model_family,
        "source_model": source_model,
        "source_revision": source_revision,
        "transformation": {"kind": kind, "method": method, "config_hash": config_hash, "tool_revision": tool_revision},
        "precision": precision,
        "artifact": {"format": fmt, "uri": uri, "files": files},
        "tokenizer": tokenizer or {"revision": source_revision, "hash": group_hash(files, TOKENIZER_FILES)},
        "processor": processor or {"revision": source_revision if group_hash(files, PROCESSOR_FILES) else None, "hash": group_hash(files, PROCESSOR_FILES)},
        "quantization": quantization or {"recipe_id": None, "calibration_revision": None, "imatrix_hash": None},
        "inference": inference or {"backend": None, "backend_revision": None, "tested_context": None},
        "validation": validation or {"integrity_passed": False, "inference_passed": None, "modalities_tested": []},
        "evaluation": evaluation or {"qualification_run_id": None, "metrics_provenance": "not-evaluated"},
    }
    if extensions:
        manifest["extensions"] = extensions
    check_manifest(manifest)
    return manifest


def check_manifest(manifest: dict[str, Any]) -> None:
    errors = validate(manifest, load_schema("artifact.schema.json"))
    if errors:
        raise PipelineError(ExitCode.VERIFY, "E_MANIFEST_INVALID", "Artifact manifest failed schema validation: " + "; ".join(errors[:10]))


class ArtifactRegistry:
    """Run-scoped index of artifacts (role -> manifest) plus the global manifest store."""

    def __init__(self, run_dir: Path, workspace: Path) -> None:
        self.index_path = run_dir / "artifacts.json"
        self.global_dir = workspace / "manifests"
        self.report_dir = run_dir / "reports" / "manifests"

    def _load(self) -> dict[str, Any]:
        return read_json(self.index_path) if self.index_path.exists() else {"artifacts": {}}

    def register(self, manifest: dict[str, Any], role: str, path: Path, stage_id: str, sidecar: bool = True) -> None:
        """Write the manifest; ``sidecar=False`` for directories we do not own (user paths, HF cache)."""
        check_manifest(manifest)
        artifact_id = manifest["artifact_id"]
        global_path = self.global_dir / f"{artifact_id}.json"
        atomic_write_json(global_path, manifest)
        manifest_path = global_path
        if sidecar:
            manifest_path = path.with_name(path.name + ".manifest.json")
            atomic_write_json(manifest_path, manifest)
        atomic_write_json(self.report_dir / f"{role.replace(':', '-')}.{artifact_id}.json", manifest)
        data = self._load()
        data["artifacts"][role] = {"artifact_id": artifact_id, "path": str(path), "manifest": str(manifest_path), "stage": stage_id}
        atomic_write_json(self.index_path, data)

    def update(self, role: str, manifest: dict[str, Any]) -> None:
        """Rewrite a registered manifest (e.g. validation results from stage 09); the ID must not change."""
        entry = self._load()["artifacts"].get(role)
        if entry is None:
            raise PipelineError(ExitCode.INTERNAL, "E_ARTIFACT_UNKNOWN", f"No artifact registered for role {role}")
        if entry["artifact_id"] != manifest["artifact_id"]:
            raise PipelineError(
                ExitCode.VERIFY, "E_ARTIFACT_ID_CHANGED", f"Artifact {role} content changed ({entry['artifact_id']} -> {manifest['artifact_id']})"
            )
        check_manifest(manifest)
        atomic_write_json(Path(entry["manifest"]), manifest)
        atomic_write_json(self.global_dir / f"{manifest['artifact_id']}.json", manifest)
        atomic_write_json(self.report_dir / f"{role.replace(':', '-')}.{manifest['artifact_id']}.json", manifest)

    def get(self, role: str) -> dict[str, Any] | None:
        entry = self._load()["artifacts"].get(role)
        if not entry:
            return None
        return {**entry, "manifest_data": read_json(entry["manifest"])} if Path(entry["manifest"]).exists() else None

    def roles(self) -> dict[str, Any]:
        return self._load()["artifacts"]

    def import_from(self, other_index: Path) -> None:
        if other_index.exists():
            data = self._load()
            for role, entry in read_json(other_index)["artifacts"].items():
                data["artifacts"].setdefault(role, entry)
            atomic_write_json(self.index_path, data)
