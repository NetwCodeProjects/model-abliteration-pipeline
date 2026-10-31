"""Stage contract: prerequisites -> execute -> validate_outputs -> atomic receipt.

Receipts are the only evidence used for resume decisions. A stage is reused only when its
receipt succeeded, its ``stage_key`` (config subset + upstream receipt keys + extra inputs) is
unchanged, and ``validate_outputs`` still passes. Directory existence alone never counts.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import ExitCode, PipelineError
from ..fsutil import atomic_write_json, read_json, remove_tree, sha256_obj
from ..telemetry import utc_now

if TYPE_CHECKING:  # pragma: no cover
    from ..config import ResolvedConfig
    from ..orchestrator import Run

RECEIPT_SCHEMA_VERSION = "1.0"
SUCCESS = "succeeded"
FAILED = "failed"
SKIPPED = "skipped"
BLOCKED = "blocked"
INTERRUPTED = "interrupted"
RUNNING = "running"


@dataclass
class StageContext:
    run: Run
    stage: Stage
    log_dir: Path
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    input_artifacts: list[str] = field(default_factory=list)
    output_artifacts: list[str] = field(default_factory=list)
    tool_revisions: dict[str, Any] = field(default_factory=dict)
    partial: bool = False

    @property
    def config(self):
        return self.run.config

    @property
    def logger(self):
        return self.run.logger

    def warn(self, message: str, **fields: Any) -> None:
        self.warnings.append(message)
        self.logger.warning(f"stage.{self.stage.id}.warning", message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        self.logger.info(f"stage.{self.stage.id}", message, **fields)

    def add_output(self, path: Path | str, kind: str, sha256: str | None = None, size: int | None = None) -> None:
        path = Path(path)
        if size is None and path.is_file():
            size = path.stat().st_size
        self.outputs.append({"path": str(path), "kind": kind, "sha256": sha256, "size_bytes": size})

    def metrics_path(self, name: str) -> Path:
        return self.run.run_dir / "metrics" / name

    @property
    def redact(self) -> bool:
        return bool(self.config.get("logging.redact_prompts"))

    @contextmanager
    def generation_dir(self, name: str) -> Iterator[Path]:
        """Where a generation step writes its spec, raw output, prompt files and tool logs.

        With ``logging.redact_prompts`` these hold prompt and response text, so they go to a temporary
        directory outside the workspace that is deleted afterwards (docs/assessments.md#redaction).
        """
        if not self.redact:
            yield self.log_dir
            return
        private = Path(tempfile.mkdtemp(prefix=f"pa-redacted-{self.stage.id}-"))
        try:
            yield private
        finally:
            remove_tree(private)
            self.diagnostics.setdefault("generation_files_discarded", []).append(f"{name} (logging.redact_prompts)")


class Stage:
    id: str = "??"
    name: str = "base"
    title: str = ""
    config_keys: tuple[str, ...] = ()
    gpu: bool = False
    always_rerun: bool = False  # host-state checks (preflight) are never reused

    # -- planning (config-only so --dry-run can use them without a Run) --------
    def dependencies(self, config: ResolvedConfig) -> tuple[str, ...]:
        return ()

    def applicable(self, config: ResolvedConfig) -> tuple[bool, str]:
        return True, ""

    def extra_fingerprint(self, run: Run) -> dict[str, Any]:
        return {}

    def components_used(self, config: ResolvedConfig) -> set[str] | None:
        """Tool components this stage executes. When set, a dependency on stage 01 is keyed by these
        components' fingerprints only (adding an unrelated quantization format must not invalidate
        abliteration)."""
        return None

    def target_outputs(self, config: ResolvedConfig) -> list[str]:
        return []

    # -- execution ------------------------------------------------------------
    def check_prerequisites(self, ctx: StageContext) -> None:
        return None

    def execute(self, ctx: StageContext) -> None:
        raise NotImplementedError

    def validate_outputs(self, run: Run, receipt: dict[str, Any]) -> list[str]:
        """Return problems that make the receipt unusable (empty list = valid)."""
        problems = []
        for output in receipt.get("outputs", []):
            path = Path(output["path"])
            if not path.exists():
                problems.append(f"missing output {path}")
            elif output.get("size_bytes") is not None and path.is_file() and path.stat().st_size != output["size_bytes"]:
                problems.append(f"size changed for {path}")
        return problems

    # -- keys -------------------------------------------------------------------
    @property
    def receipt_name(self) -> str:
        return f"{self.id}-{self.name}.json"

    def stage_key(self, run: Run, dep_keys: dict[str, str | None]) -> tuple[str, str]:
        config_subset = run.config.subset(list(self.config_keys))
        stage_config_hash = sha256_obj(config_subset)
        key = sha256_obj({"stage": self.id, "config": stage_config_hash, "extra": self.extra_fingerprint(run), "deps": dep_keys})
        return key, stage_config_hash


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    receipt.setdefault("schema_version", RECEIPT_SCHEMA_VERSION)
    atomic_write_json(path, receipt)


def load_receipt(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("stage_id") else None


def base_receipt(ctx: StageContext, stage_key: str, stage_config_hash: str, dep_keys: dict[str, Any], started: str) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "stage_id": ctx.stage.id,
        "stage_name": ctx.stage.name,
        "run_id": ctx.run.run_id,
        "started_utc": started,
        "ended_utc": utc_now(),
        "config_hash": ctx.run.config.config_hash,
        "stage_config_hash": stage_config_hash,
        "stage_key": stage_key,
        "dependency_receipts": dep_keys,
        "input_artifact_ids": ctx.input_artifacts,
        "output_artifact_ids": ctx.output_artifacts,
        "tool_revisions": ctx.tool_revisions,
        "outputs": ctx.outputs,
        "warnings": ctx.warnings,
        "diagnostics": ctx.diagnostics,
        "partial": ctx.partial,
    }


def require(condition: bool, error_id: str, message: str, code: ExitCode = ExitCode.DEPENDENCY, hint: str | None = None) -> None:
    if not condition:
        raise PipelineError(code, error_id, message, hint=hint)
