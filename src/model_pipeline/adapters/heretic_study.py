"""Physical Heretic studies: identity, directory, lock and append-only history (docs/study-lifecycle.md).

A study lives in ``runs/<run>/checkpoints/heretic/study-<study_id>/``:
``study.json`` (identity record), ``checkpoints/*.jsonl`` (Optuna journal), ``study-history.jsonl``,
``.lock`` and Heretic's working files. Only the run that owns the directory writes to it; a fork
copies the journal into its own study directory.
"""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from ..evaluation.selection import trial_set_sha256
from ..fsutil import FileLock, atomic_write_json, read_json, sha256_obj
from ..telemetry import utc_now

HISTORY_SCHEMA_VERSION = "1.0"
# Control and budget keys; everything else in the native settings decides which study is used.
IDENTITY_EXCLUDED = {"trial_index", "n_trials", "save_directory", "checkpoint_action", "model_action"}
SEARCH_ACTIONS = ("new", "continue", "replay")
END_PHASES = ("completion", "interruption", "failure", "reconciled")


def environment_fingerprint(record) -> dict[str, Any]:
    """Tool environment of the Heretic component: torch (version carries the CUDA build), transformers, optuna, python."""
    versions = record.versions or {}
    torch_spec = (record.spec or {}).get("torch") or {}
    return {
        "torch": versions.get("torch"),
        "transformers": versions.get("transformers"),
        "optuna": versions.get("optuna"),
        "python": versions.get("python"),
        "torch_source": torch_spec.get("source"),
        "torch_backend": torch_spec.get("backend"),
    }


def study_identity(native: dict[str, Any], engine_commit: str, environment: dict[str, Any], budget_derived: tuple[str, ...] = ()) -> dict[str, Any]:
    excluded = IDENTITY_EXCLUDED | set(budget_derived)
    inputs = {"native": {k: v for k, v in native.items() if k not in excluded}, "engine": engine_commit}
    inputs_id = sha256_obj(inputs)[:16]
    return {
        "study_id": sha256_obj({"inputs_id": inputs_id, "environment": environment})[:16],
        "inputs_id": inputs_id,
        "engine": engine_commit,
        "environment": environment,
    }


def study_state(export: dict[str, Any] | None) -> dict[str, Any]:
    """Observable state of a study (from the study tool's ``export``); used as before/after in the history."""
    if export is None:
        return {"trials": 0, "states": {}, "trial_set_sha256": trial_set_sha256([]), "stored_n_trials": None, "finished": None, "settings_sha256": None}
    trials = export.get("trials") or []
    return {
        "trials": len(trials),
        "states": dict(Counter(t.get("state") for t in trials)),
        "trial_set_sha256": trial_set_sha256(trials),
        "stored_n_trials": export.get("n_trials_setting"),
        "finished": export.get("finished"),
        "settings_sha256": export.get("settings_sha256"),
        "running": [t.get("number") for t in trials if t.get("state") == "RUNNING"],
    }


class StudyDir:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.name = self.path.name
        self.journal_dir = self.path / "checkpoints"
        self.record_path = self.path / "study.json"
        self.history = History(self.path / "study-history.jsonl")

    def lock(self) -> FileLock:
        return FileLock(self.path / ".lock", f"Heretic study {self.name}", error_id="E_STUDY_LOCKED")

    def journal(self) -> Path | None:
        journals = sorted(self.journal_dir.glob("*.jsonl")) if self.journal_dir.is_dir() else []
        return journals[0] if journals else None

    def record(self) -> dict[str, Any] | None:
        return read_json(self.record_path) if self.record_path.is_file() else None

    def write_record(self, identity: dict[str, Any], **extra: Any) -> None:
        atomic_write_json(self.record_path, {**identity, **extra, "created_utc": utc_now()})


def studies_with_inputs(root: Path, inputs_id: str, exclude: Path) -> list[StudyDir]:
    """Recorded studies under ``root`` (other than ``exclude``) with the same inputs, whatever their environment."""
    found = []
    for path in sorted(root.glob("study-*")) if root.is_dir() else []:
        study = StudyDir(path)
        record = study.record()
        if path.resolve() != Path(exclude).resolve() and record and record.get("inputs_id") == inputs_id:
            found.append(study)
    return found


class History:
    """Append-only ``study-history.jsonl``; callers hold the study lock."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, phase: str, action: str, invocation_id: str, run_id: str, **fields: Any) -> dict[str, Any]:
        entry = {
            "schema_version": HISTORY_SCHEMA_VERSION,
            "seq": len(self.entries()) + 1,
            "utc": utc_now(),
            "run_id": run_id,
            "invocation_id": invocation_id,
            "phase": phase,
            "action": action,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return entry

    def intent(self, action: str, run_id: str, **fields: Any) -> str:
        invocation_id = f"inv-{uuid.uuid4().hex[:12]}"
        self.append("intent", action, invocation_id, run_id, **fields)
        return invocation_id

    def open_intents(self) -> list[dict[str, Any]]:
        entries = self.entries()
        ended = {e["invocation_id"] for e in entries if e["phase"] in END_PHASES}
        return [e for e in entries if e["phase"] == "intent" and e["invocation_id"] not in ended]

    def finished_searches(self) -> set[str]:
        """Search fingerprints whose search ended normally (including budget stops), here or in a fork's parent."""
        done: set[str] = set()
        for e in self.entries():
            if e["phase"] == "completion" and e["action"] in SEARCH_ACTIONS and e.get("search_fingerprint"):
                done.add(e["search_fingerprint"])
            elif e["phase"] == "completion" and e["action"] == "fork-continue":
                done.update(e.get("parent_finished_searches") or [])
        return done

    def last_completion(self, action: str) -> dict[str, Any] | None:
        return next((e for e in reversed(self.entries()) if e["phase"] == "completion" and e["action"] == action), None)
