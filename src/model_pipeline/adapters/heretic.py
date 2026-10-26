"""Adapter for Heretic (https://github.com/p-e-w/heretic), pinned to a verified commit.

Verified facts about the pinned engine (see docs/heretic-adapter.md):
* Configuration is read from ``config.toml`` in the *working directory* (plus CLI/env); we
  generate that file per run and execute Heretic with ``cwd`` set to a run-private directory.
* Headless operation requires ``export_strategy``, ``checkpoint_action``, ``trial_index``,
  ``model_action`` and ``save_directory`` (upstream e2e tests use the same keys).
* ``trial_index`` indexes the Pareto front sorted by objective values (fewest refusals first).
* On SIGINT Heretic stops the Optuna study and proceeds to export the configured trial when at
  least one trial completed; that is how the wall-clock budget is enforced.
* Heretic exits with status 0 even after configuration errors, so success is decided by
  validating the saved checkpoint, never by the exit code.
* ``--reproduce`` accepts version-4 reproduce.json files; settings in that file (including
  ``model_action``/``trial_index``) override config.toml, hence the control-flow patching below.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import configspec
from ..errors import ExitCode, PipelineError
from ..fsutil import atomic_write_json, atomic_write_text, read_json, sha256_file, sha256_obj
from ..telemetry import utc_now
from . import toml_io

ARA_PLUGIN = "heretic.modifiers.ara.ARA"
ABLITERATION_PLUGIN = "heretic.modifiers.abliteration.Abliteration"
# Keys the orchestrator owns; templates must not set them.
CONTROL_KEYS = {
    "model",
    "model_commit",
    "export_strategy",
    "checkpoint_action",
    "trial_index",
    "n_additional_trials",
    "model_action",
    "save_directory",
    "n_trials",
    "seed",
    "study_checkpoint_dir",
    "upload_repo_id",
    "upload_repo_private",
    "upload_reproducibility_information",
    "evaluate_model",
    "collect_reproducibles",
    "reproduce",
    "ignore_mismatches",
}
PLUGIN_TABLES = {"scorer", "modifier"}
STUDY_EXCLUDED = {"trial_index", "n_trials", "save_directory", "checkpoint_action"}

RE_TRIAL = re.compile(r"Running trial\s+(\d+)\s+of\s+(\d+)")
RE_RESTORE = re.compile(r"Restoring model from trial\s+(\d+)")
RE_SAVED = re.compile(r"Model saved to\s+(.+?)\.?\s*$")
RE_HASH = re.compile(r"^(.+?):\s+(Hash matches|Hash doesn't match|File not found)")


@dataclass
class EngineOutcome:
    trials_started: int = 0
    trials_planned: int | None = None
    restored_trial: int | None = None
    saved_to: str | None = None
    config_error: bool = False
    resumed: bool = False
    shutdown_message: bool = False
    hash_checks: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def on_line(self, _label: str, line: str) -> None:
        text = line.strip()
        if m := RE_TRIAL.search(text):
            self.trials_started = max(self.trials_started, int(m.group(1)))
            self.trials_planned = int(m.group(2))
        elif m := RE_RESTORE.search(text):
            self.restored_trial = int(m.group(1))
        elif m := RE_SAVED.search(text):
            self.saved_to = m.group(1).strip()
        elif "Configuration contains" in text:
            self.config_error = True
        elif "Resuming existing study" in text:
            self.resumed = True
        elif "Shutting down" in text:
            self.shutdown_message = True
        elif m := RE_HASH.search(text):
            self.hash_checks[m.group(1).strip()] = m.group(2)
        elif text.startswith("Error") or "[red]Error" in text:
            self.errors.append(text[:500])

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def native_template(path: Path) -> dict[str, Any]:
    data = toml_io.load(path)
    owned = sorted(set(data) & CONTROL_KEYS)
    if owned:
        raise PipelineError(
            ExitCode.CONFIG,
            "E_HERETIC_TEMPLATE_CONTROL_KEY",
            f"Heretic template {path} sets keys owned by the orchestrator: {', '.join(owned)}",
            hint="Use the pipeline YAML (abliteration.*) for trials, seed, budget, model and export settings.",
        )
    return data


def check_settings(native: dict[str, Any], settings_fields: list[str]) -> list[str]:
    """Top-level keys unknown to the installed Heretic Settings model (plugin tables excepted)."""
    known = set(settings_fields) | PLUGIN_TABLES
    return sorted(k for k in native if k not in known)


def modifier_plugins(native: dict[str, Any]) -> list[str]:
    modifiers = native.get("modifiers")
    if modifiers is None:
        return [ARA_PLUGIN]  # upstream default at the pinned commit
    return [m["plugin"] for m in modifiers]


def dataset_specs(native: dict[str, Any]) -> list[dict[str, Any]]:
    """All dataset specifications in a native config (to prefetch and to record provenance)."""
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "dataset" in node and isinstance(node["dataset"], str):
                found.append({k: node.get(k) for k in ("dataset", "commit", "config", "split", "column")})
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(native)
    unique = {json.dumps(d, sort_keys=True): d for d in found}
    return list(unique.values())


def derived_seed(run_id: str) -> int:
    return int(sha256_obj(run_id)[:8], 16)


def build_optimize_config(
    template: dict[str, Any],
    *,
    model_path: str,
    n_trials: int,
    n_startup_trials: int | None,
    seed: int,
    trial_index: int,
    save_directory: str,
    checkpoint_action: str,
) -> dict[str, Any]:
    native = copy.deepcopy(template)
    startup = n_startup_trials if n_startup_trials is not None else native.get("n_startup_trials", 30)
    startup = min(int(startup), max(1, n_trials // 3)) if n_startup_trials is None else int(startup)
    native.update(
        model=model_path,
        n_trials=int(n_trials),
        n_startup_trials=startup,
        seed=int(seed),
        trial_index=int(trial_index),
        model_action="save",
        export_strategy="merge",
        save_directory=save_directory,
        checkpoint_action=checkpoint_action,
        study_checkpoint_dir="checkpoints",
    )
    return native


def study_hash(native: dict[str, Any], engine_commit: str) -> str:
    relevant = {k: v for k, v in native.items() if k not in STUDY_EXCLUDED}
    return sha256_obj({"native": relevant, "engine": engine_commit})[:16]


def patch_reproduce_file(original: dict[str, Any], model_path: str) -> tuple[dict[str, Any], list[str]]:
    """Patch only control-flow settings of a real reproduce.json; algorithmic settings stay untouched."""
    if str(original.get("version")) != "4":
        raise PipelineError(
            ExitCode.UNSUPPORTED,
            "E_REPRODUCE_VERSION",
            f"reproduce.json version {original.get('version')!r} is not readable by the pinned Heretic (needs version 4)",
            hint="Older files require Heretic 1.4, which has no headless mode; exact reproduction is blocked.",
        )
    data = copy.deepcopy(original)
    settings = data.setdefault("settings", {})
    changes = []
    for key, value in (("model", model_path), ("model_action", "save"), ("trial_index", 0)):
        if settings.get(key) != value:
            changes.append(f"settings.{key}: {settings.get(key)!r} -> {value!r}")
            settings[key] = value
    for key in ("upload_repo_id", "upload_repo_private", "upload_reproducibility_information"):
        if settings.get(key) is not None:
            changes.append(f"settings.{key} cleared")
            settings[key] = None
    if not settings.get("export_strategy"):
        settings["export_strategy"] = "merge"
        changes.append("settings.export_strategy: None -> 'merge' (original value unknown)")
    return data, changes


def synthesize_reproduce_file(published: dict[str, Any], model_path: str, seed: int, dataset_commits: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    """Approximate reproduction input built from published parameters (clearly labelled)."""
    params = published.get("parameters")
    if not isinstance(params, dict) or not params:
        raise PipelineError(
            ExitCode.UNSUPPORTED,
            "E_REPRODUCE_NO_PARAMETERS",
            "Approximate reproduction requested but the baseline file has no published_method.parameters",
        )
    plugin = published.get("modifier_plugin", ARA_PLUGIN)
    class_name = plugin.rsplit(".", 1)[-1]
    modifier_settings = copy.deepcopy(published.get("modifier_settings") or {})
    assumptions = [
        "Engine environment of the reference is unknown; Heretic environment checks are bypassed (ignore_mismatches=true).",
        "Datasets, dataset slices and seeds used by the reference are not published; pinned Heretic defaults are assumed.",
    ]
    for key in ("good_prompts", "bad_prompts"):
        if key not in modifier_settings:
            dataset = configspec.HARMLESS_DATASET if key == "good_prompts" else configspec.HARMFUL_DATASET
            modifier_settings[key] = {
                "dataset": dataset,
                "commit": dataset_commits.get(dataset),
                "split": "train[:400]",
                "column": "text",
            }
    unpublished = [k for k in ("lora_rank", "n_optimization_steps", "learning_rate", "max_iter", "history_size") if k not in modifier_settings]
    if unpublished:
        assumptions.append(f"Unpublished modifier settings use pinned defaults: {', '.join(unpublished)}.")
    data = {
        "version": "4",
        "timestamp": utc_now()[:19],
        "environment": {
            "heretic": {"version": str(published.get("claimed_engine_version", "unknown")), "is_standard_pypi": False, "metadata": {}},
            "pytorch_version": "unknown",
            "requirements": {},
        },
        "settings": {
            "model": model_path,
            "seed": seed,
            "modifiers": [{"plugin": plugin}],
            "modifier": {class_name: modifier_settings},
            "model_action": "save",
            "export_strategy": "merge",
            "trial_index": 0,
        },
        "parameters": params,
        "scores": [],
        "hashes": {},
        "_project_a": {"synthesized": True, "label": "approximate_reproduction", "source": published.get("source_url")},
    }
    return data, assumptions


def candidate_ok(path: Path) -> list[str]:
    problems = []
    if not path.is_dir():
        return [f"{path} does not exist"]
    if not (path / "config.json").is_file():
        problems.append("config.json missing")
    if not list(path.glob("*.safetensors")):
        problems.append("no *.safetensors written")
    index = path / "model.safetensors.index.json"
    if index.exists():
        try:
            shards = set(read_json(index)["weight_map"].values())
        except (ValueError, KeyError):
            problems.append("unreadable safetensors index")
        else:
            missing = [s for s in shards if not (path / s).is_file()]
            if missing:
                problems.append(f"index references missing shards: {missing[:3]}")
    return problems


MODEL_LINK_NAME = "model"


def link_model(workdir: Path, target: Path) -> dict[str, str]:
    """Create ``workdir/model`` -> ``target`` (symlink; NTFS junction fallback on Windows)."""
    link = workdir / MODEL_LINK_NAME
    target = Path(target).resolve()
    if os.path.lexists(link):
        if link.resolve() == target:
            return {"link": str(link), "target": str(target), "kind": "existing"}
        if link.is_symlink():
            os.unlink(link)
        elif getattr(os.path, "isjunction", lambda _p: False)(link):
            os.rmdir(link)  # removes the junction only, never the target's contents
        else:
            raise PipelineError(ExitCode.ENGINE, "E_MODEL_LINK", f"{link} exists and is not a link")
    try:
        os.symlink(target, link, target_is_directory=True)
        kind = "symlink"
    except OSError:
        if os.name != "nt":
            raise
        import _winapi  # noqa: PLC0415 - Windows-only stdlib module

        _winapi.CreateJunction(str(target), str(link))
        kind = "junction"
    return {"link": str(link), "target": str(target), "kind": kind}


def write_engine_files(workdir: Path, native: dict[str, Any], reproduce: dict[str, Any] | None) -> dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    config_path = workdir / "config.toml"
    atomic_write_text(config_path, "# Generated by Project A; do not edit (see engine-plan.json)\n" + toml_io.dumps(native))
    toml_io.load(config_path)  # round-trip syntax check
    info = {"config_toml": str(config_path), "config_sha256": sha256_file(config_path), "reproduce_json": None, "reproduce_sha256": None}
    if reproduce is not None:
        rpath = workdir / "reproduce.json"
        atomic_write_json(rpath, reproduce)
        info.update(reproduce_json=str(rpath), reproduce_sha256=sha256_file(rpath))
    return info


def engine_env(base: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    env.update({"COLUMNS": "400", "NO_COLOR": "1", "TERM": "dumb", "PYTHONIOENCODING": "utf-8"})
    return env


def summarize_study(export: dict[str, Any], restored_trial: int | None, refusal_name: str, kl_name: str) -> dict[str, Any]:
    trials = export.get("trials", [])
    complete = [t for t in trials if t.get("state") == "COMPLETE"]
    exported = None
    if restored_trial is not None:
        exported = next((t for t in trials if t.get("index") == restored_trial), None)
    if exported is None and export.get("pareto"):
        idx = export.get("requested_trial_index", 0)
        pareto = export["pareto"]
        exported = pareto[idx] if idx < len(pareto) else None

    def score(trial: dict[str, Any] | None, name: str) -> float | None:
        if not trial:
            return None
        for rec in trial.get("scores") or []:
            if rec.get("name") == name:
                return rec.get("score", {}).get("value")
        return None

    return {
        "trials_total": len(trials),
        "trials_complete": len(complete),
        "finished": export.get("finished"),
        "pareto_size": len(export.get("pareto", [])),
        "exported_trial_index": exported.get("index") if exported else None,
        "exported_parameters": exported.get("parameters") if exported else None,
        "optimizer_scores": exported.get("scores") if exported else None,
        "optimizer_refusal_rate": score(exported, refusal_name),
        "optimizer_kl": score(exported, kl_name),
        "provenance": "heretic_optimizer (in-loop scores on Heretic's evaluation prompts; not held-out validation)",
    }
