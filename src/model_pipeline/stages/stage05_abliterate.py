"""Stage 05 - abliterate: search, select and export a Heretic candidate (docs/study-lifecycle.md).

Optimize runs have four internal phases, all under the study lock (taken after the run lock):

1. locate - the study directory is keyed by the study identity (search settings without budget,
   engine commit, tool environment). A fork copies its parent's journal; a study recorded under
   another environment is refused.
2. search - Heretic with ``model_action = ""`` (search only) until the budget is recorded.
3. select - constraint-based choice among completed trials (``evaluation.selection``).
4. export - export-only: stored ``n_trials = len(trials)``, ``trial_index`` = the chosen trial's
   Pareto position, ``model_action = "save"``; then verify zero added trials and the restored trial.

Each phase that modifies the study writes an intent to ``study-history.jsonl`` first and an end
entry afterwards. Reproduction runs have no study and keep their single ``--reproduce`` call.
Output is written to ``<candidate>.partial`` and renamed only after validation.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from ..adapters import heretic as H
from ..adapters import heretic_study as HS
from ..artifacts.safetensors_meta import dtype_summary, tensor_diff, tensor_map
from ..errors import ExitCode, Interrupted, PipelineError
from ..evaluation.selection import select_trial
from ..fsutil import atomic_write_json, disk_free_bytes, human_bytes, read_json, remove_tree, sha256_file, sha256_obj
from .base import Stage, StageContext

# Inputs of the selection phase. They key stage 05 for optimize runs only, so changing them in a
# fork re-runs selection and export (zero new trials) without touching reproduction runs.
SELECTION_KEYS = (
    "abliteration.trial_id",
    "abliteration.pareto_trial_index",
    "selection.policy",
    "selection.max_optimizer_refusal_rate",
    "selection.max_optimizer_kl",
    "selection.optimizer_refusal_score",
    "selection.optimizer_kl_score",
    "selection.allow_ineligible_choice",
)


class AbliterateStage(Stage):
    id = "05"
    name = "abliterate"
    title = "Run Heretic"
    gpu = True
    config_keys = ("abliteration.experiment", "abliteration.save_intermediate")

    def dependencies(self, config) -> tuple[str, ...]:
        return ("04",)

    def applicable(self, config) -> tuple[bool, str]:
        return (True, "") if config.get("abliteration.enabled") else (False, "abliteration disabled (quantize-only)")

    def extra_fingerprint(self, run) -> dict[str, Any]:
        if run.config.get("abliteration.experiment") != "optimize":
            return {}
        return {"selection": run.config.subset(list(SELECTION_KEYS))}

    def target_outputs(self, config) -> list[str]:
        return [
            "workspace/models/candidates/<run>/<candidate>/ (Heretic merged export)",
            "runs/<run>/metrics/candidate.json",
            "runs/<run>/metrics/selection-candidates.json (optimize)",
            "runs/<run>/checkpoints/heretic/study-<id>/study-history.jsonl (optimize)",
        ]

    # -- helpers ------------------------------------------------------------------
    def _study_tool(self, ctx: StageContext, record, args: list[str], name: str) -> dict[str, Any]:
        out = ctx.log_dir / f"{name}.json"
        result = ctx.run.runner.run(
            [*record.script("heretic_study_tool.py"), *args, "--out", str(out)],
            cwd=None,
            env=ctx.run.tool_env(record),
            log_dir=ctx.log_dir,
            name=name,
            timeout_seconds=900,
        )
        if not result.ok or not out.exists():
            raise PipelineError(
                ExitCode.ENGINE, "E_HERETIC_STUDY_TOOL", f"Study helper '{name}' failed", hint=f"{result.stderr_log}: {' | '.join(result.tail[-4:])}"
            )
        return read_json(out)

    def _read_study(self, ctx: StageContext, record, study: HS.StudyDir, name: str) -> dict[str, Any] | None:
        journal = study.journal()
        return self._study_tool(ctx, record, ["export", "--journal", str(journal)], name) if journal else None

    def _run_heretic(self, ctx: StageContext, record, study: HS.StudyDir, native: dict[str, Any], name: str, **limits: Any):
        run = ctx.run
        native = dict(native)
        # Heretic names its study journal after settings.model; a long absolute snapshot path can exceed
        # filename limits (found with the real engine). Point it at a short link in its private cwd.
        link = H.link_model(study.path, Path(native["model"]))
        native["model"] = H.MODEL_LINK_NAME
        files = H.write_engine_files(study.path, native, None)
        outcome = H.EngineOutcome()
        result = run.runner.run(
            [*record.cmd("heretic")],
            cwd=study.path,
            env=H.engine_env(run.tool_env(record)),
            log_dir=ctx.log_dir,
            name=name,
            on_line=outcome.on_line,
            **limits,
        )
        return result, outcome, {**files, "model_link": link, "result": result.to_dict(), "outcome": outcome.to_dict()}

    # -- execute ------------------------------------------------------------------
    def execute(self, ctx: StageContext) -> None:
        run = ctx.run
        prep = run.load_receipt("04") or {}
        plan = read_json(prep["diagnostics"]["plan"])
        record = run.components(ctx.log_dir).get("heretic")
        commit = plan["heretic_commit"] or "unknown"
        candidates_root = run.workspace / "models" / "candidates" / run.run_id
        if plan["label"] == "optimize":
            final, engine, summary, extra = self._optimize(ctx, plan, record, commit, candidates_root)
        else:
            final, engine, summary, extra = self._reproduce(ctx, plan, record, commit, candidates_root)
        source_tensors = tensor_map(run.registry.get("source")["manifest_data"]["artifact"]["uri"])
        cand_tensors = tensor_map(final)
        diff = tensor_diff(source_tensors, cand_tensors)
        if diff["missing_count"]:
            ctx.warn(
                f"Candidate lacks {diff['missing_count']} source tensors (prefixes {diff['missing_prefixes']}); "
                "see selection.graft_missing_source_tensors to restore them verbatim"
            )
        candidate = {
            "candidate_id": final.name,
            "path": str(final),
            "label": plan["label"],
            "modifiers": plan.get("modifiers", []),
            "study_hash": extra.get("study_id"),
            "seed": plan["seed"],
            "engine": engine,
            "summary": summary,
            "selection": extra.get("selection"),
            "dtype_summary": dtype_summary(cand_tensors),
            "tensor_diff_vs_source": {**diff, "missing_from_target": diff["missing_from_target"][:200]},
            "limitations": plan.get("limitations", []),
            "source_artifact_id": plan["source_artifact_id"],
        }
        cand_path = ctx.metrics_path("candidate.json")
        atomic_write_json(cand_path, candidate)
        ctx.add_output(cand_path, "candidate_metadata")
        for shard in sorted(final.iterdir()):
            if shard.is_file():
                ctx.add_output(shard, "candidate_file")
        ctx.input_artifacts.append(plan["source_artifact_id"])
        ctx.tool_revisions["heretic"] = commit
        ctx.diagnostics.update(
            candidate=str(cand_path),
            candidate_path=str(final),
            label=plan["label"],
            trials_started=engine.get("search", {}).get("outcome", {}).get("trials_started", engine.get("outcome", {}).get("trials_started", 0)),
            budget_stopped=engine.get("search", {}).get("result", {}).get("budget_stopped", engine.get("result", {}).get("budget_stopped")),
        )

    # -- reproduction (no study) ----------------------------------------------------
    def _reproduce(self, ctx: StageContext, plan, record, commit: str, candidates_root: Path):
        run, config = ctx.run, ctx.config
        label = plan["label"]
        workdir = run.run_dir / "checkpoints" / "heretic" / label
        final = candidates_root / f"c-{label}"
        engine: dict[str, Any] = {"workdir": str(workdir), "label": label, "commit": commit}
        if self._reusable(ctx, final):
            engine["reused_candidate"] = True
        else:
            partial = self._prepare_target(ctx, final, candidates_root)
            native = dict(plan["native"])
            native["save_directory"] = str(partial)
            reproduce_doc = read_json(plan["reproduce_json"])
            reproduce_doc["settings"]["model"] = H.MODEL_LINK_NAME
            workdir.mkdir(parents=True, exist_ok=True)
            engine["model_link"] = H.link_model(workdir, Path(native["model"]))
            native["model"] = H.MODEL_LINK_NAME
            files = H.write_engine_files(workdir, native, reproduce_doc)
            engine.update(files)
            budget_min = config.get("abliteration.max_runtime_minutes")
            grace = float(config.get("abliteration.shutdown_grace_minutes")) * 60
            outcome = H.EngineOutcome()
            ctx.info(f"Running Heretic ({label}) in {workdir}")
            result = run.runner.run(
                [*record.cmd("heretic"), "--reproduce", files["reproduce_json"]],
                cwd=workdir,
                env=H.engine_env(run.tool_env(record)),
                log_dir=ctx.log_dir,
                name="heretic",
                grace_seconds=grace,
                timeout_seconds=(float(budget_min) * 60 + grace) if budget_min else None,
                on_line=outcome.on_line,
            )
            engine.update(result=result.to_dict(), outcome=outcome.to_dict())
            if result.interrupted:
                remove_tree(partial)
                raise Interrupted("Heretic interrupted during reproduction; it is re-run on resume")
            problems = H.candidate_ok(partial)
            if problems:
                remove_tree(partial)
                ctx.diagnostics["engine"] = engine
                raise self._diagnose(outcome, result, problems, label)
            os.replace(partial, final)
        doc = read_json(plan["reproduce_json"])
        summary: dict[str, Any] = {
            "trials_total": 0,
            "exported_parameters": doc.get("parameters"),
            "optimizer_scores": doc.get("scores") or None,
            "optimizer_refusal_rate": None,
            "optimizer_kl": None,
            "provenance": "reproduction: no optimization ran; parameters come from the reproduce input",
        }
        if doc.get("hashes"):
            ours = {p.name: sha256_file(p) for p in final.glob("*.safetensors")}
            matches = {name: ours.get(name) == digest.lower() for name, digest in doc["hashes"].items()}
            summary["bitwise_identical"] = bool(matches) and all(matches.values())
            summary["hash_matches"] = matches
            if not summary["bitwise_identical"]:
                ctx.warn("Reproduced weights are NOT bitwise identical to the reference hashes")
        return final, engine, summary, {}

    def _reusable(self, ctx: StageContext, final: Path) -> bool:
        if final.is_dir() and not H.candidate_ok(final):
            ctx.info(f"Validated candidate {final} already exists from an earlier attempt; not re-running Heretic")
            return True
        return False

    def _prepare_target(self, ctx: StageContext, final: Path, candidates_root: Path) -> Path:
        run = ctx.run
        partial = final.with_name(final.name + ".partial")
        if final.exists():
            remove_tree(final)
        if partial.exists():
            ctx.warn(f"Removing incomplete export from an interrupted attempt: {partial}")
            remove_tree(partial)
        source_size = sum(f["size_bytes"] for f in run.registry.get("source")["manifest_data"]["artifact"]["files"])
        free = disk_free_bytes(candidates_root.parent if candidates_root.parent.exists() else run.workspace)
        if free < 1.05 * source_size:
            raise PipelineError(ExitCode.ENGINE, "E_DISK_INSUFFICIENT", f"Candidate export needs ~{human_bytes(source_size)}; {human_bytes(free)} free")
        candidates_root.mkdir(parents=True, exist_ok=True)
        return partial

    # -- optimize: locate, search, select, export --------------------------------------
    def _optimize(self, ctx: StageContext, plan, record, commit: str, candidates_root: Path):
        run, config = ctx.run, ctx.config
        native = dict(plan["native"])
        identity = HS.study_identity(native, commit, HS.environment_fingerprint(record), tuple(plan.get("budget_derived_keys", ())))
        heretic_root = run.run_dir / "checkpoints" / "heretic"
        study = HS.StudyDir(heretic_root / f"study-{identity['study_id']}")
        search_fp = sha256_obj(
            {"study_id": identity["study_id"], "n_trials": native["n_trials"], "max_runtime_minutes": config.get("abliteration.max_runtime_minutes")}
        )
        fingerprints = {"study_identity": identity["study_id"], "search": search_fp, "selection": None, "export": None}
        engine: dict[str, Any] = {
            "workdir": str(study.path),
            "label": "optimize",
            "commit": commit,
            "study_id": identity["study_id"],
            "history": str(study.history.path),
            "phases": {},
        }
        ctx.diagnostics.update(fingerprints=fingerprints, study_dir=str(study.path), history=str(study.history.path), engine=engine)
        with study.lock():
            ctx.diagnostics["reconciled"] = self._reconcile(ctx, record, study)
            action = "continue" if study.journal() else self._locate(ctx, record, study, identity, heretic_root)
            export = self._search(ctx, record, study, native, search_fp, action, engine)
            selection = self._select(ctx, study, export, identity)
            fingerprints["selection"] = selection["selection_fingerprint"]
            final, export_fp, after = self._export(ctx, plan, record, study, native, export, selection, engine, candidates_root, commit)
            fingerprints["export"] = export_fp
        summary = H.summarize_study(
            after, selection["chosen"]["index"], config.get("selection.optimizer_refusal_score"), config.get("selection.optimizer_kl_score")
        )
        study_path = ctx.metrics_path("heretic_study.json")
        atomic_write_json(study_path, after)
        ctx.add_output(study_path, "heretic_study")
        chosen = selection["chosen"]
        extra = {
            "study_id": identity["study_id"],
            "selection": {
                "chosen": chosen,
                "origin": selection["origin"],
                "policy": selection["policy"],
                "constraints": selection["constraints"],
                "selection_fingerprint": selection["selection_fingerprint"],
                "export_fingerprint": export_fp,
                "candidates": str(ctx.metrics_path("selection-candidates.json")),
            },
        }
        ctx.info(f"Candidate {final.name}: trial {chosen['index']} (Pareto position {chosen['pareto_position']}, chosen by {chosen['by']})")
        return final, engine, summary, extra

    def _reconcile(self, ctx: StageContext, record, study: HS.StudyDir) -> list[dict[str, Any]]:
        """Close intents left open by a process that stopped without recording an outcome."""
        intents = study.history.open_intents()
        if not intents:
            return []
        state = HS.study_state(self._read_study(ctx, record, study, "study-reconcile"))
        out = []
        for intent in intents:
            added = state["trials"] - int((intent.get("before") or {}).get("trials") or 0)
            entry = study.history.append(
                "reconciled",
                intent["action"],
                intent["invocation_id"],
                ctx.run.run_id,
                observed=state,
                trials_added=added,
                orphaned_running=state["running"],
                note="the invoking process stopped without recording an outcome",
            )
            out.append(entry)
            ctx.warn(
                f"Study {study.name}: '{intent['action']}' invocation {intent['invocation_id']} (run {intent['run_id']}) never recorded an "
                f"outcome; the journal now has {state['trials']} trials ({added:+d} since its intent)"
                + (f"; orphaned RUNNING trials {state['running']} count towards the budget" if state["running"] else "")
            )
        return out

    def _locate(self, ctx: StageContext, record, study: HS.StudyDir, identity: dict[str, Any], heretic_root: Path) -> str:
        """No journal yet: copy the parent's (fork), refuse an environment change, or start a new study."""
        run = ctx.run
        lineage = [heretic_root]
        if run.parent_run_id:
            parent_root = run.workspace / "runs" / run.parent_run_id / "checkpoints" / "heretic"
            lineage.append(parent_root)
            parent = HS.StudyDir(parent_root / study.name)
            if parent.journal():
                self._fork_copy(ctx, record, study, identity, parent)
                return "continue"
        for root in lineage:
            for other in HS.studies_with_inputs(root, identity["inputs_id"], exclude=study.path):
                old = other.record()
                if old.get("study_id") == identity["study_id"]:
                    continue
                changed = {k: [old["environment"].get(k), v] for k, v in identity["environment"].items() if old["environment"].get(k) != v}
                raise PipelineError(
                    ExitCode.DEPENDENCY,
                    "E_STUDY_ENVIRONMENT_CHANGED",
                    f"Study {other.path} has the same inputs but was recorded under another tool environment ({changed}); "
                    "continuing it would mix trials scored with different numerics",
                    hint="Re-provision the environment recorded in its study.json, or start a new study with --new-run (see docs/study-lifecycle.md).",
                    details={"study": str(other.path), "environment_changes": changed},
                )
        replays = [
            str(other.path)
            for root in sorted(run.workspace.glob("runs/*/checkpoints/heretic"))
            if root != heretic_root
            for other in HS.studies_with_inputs(root, identity["inputs_id"], exclude=study.path)
        ]
        study.write_record(identity, created_by_run=run.run_id, replay_of=replays or None)
        return "replay" if replays else "new"

    def _fork_copy(self, ctx: StageContext, record, study: HS.StudyDir, identity: dict[str, Any], parent: HS.StudyDir) -> None:
        """A fork owns an independent copy of the parent journal; the parent is only read (under its lock)."""
        run = ctx.run
        parent_info = {"run_id": run.parent_run_id, "study_dir": str(parent.path), "study_id": identity["study_id"]}
        with parent.lock():
            invocation = study.history.intent("fork-continue", run.run_id, parent=parent_info, before=HS.study_state(None))
            state = HS.study_state(self._read_study(ctx, record, parent, "parent-study-export"))
            source = parent.journal()
            study.journal_dir.mkdir(parents=True, exist_ok=True)
            partial = study.journal_dir / (source.name + ".partial")
            shutil.copy2(source, partial)
            os.replace(partial, study.journal_dir / source.name)
            finished = sorted(parent.history.finished_searches())
        study.write_record(identity, created_by_run=run.run_id, forked_from=parent_info)
        study.history.append(
            "completion",
            "fork-continue",
            invocation,
            run.run_id,
            parent=parent_info,
            trials_at_copy=state["trials"],
            after=state,
            parent_finished_searches=finished,
        )
        ctx.info(f"Continuing a copy of the Heretic study of parent run {run.parent_run_id} ({state['trials']} trials)")
        ctx.diagnostics["study_copied_from_run"] = run.parent_run_id

    def _search(self, ctx: StageContext, record, study: HS.StudyDir, native, search_fp: str, action: str, engine) -> dict[str, Any]:
        run, config = ctx.run, ctx.config
        current = self._read_study(ctx, record, study, "study-before-search")
        state = HS.study_state(current)
        n_trials = int(native["n_trials"])
        reason = None
        if current is not None and state["trials"] >= n_trials:
            reason = f"budget reached: {state['trials']} recorded trials >= max_trials {n_trials}"
        elif current is not None and search_fp in study.history.finished_searches():
            reason = "a search with this budget already finished (budget stop); a larger budget continues it"
        if reason:
            engine["phases"]["search"] = {"ran": False, "reason": reason}
            ctx.info(f"Search skipped: {reason}")
            return current
        budget_min = config.get("abliteration.max_runtime_minutes")
        grace = float(config.get("abliteration.shutdown_grace_minutes")) * 60
        planned = {
            "n_trials": n_trials,
            "max_runtime_minutes": budget_min,
            "model_action": "",
            "trial_index": 0,
            "checkpoint_action": "continue" if current is not None else "restart",
        }
        invocation = study.history.intent(action, run.run_id, search_fingerprint=search_fp, planned=planned, before=state)
        if current is not None:
            self._study_tool(
                ctx,
                record,
                ["set-settings", "--journal", str(study.journal()), "--n-trials", str(n_trials), "--trial-index", "0", "--model-action", "none"],
                "study-set-search",
            )
        run_native = {**native, "checkpoint_action": planned["checkpoint_action"], "model_action": "", "trial_index": 0}
        run_native["save_directory"] = str(study.path / "unused-save-directory")
        ctx.info(f"Searching ({action}) in {study.path}: up to {n_trials} trials" + (f" within {budget_min} min" if budget_min else ""))
        result, outcome, info = self._run_heretic(
            ctx, record, study, run_native, "heretic", budget_seconds=float(budget_min) * 60 if budget_min else None, grace_seconds=grace
        )
        engine["search"] = info
        after = self._read_study(ctx, record, study, "study-after-search")
        astate = HS.study_state(after)
        added = astate["trials"] - state["trials"]
        if result.interrupted:
            study.history.append("interruption", action, invocation, run.run_id, after=astate, trials_added=added)
            raise Interrupted("Heretic search interrupted; the study journal is kept and continued on resume")
        complete = astate["states"].get("COMPLETE", 0)
        if outcome.config_error or result.timed_out or result.killed or complete == 0 or (result.returncode != 0 and not result.budget_stopped):
            error = self._diagnose_search(outcome, result, complete)
            study.history.append("failure", action, invocation, run.run_id, after=astate, trials_added=added, error_id=error.error_id)
            raise error
        study.history.append(
            "completion",
            action,
            invocation,
            run.run_id,
            search_fingerprint=search_fp,
            after=astate,
            trials_added=added,
            budget_stopped=result.budget_stopped,
        )
        engine["phases"]["search"] = {"ran": True, "action": action, "trials_added": added, "budget_stopped": result.budget_stopped}
        if result.budget_stopped:
            ctx.warn(f"Budget of {budget_min} min reached: {astate['trials']} of {n_trials} trials recorded; selecting among the completed ones")
        return after

    def _select(self, ctx: StageContext, study: HS.StudyDir, export: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
        config = ctx.config
        doc = select_trial(
            export,
            study_id=identity["study_id"],
            policy=config.get("selection.policy"),
            max_refusal=config.get("selection.max_optimizer_refusal_rate"),
            max_kl=config.get("selection.max_optimizer_kl"),
            refusal_name=config.get("selection.optimizer_refusal_score"),
            kl_name=config.get("selection.optimizer_kl_score"),
            trial_id=config.get("abliteration.trial_id"),
            pareto_position=config.get("abliteration.pareto_trial_index"),
            allow_ineligible=config.get("selection.allow_ineligible_choice"),
        )
        path = ctx.metrics_path("selection-candidates.json")
        atomic_write_json(path, doc)
        ctx.add_output(path, "selection_candidates")
        study.history.append(
            "failure" if doc["error"] else "completion",
            "select",
            f"inv-{uuid.uuid4().hex[:12]}",
            ctx.run.run_id,
            selection_fingerprint=doc["selection_fingerprint"],
            selection={k: doc[k] for k in ("chosen", "origin", "policy", "constraints", "manual", "eligible_trials", "complete_trials", "error")},
        )
        if doc["error"]:
            err = doc["error"]
            raise PipelineError(ExitCode(err["exit_code"]), err["error_id"], err["message"], hint=err["hint"])
        chosen = doc["chosen"]
        if chosen["ineligible_override"]:
            ctx.warn(f"Exporting trial {chosen['index']} although it fails the optimizer constraints (selection.allow_ineligible_choice)")
        return doc

    def _export(self, ctx: StageContext, plan, record, study: HS.StudyDir, native, export, selection, engine, candidates_root: Path, commit: str):
        run, config = ctx.run, ctx.config
        chosen = selection["chosen"]
        export_fp = sha256_obj(
            {
                "study_id": selection["study_id"],
                "trial": [chosen["number"], chosen["index"]],
                "engine": commit,
                "export_strategy": native.get("export_strategy"),
                "source": plan["source_artifact_id"],
            }
        )
        final = candidates_root / f"c-{export_fp[:12]}-t{chosen['index']}"
        if self._reusable(ctx, final):
            engine["phases"]["export"] = {"ran": False, "reason": "validated candidate for this export fingerprint exists"}
            return final, export_fp, export
        partial = self._prepare_target(ctx, final, candidates_root)
        state = HS.study_state(export)
        planned = {
            "n_trials": state["trials"],
            "trial_index": chosen["pareto_position"],
            "model_action": "save",
            "trial": {"number": chosen["number"], "index": chosen["index"]},
            "target": str(final),
        }
        # The intent records the stored n_trials/finished that export-only overwrites (step0-feasibility.md C5).
        invocation = study.history.intent("export", run.run_id, export_fingerprint=export_fp, planned=planned, before=state)
        self._study_tool(
            ctx,
            record,
            [
                "set-settings",
                "--journal",
                str(study.journal()),
                "--n-trials",
                str(state["trials"]),
                "--trial-index",
                str(chosen["pareto_position"]),
                "--model-action",
                "save",
            ],
            "study-set-export",
        )
        run_native = {
            **native,
            "checkpoint_action": "continue",
            "model_action": "save",
            "n_trials": state["trials"],
            "trial_index": chosen["pareto_position"],
            "save_directory": str(partial),
        }
        grace = float(config.get("abliteration.shutdown_grace_minutes")) * 60
        ctx.info(f"Exporting trial {chosen['index']} (export only, zero new trials)")
        result, outcome, info = self._run_heretic(ctx, record, study, run_native, "heretic-export", grace_seconds=grace, timeout_seconds=grace)
        engine["export"] = info
        after = self._read_study(ctx, record, study, "study-after-export")
        astate = HS.study_state(after)
        if result.interrupted:
            remove_tree(partial)
            study.history.append("interruption", "export", invocation, run.run_id, after=astate, trials_added=astate["trials"] - state["trials"])
            raise Interrupted("Heretic export interrupted; it is redone on resume (zero new trials)")
        error = None
        problems = H.candidate_ok(partial)
        if astate["trial_set_sha256"] != state["trial_set_sha256"]:
            error = PipelineError(
                ExitCode.ENGINE,
                "E_EXPORT_ADDED_TRIALS",
                f"Export-only changed the study's trial set ({state['trials']} -> {astate['trials']} trials)",
                hint="This should not happen with the pinned engine (docs/step0-feasibility.md); see the study history.",
            )
        elif problems:
            error = self._diagnose(outcome, result, problems, "optimize")
        elif outcome.restored_trial != chosen["index"]:
            error = PipelineError(
                ExitCode.ENGINE,
                "E_EXPORT_TRIAL_MISMATCH",
                f"Heretic restored trial {outcome.restored_trial}, but trial {chosen['index']} (Pareto position {chosen['pareto_position']}) was selected",
                hint="The Pareto order differs from the study tool's; the export was discarded.",
            )
        if error is not None:
            remove_tree(partial)
            study.history.append(
                "failure", "export", invocation, run.run_id, after=astate, trials_added=astate["trials"] - state["trials"], error_id=error.error_id
            )
            raise error
        if result.returncode != 0:
            ctx.warn(f"Heretic exited with status {result.returncode} after saving a verified export of trial {chosen['index']}")
        os.replace(partial, final)
        study.history.append(
            "completion",
            "export",
            invocation,
            run.run_id,
            export_fingerprint=export_fp,
            after=astate,
            trials_added=0,
            export={**planned["trial"], "pareto_position": chosen["pareto_position"], "restored_trial": outcome.restored_trial, "target": str(final)},
        )
        engine["phases"]["export"] = {"ran": True, "trial": chosen["index"], "restored_trial": outcome.restored_trial}
        return final, export_fp, after

    @staticmethod
    def _diagnose_search(outcome: H.EngineOutcome, result, complete: int) -> PipelineError:
        tail = " | ".join(result.tail[-6:])
        if outcome.config_error:
            return PipelineError(ExitCode.ENGINE, "E_HERETIC_CONFIG", "Heretic rejected its configuration", hint=f"See {result.stdout_log}: {tail}")
        if result.budget_stopped and complete == 0:
            return PipelineError(
                ExitCode.ENGINE,
                "E_HERETIC_BUDGET_EXHAUSTED",
                "The runtime budget ended before any trial completed",
                hint="Increase --max-runtime-minutes; resuming continues the same study.",
            )
        if result.timed_out or result.killed:
            return PipelineError(ExitCode.ENGINE, "E_HERETIC_TIMEOUT", "Heretic did not stop within the grace period", hint=tail)
        return PipelineError(
            ExitCode.ENGINE,
            "E_HERETIC_SEARCH_FAILED",
            f"Heretic search exited ({result.returncode}) with {complete} completed trials",
            hint=f"Resuming continues the same study. Logs: {result.stdout_log}. Last lines: {tail}",
        )

    @staticmethod
    def _diagnose(outcome: H.EngineOutcome, result, problems: list[str], label: str) -> PipelineError:
        tail = " | ".join(result.tail[-6:])
        if outcome.config_error:
            return PipelineError(ExitCode.ENGINE, "E_HERETIC_CONFIG", "Heretic rejected its configuration", hint=f"See {result.stdout_log}: {tail}")
        if result.timed_out or result.killed:
            return PipelineError(ExitCode.ENGINE, "E_HERETIC_TIMEOUT", "Heretic did not finish within the budget plus grace period", hint=tail)
        if label == "exact_reproduction_attempt":
            return PipelineError(
                ExitCode.ENGINE,
                "E_REPRODUCE_FAILED",
                "Heretic did not save a reproduced model (environment mismatches are refused unless ignore_environment_mismatches=true)",
                hint=tail,
            )
        return PipelineError(
            ExitCode.ENGINE,
            "E_HERETIC_NO_OUTPUT",
            f"Heretic exited ({result.returncode}) without a valid checkpoint: {problems}",
            hint=f"Exit status 0 does not imply success for Heretic. Logs: {result.stdout_log}. Last lines: {tail}",
        )
