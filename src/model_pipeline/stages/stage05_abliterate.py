"""Stage 05 - abliterate: run Heretic headlessly with a bounded budget and recover a candidate.

* The study journal lives in a run-private directory keyed by the study-relevant settings, so an
  interrupted stage resumes the same Optuna study (``checkpoint_action=continue``).
* Before continuing an existing study, its stored settings are updated (n_trials/trial_index) via
  Optuna's public API so Heretic exports the requested Pareto trial.
* Output is written to ``<candidate>.partial`` and renamed only after validation.
"""

from __future__ import annotations

import os
from typing import Any

from ..adapters import heretic as H
from ..artifacts.safetensors_meta import dtype_summary, tensor_diff, tensor_map
from ..errors import ExitCode, Interrupted, PipelineError
from ..fsutil import atomic_write_json, disk_free_bytes, human_bytes, read_json, remove_tree, sha256_file
from .base import Stage, StageContext


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

    def target_outputs(self, config) -> list[str]:
        return ["workspace/models/candidates/<run>/<candidate>/ (Heretic merged export)", "runs/<run>/metrics/candidate.json"]

    def _study_tool(self, ctx: StageContext, record, args: list[str], name: str) -> dict[str, Any]:
        out = ctx.log_dir / f"{name}.json"
        result = ctx.run.runner.run(
            [*record.script("heretic_study_tool.py"), *args, "--out", str(out)],
            cwd=None,
            env=ctx.run.child_env(),
            log_dir=ctx.log_dir,
            name=name,
            timeout_seconds=900,
        )
        if not result.ok or not out.exists():
            raise PipelineError(
                ExitCode.ENGINE, "E_HERETIC_STUDY_TOOL", f"Study helper '{name}' failed", hint=f"{result.stderr_log}: {' | '.join(result.tail[-4:])}"
            )
        return read_json(out)

    def execute(self, ctx: StageContext) -> None:
        run, config = ctx.run, ctx.config
        prep = run.load_receipt("04") or {}
        plan = read_json(prep["diagnostics"]["plan"])
        record = run.components(ctx.log_dir).get("heretic")
        native = dict(plan["native"])
        label = plan["label"]
        commit = plan["heretic_commit"] or "unknown"
        candidates_root = run.workspace / "models" / "candidates" / run.run_id
        source_size = sum(f["size_bytes"] for f in run.registry.get("source")["manifest_data"]["artifact"]["files"])
        if label == "optimize":
            shash = H.study_hash(native, commit)
            workdir = run.run_dir / "checkpoints" / "heretic" / f"study-{shash}"
            candidate_name = f"c-{shash[:8]}-p{plan['pareto_trial_index']}"
        else:
            shash = None
            workdir = run.run_dir / "checkpoints" / "heretic" / label
            candidate_name = f"c-{label}"
        final = candidates_root / candidate_name
        partial = final.with_name(final.name + ".partial")
        workdir.mkdir(parents=True, exist_ok=True)
        journals = sorted((workdir / "checkpoints").glob("*.jsonl")) if (workdir / "checkpoints").is_dir() else []
        engine: dict[str, Any] = {"workdir": str(workdir), "label": label, "commit": commit}
        outcome = H.EngineOutcome()
        if final.is_dir() and not H.candidate_ok(final):
            ctx.info(f"Validated candidate {final} already exists from an earlier attempt; not re-running Heretic")
            engine["reused_candidate"] = True
        else:
            if final.exists():
                remove_tree(final)
            if partial.exists():
                ctx.warn(f"Removing incomplete export from an interrupted attempt: {partial}")
                remove_tree(partial)
            free = disk_free_bytes(candidates_root.parent if candidates_root.parent.exists() else run.workspace)
            if free < 1.05 * source_size:
                raise PipelineError(ExitCode.ENGINE, "E_DISK_INSUFFICIENT", f"Candidate export needs ~{human_bytes(source_size)}; {human_bytes(free)} free")
            candidates_root.mkdir(parents=True, exist_ok=True)
            reproduce_doc = None
            if label == "optimize":
                if journals:
                    native["checkpoint_action"] = "continue"
                    update = self._study_tool(
                        ctx,
                        record,
                        ["update-settings", "--journal", str(journals[0]), "--n-trials", str(native["n_trials"]), "--trial-index", str(native["trial_index"])],
                        "study-update",
                    )
                    engine["continued_study"] = update
                    ctx.info(f"Continuing existing study ({update.get('trials', '?')} trials recorded)")
                else:
                    native["checkpoint_action"] = "restart"
            else:
                reproduce_doc = read_json(plan["reproduce_json"])
            native["save_directory"] = str(partial)
            files = H.write_engine_files(workdir, native, reproduce_doc)
            engine.update(files)
            argv = [*record.cmd("heretic")]
            if reproduce_doc is not None:
                argv += ["--reproduce", files["reproduce_json"]]
            budget_min = config.get("abliteration.max_runtime_minutes")
            grace = float(config.get("abliteration.shutdown_grace_minutes")) * 60
            budget = float(budget_min) * 60 if (budget_min and label == "optimize") else None
            timeout = (float(budget_min) * 60 + grace) if (budget_min and label != "optimize") else None
            ctx.info(f"Running Heretic ({label}) in {workdir}" + (f" with a {budget_min} min budget" if budget else ""))
            result = run.runner.run(
                argv,
                cwd=workdir,
                env=H.engine_env(run.child_env()),
                log_dir=ctx.log_dir,
                name="heretic",
                budget_seconds=budget,
                grace_seconds=grace,
                timeout_seconds=timeout,
                on_line=outcome.on_line,
            )
            engine.update(result=result.to_dict(), outcome=outcome.to_dict())
            if result.interrupted:
                if partial.exists():
                    remove_tree(partial)
                raise Interrupted("Heretic interrupted; the study journal is kept and will be continued on resume")
            problems = H.candidate_ok(partial)
            if problems:
                if partial.exists():
                    remove_tree(partial)
                ctx.diagnostics["engine"] = engine
                raise self._diagnose(outcome, result, problems, label)
            os.replace(partial, final)
            if result.budget_stopped:
                ctx.warn(f"Budget of {budget_min} min reached after {outcome.trials_started} started trials; exported the best completed trial")
        summary: dict[str, Any]
        if label == "optimize":
            journals = sorted((workdir / "checkpoints").glob("*.jsonl"))
            if not journals:
                raise PipelineError(ExitCode.ENGINE, "E_HERETIC_NO_JOURNAL", f"No Optuna journal under {workdir / 'checkpoints'}")
            export = self._study_tool(ctx, record, ["export", "--journal", str(journals[0]), "--trial-index", str(plan["pareto_trial_index"])], "study-export")
            study_path = ctx.metrics_path("heretic_study.json")
            atomic_write_json(study_path, export)
            ctx.add_output(study_path, "heretic_study")
            summary = H.summarize_study(
                export, outcome.restored_trial, config.get("selection.optimizer_refusal_score"), config.get("selection.optimizer_kl_score")
            )
            if outcome.restored_trial is not None and summary["exported_trial_index"] not in (None, outcome.restored_trial):
                ctx.warn(f"Heretic restored trial {outcome.restored_trial} but the Pareto export points at {summary['exported_trial_index']}")
        else:
            doc = read_json(plan["reproduce_json"])
            summary = {
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
        source_tensors = tensor_map(run.registry.get("source")["manifest_data"]["artifact"]["uri"])
        cand_tensors = tensor_map(final)
        diff = tensor_diff(source_tensors, cand_tensors)
        if diff["missing_count"]:
            ctx.warn(
                f"Candidate lacks {diff['missing_count']} source tensors (prefixes {diff['missing_prefixes']}); "
                "see selection.graft_missing_source_tensors to restore them verbatim"
            )
        candidate = {
            "candidate_id": candidate_name,
            "path": str(final),
            "label": label,
            "modifiers": plan.get("modifiers", []),
            "study_hash": shash,
            "seed": plan["seed"],
            "engine": engine,
            "summary": summary,
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
            label=label,
            trials_started=outcome.trials_started,
            budget_stopped=engine.get("result", {}).get("budget_stopped"),
        )

    @staticmethod
    def _diagnose(outcome: H.EngineOutcome, result, problems: list[str], label: str) -> PipelineError:
        tail = " | ".join(result.tail[-6:])
        if outcome.config_error:
            return PipelineError(ExitCode.ENGINE, "E_HERETIC_CONFIG", "Heretic rejected its configuration", hint=f"See {result.stdout_log}: {tail}")
        if result.budget_stopped and outcome.trials_started <= 1:
            return PipelineError(
                ExitCode.ENGINE,
                "E_HERETIC_BUDGET_EXHAUSTED",
                "The runtime budget ended before any trial completed",
                hint="Increase --max-runtime-minutes; resuming continues the same study.",
            )
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
