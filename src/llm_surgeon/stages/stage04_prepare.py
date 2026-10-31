"""Stage 04 - prepare_engine: inspect the *installed* Heretic, translate our config into native
settings, validate them against the installed Settings model, and decide the experiment label.

No architecture compatibility resolver: the model is never classified here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import configspec
from ..adapters import heretic as H
from ..adapters import toml_io
from ..config import load_yaml
from ..errors import ExitCode, PipelineError
from ..evaluation.baselines import published_method
from ..fsutil import atomic_write_json, atomic_write_text, read_json, sha256_file
from .base import Stage, StageContext
from .stage02_acquire import template_path


class PrepareEngineStage(Stage):
    id = "04"
    name = "prepare_engine"
    title = "Prepare Heretic invocation"
    config_keys = (
        "model.trust_remote_code",
        "abliteration.engine",
        "abliteration.experiment",
        "abliteration.engine_repo",
        "abliteration.engine_revision",
        "abliteration.heretic_config",
        "abliteration.max_trials",
        "abliteration.n_startup_trials",
        "abliteration.seed",
        "abliteration.reproduce",
        "evaluation.published_baselines",
    )

    def dependencies(self, config) -> tuple[str, ...]:
        return ("01", "02")

    def applicable(self, config) -> tuple[bool, str]:
        return (True, "") if config.get("abliteration.enabled") else (False, "abliteration disabled (quantize-only)")

    def components_used(self, config) -> set[str]:
        return {"heretic"}

    def extra_fingerprint(self, run) -> dict[str, Any]:
        tpl = template_path(run.config)
        base = run.config.get("evaluation.published_baselines")
        return {
            "template_sha256": sha256_file(tpl) if tpl.is_file() else None,
            "baselines_sha256": sha256_file(base) if base and Path(base).is_file() else None,
        }

    def target_outputs(self, config) -> list[str]:
        return ["runs/<run>/checkpoints/heretic/engine-plan.json", "runs/<run>/checkpoints/heretic/prepared-config.toml"]

    @staticmethod
    def _lineage_seed(run) -> int:
        """Derived seed shared by a run and its forks, so a fork can continue the parent's Heretic study."""
        if run.parent_run_id:
            parent = run.workspace / "runs" / run.parent_run_id / "receipts" / f"{PrepareEngineStage.id}-{PrepareEngineStage.name}.json"
            if parent.exists():
                seed = (read_json(parent).get("diagnostics") or {}).get("seed")
                if seed is not None:
                    return int(seed)
        return H.derived_seed(run.run_id)

    def _introspect(self, ctx: StageContext, record) -> dict[str, Any]:
        out = ctx.run.run_dir / "checkpoints" / "heretic" / "introspection.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        result = ctx.run.runner.run(
            [*record.script("heretic_introspect.py"), "--out", str(out)],
            cwd=None,
            env=ctx.run.tool_env(record),
            log_dir=ctx.log_dir,
            name="heretic-introspect",
            timeout_seconds=900,
        )
        if not result.ok or not out.exists():
            raise PipelineError(
                ExitCode.ENGINE,
                "E_HERETIC_INTROSPECT",
                "Could not introspect the installed Heretic",
                hint=f"See {result.stderr_log}: " + " | ".join(result.tail[-5:]),
            )
        data = read_json(out)
        ctx.add_output(out, "heretic_introspection")
        help_res = ctx.run.runner.run(
            [*record.cmd("heretic"), "--help"],
            cwd=None,
            env=ctx.run.tool_env(record, COLUMNS="200", NO_COLOR="1"),
            log_dir=ctx.log_dir,
            name="heretic-help",
            timeout_seconds=600,
        )
        help_path = Path(help_res.stdout_log)
        if not help_res.ok or "--model" not in help_path.read_text(encoding="utf-8", errors="replace"):
            raise PipelineError(ExitCode.ENGINE, "E_HERETIC_HELP", "`heretic --help` failed or did not list --model", hint=help_res.stderr_log)
        data["help_sha256"] = sha256_file(help_path)
        return data

    def execute(self, ctx: StageContext) -> None:
        run, config = ctx.run, ctx.config
        record = run.components(ctx.log_dir).get("heretic")
        intro = self._introspect(ctx, record)
        expected = record.versions.get("heretic_commit")
        installed = (intro.get("direct_url") or {}).get("commit")
        if expected and installed and installed != expected:
            raise PipelineError(ExitCode.ENGINE, "E_PIN_MISMATCH", f"Installed Heretic commit {installed} != provisioned {expected}")
        ctx.tool_revisions["heretic"] = {"version": intro.get("heretic_version"), "commit": installed or expected}
        fields = intro.get("settings_fields") or []
        if not fields:
            raise PipelineError(ExitCode.ENGINE, "E_HERETIC_INTROSPECT", "Introspection returned no Settings fields")
        source = run.registry.get("source")
        if source is None:
            raise PipelineError(ExitCode.DEPENDENCY, "E_MISSING_DEPENDENCY", "No source artifact registered (stage 02)")
        source_path = source["manifest_data"]["artifact"]["uri"]
        seed = config.get("abliteration.seed")
        seed = int(seed) if seed is not None else self._lineage_seed(run)
        workdir = run.run_dir / "checkpoints" / "heretic"
        experiment = config.get("abliteration.experiment")
        limitations: list[str] = []
        reproduce_doc = None
        reproduce_notes: list[str] = []
        if experiment == "optimize":
            template = H.native_template(template_path(config))
            native = H.build_optimize_config(
                template,
                model_path=source_path,
                n_trials=config.get("abliteration.max_trials"),
                n_startup_trials=config.get("abliteration.n_startup_trials"),
                seed=seed,
                trial_index=0,  # stage 05 sets trial_index/model_action per phase (search-only, export-only)
                save_directory="<set by stage 05>",
                checkpoint_action="restart",
            )
            label = "optimize"
        else:
            template = H.native_template(template_path(config))
            native = {**template, "model": source_path, "save_directory": "<set by stage 05>"}
            rep_diag = (run.load_receipt("02") or {}).get("diagnostics", {}).get("reproduce", {})
            if rep_diag.get("exact_file_available"):
                original = read_json(rep_diag["path"])
                reproduce_doc, reproduce_notes = H.patch_reproduce_file(original, source_path)
                native["ignore_mismatches"] = bool(config.get("abliteration.reproduce.ignore_environment_mismatches"))
                label = "exact_reproduction_attempt"
                limitations.append("Bitwise identity is only claimed if output hashes match the reproduce.json hashes.")
            elif config.get("abliteration.reproduce.allow_approximate"):
                baselines = config.get("evaluation.published_baselines")
                if not baselines:
                    raise PipelineError(
                        ExitCode.UNSUPPORTED, "E_REPRODUCE_NO_PARAMETERS", "Approximate reproduction needs evaluation.published_baselines with published_method"
                    )
                method = published_method(load_yaml(baselines), config.get("abliteration.reproduce.reference_repo"))
                if method is None:
                    raise PipelineError(ExitCode.UNSUPPORTED, "E_REPRODUCE_NO_PARAMETERS", f"No published_method in {baselines}")
                reproduce_doc, reproduce_notes = H.synthesize_reproduce_file(
                    method,
                    source_path,
                    seed,
                    {configspec.HARMLESS_DATASET: configspec.HARMLESS_DATASET_COMMIT, configspec.HARMFUL_DATASET: configspec.HARMFUL_DATASET_COMMIT},
                )
                native["ignore_mismatches"] = True
                label = "approximate_reproduction"
                limitations.extend(reproduce_notes)
                if method.get("claimed_engine_version"):
                    limitations.append(
                        f"Reference claims Heretic {method['claimed_engine_version']} with {method.get('claimed_method', 'an unreleased method')}; "
                        "the pinned engine's implementation may differ from the one used for the reference."
                    )
            else:
                raise PipelineError(
                    ExitCode.UNSUPPORTED,
                    "E_EXACT_REPRODUCTION_UNAVAILABLE",
                    "experiment=reproduce but no reproduce.json exists for the reference and approximate reproduction is not allowed",
                    hint="Set abliteration.reproduce.allow_approximate: true (results are labelled approximate_reproduction) or use --experiment optimize.",
                )
        unknown = H.check_settings(native, fields)
        if unknown:
            raise PipelineError(
                ExitCode.CONFIG,
                "E_HERETIC_UNKNOWN_SETTING",
                f"Settings unknown to the installed Heretic: {', '.join(unknown)}",
                hint="Remove them from the template; see heretic-help output in the stage logs.",
            )
        modifiers = H.modifier_plugins(reproduce_doc["settings"] if reproduce_doc else native)
        available = intro.get("modifiers") or {}
        missing = [m for m in modifiers if not available.get(m)]
        if missing:
            raise PipelineError(
                ExitCode.UNSUPPORTED,
                "E_MODIFIER_UNAVAILABLE",
                f"Requested modifier plugin(s) not importable in the pinned Heretic: {missing}",
                hint="ARA exists only on Heretic master after v1.4.0; keep the default engine_revision.",
            )
        for spec in H.dataset_specs(native):
            ds = spec["dataset"]
            if (ds.startswith(("/", ".")) or "\\" in ds) and not Path(ds).exists():
                raise PipelineError(ExitCode.CONFIG, "E_DATASET_PATH", f"Local dataset path does not exist: {ds}")
        if native.get("quantization", "none") != "none":
            ctx.warn(
                f"Heretic loads the model with quantization={native['quantization']}: this is memory-saving loading, not a deployment "
                "quantization; the merged master's precision is recorded from the saved files."
            )
        if config.get("model.trust_remote_code"):
            ctx.warn(
                "model.trust_remote_code=true applies to Project A loaders only. Heretic has no setting for it and transformers would "
                "prompt on stdin, which is closed: remote-code models fail in stage 05."
            )
        preview = workdir / "prepared-config.toml"
        atomic_write_text(preview, toml_io.dumps(native))
        toml_io.load(preview)
        reproduce_path = None
        if reproduce_doc is not None:
            reproduce_path = workdir / f"{label}.reproduce.json"
            atomic_write_json(reproduce_path, reproduce_doc)
            ctx.add_output(reproduce_path, "reproduce_input", sha256=sha256_file(reproduce_path))
        plan = {
            "label": label,
            "experiment": experiment,
            "native": native,
            "seed": seed,
            # Native keys computed from the budget (n_startup_trials = max_trials // 3 when unset). They are not
            # part of the study identity: a continued study keeps its stored value whatever the new budget.
            "budget_derived_keys": ["n_startup_trials"] if label == "optimize" and config.get("abliteration.n_startup_trials") is None else [],
            "reproduce_json": str(reproduce_path) if reproduce_path else None,
            "reproduce_changes_or_assumptions": reproduce_notes,
            "limitations": limitations,
            "heretic_commit": installed or expected,
            "settings_fields": len(fields),
            "modifiers": modifiers,
            "datasets": H.dataset_specs(reproduce_doc["settings"] if reproduce_doc else native),
            "source_path": source_path,
            "source_artifact_id": source["artifact_id"],
        }
        plan_path = workdir / "engine-plan.json"
        atomic_write_json(plan_path, plan)
        ctx.add_output(plan_path, "engine_plan", sha256=sha256_file(plan_path))
        ctx.add_output(preview, "native_config_preview", sha256=sha256_file(preview))
        ctx.input_artifacts.append(source["artifact_id"])
        ctx.diagnostics.update(label=label, seed=seed, modifiers=modifiers, limitations=limitations, plan=str(plan_path))
        for item in limitations:
            ctx.warn(item)
        ctx.info(f"Engine plan ready: {label} with {', '.join(modifiers)} (seed {seed})")
