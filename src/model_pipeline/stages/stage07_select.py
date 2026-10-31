"""Stage 07 - select_master: apply gates and promote the candidate to the immutable master.

Nothing is promoted silently: the decision table (``evaluation.promotion.decide``) yields
``promoted:qualified``, ``promoted:research``, ``promoted:ungated`` or ``not_promoted``, recorded
with its reason in ``selection.json`` and the master manifest. The master's precision is read from
the saved tensors (never assumed) and its lineage points at the source artifact.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..artifacts.manifest import build_manifest
from ..artifacts.safetensors_meta import dtype_summary, graft_tensors, tensor_diff, tensor_map
from ..errors import ExitCode, PipelineError
from ..evaluation.promotion import decide, evidence_line
from ..evaluation.qualification import evaluate_gates
from ..fsutil import atomic_write_json, file_inventory, link_or_copy_tree, make_read_only, read_json, remove_tree, safe_component, sha256_obj
from .base import Stage, StageContext


def precision_dir(precision: str) -> str:
    return precision if precision in ("bf16", "f16", "f32") else "mixed"


class SelectMasterStage(Stage):
    id = "07"
    name = "select_master"
    title = "Gate and promote master"
    config_keys = ("selection", "storage.link_strategy", "abliteration.save_intermediate")

    def dependencies(self, config) -> tuple[str, ...]:
        return ("05", "06")

    def applicable(self, config) -> tuple[bool, str]:
        return (True, "") if config.get("abliteration.enabled") else (False, "abliteration disabled (quantize-only)")

    def target_outputs(self, config) -> list[str]:
        return ["workspace/exports/<precision>/<family>-<label>-<run>/ (immutable master)", "runs/<run>/metrics/selection.json"]

    def execute(self, ctx: StageContext) -> None:
        run, config = ctx.run, ctx.config
        candidate = read_json(run.load_receipt("05")["diagnostics"]["candidate"])
        qual_receipt = run.load_receipt("06")
        qualification = None
        if qual_receipt and qual_receipt.get("outcome") == "succeeded" and qual_receipt["diagnostics"].get("qualification"):
            qualification = read_json(qual_receipt["diagnostics"]["qualification"])
            qualification["metrics"] = qualification["independent"]
        selection_cfg = config.get("selection")
        gates = evaluate_gates(selection_cfg, candidate["summary"], qualification)
        selected = candidate.get("selection") or {}
        origin = selected.get("origin") or {"kind": "reproduction" if candidate["label"] != "optimize" else "unknown"}
        promotion = decide(gates, selection_cfg, run_mode=candidate["label"], origin=origin)
        ungated = promotion["outcome"] == "promoted:ungated"
        decision = {
            "candidate": candidate["candidate_id"],
            **promotion,
            "failed": [g["gate"] for g in promotion["gates"] if g["required"] and g["status"] == "failed"],
            "unmeasured": [g["gate"] for g in promotion["gates"] if g["result"] == "unmeasured"],
            "qualification_mode": qualification["mode"] if qualification else "off",
        }
        for gate in decision["unmeasured"]:
            ctx.warn(f"Gate {gate} could not be evaluated (metric not available for {candidate['label']})")
        for warning in promotion["warnings"]:
            ctx.warn(warning.capitalize())
        if not promotion["promoted"]:
            atomic_write_json(ctx.metrics_path("selection.json"), decision)
            blocking = [g for g in promotion["gates"] if g["gate"] in promotion["triggered_gates"]]
            table = "; ".join(f"{g['gate']}={g['value']} ({g['status']}, needs {g['op']} {g['threshold']})" for g in blocking)
            if promotion["reason"] == "required_gate_failed":
                error_id, what = "E_GATES_FAILED", "failed promotion gates"
                hint = (
                    "No master was promoted. Choose another trial (--resume RUN --fork --trial-id N: zero new trials, "
                    "see metrics/selection-candidates.json) or override specific gates with selection.override_gates."
                )
            else:
                error_id, what = "E_GATES_UNMEASURED", "has required gates without evidence"
                hint = (
                    "No master was promoted. Measure the missing metrics, set selection.profile: research to promote with "
                    "missing evidence (labelled research), or list the gates in selection.optional_gates (see docs/evaluation.md)."
                )
            raise PipelineError(ExitCode.GATE, error_id, f"Candidate {candidate['candidate_id']} {what}: {table}", hint=hint)
        if ungated:
            ctx.warn(f"Promoting with overridden gates ({', '.join(promotion['triggered_gates'])}); master is labelled 'ungated'")
        elif promotion["outcome"] == "promoted:research":
            ctx.warn(f"Promoting without evidence for {', '.join(promotion['triggered_gates'])} (research profile); master is labelled 'research'")
        source = run.registry.get("source")
        source_manifest = source["manifest_data"]
        label = candidate["label"]
        summary = candidate["dtype_summary"]
        name = safe_component(f"{config.get('model.family')}-{label}-{run.run_id}")
        dest = run.workspace / "exports" / precision_dir(summary["precision"]) / name
        partial = dest.with_name(dest.name + ".partial")
        for path in (dest, partial):
            if path.exists():
                remove_tree(path)
        semantics = link_or_copy_tree(candidate["path"], partial, config.get("storage.link_strategy"))
        grafted: dict[str, Any] = {"grafted": []}
        prefixes = selection_cfg["graft_missing_source_tensors"]
        if prefixes:
            grafted = graft_tensors(Path(source_manifest["artifact"]["uri"]), partial, prefixes)
            if grafted["grafted"]:
                ctx.warn(f"Grafted {len(grafted['grafted'])} untouched source tensors ({', '.join(prefixes)}) into the master")
        os.replace(partial, dest)
        tensors = tensor_map(dest)
        summary = dtype_summary(tensors)
        diff = tensor_diff(tensor_map(source_manifest["artifact"]["uri"]), tensors)
        ctx.info(f"Hashing master files in {dest}")
        files = [e.to_dict() for e in file_inventory(dest)]
        try:
            make_read_only(dest)
            read_only = True
        except OSError:
            read_only = False
        engine_files = candidate["engine"]
        qualified = qualification is not None
        manifest = build_manifest(
            run_id=run.run_id,
            model_family=config.get("model.family"),
            source_model=source_manifest["source_model"],
            source_revision=source_manifest["source_revision"],
            parent_artifact_id=source["artifact_id"],
            kind="abliteration",
            method=f"heretic:{'+'.join(m.rsplit('.', 1)[-1] for m in candidate.get('modifiers', [])) or 'modifier'}:{label}",
            config_hash=engine_files.get("config_sha256") or sha256_obj(candidate["summary"]),
            tool_revision=engine_files.get("commit"),
            precision=summary["precision"],
            fmt="safetensors",
            uri=str(dest),
            files=files,
            inference={
                "backend": "transformers",
                "backend_revision": f"transformers {(qualification.get('runtime_versions') or {}).get('transformers')}",
                "tested_context": qualification.get("max_total_tokens"),
            }
            if qualified
            else None,
            validation={
                "integrity_passed": True,
                "inference_passed": bool(qualification["independent"]["reload_ok"]) if qualified else None,
                "modalities_tested": ["text"] if qualified and qualification["independent"]["reload_ok"] else [],
            },
            evaluation={
                "qualification_run_id": run.run_id if qualified else None,
                "metrics_provenance": f"{qualification['mode']}-qualification"
                if qualified
                else ("optimizer-only" if candidate["summary"].get("optimizer_scores") else "not-evaluated"),
            },
            extensions={
                "label": label,
                "ungated": ungated,
                "promotion": {k: promotion[k] for k in ("outcome", "reason", "deciding_gate", "triggered_gates", "profile", "run_mode", "candidate_origin")},
                "evidence_status": evidence_line(promotion),
                "gates": promotion["gates"],
                "candidate_id": candidate["candidate_id"],
                "heretic_parameters": candidate["summary"].get("exported_parameters"),
                "heretic_trial_index": candidate["summary"].get("exported_trial_index"),
                "selection": {k: selected.get(k) for k in ("chosen", "origin", "policy", "constraints", "selection_fingerprint", "export_fingerprint")}
                if selected
                else None,
                "optimizer_metrics": {k: candidate["summary"].get(k) for k in ("optimizer_refusal_rate", "optimizer_kl", "provenance")},
                "seed": candidate.get("seed"),
                "dtype_summary": summary,
                "tensor_diff_vs_source": {**diff, "missing_from_target": diff["missing_from_target"][:200]},
                "grafted_tensors": grafted,
                "storage_semantics": semantics,
                "read_only": read_only,
                "limitations": candidate.get("limitations", []),
            },
        )
        if manifest["validation"]["inference_passed"] is None:
            manifest["inference"] = {"backend": None, "backend_revision": None, "tested_context": None}
        run.registry.register(manifest, "master", dest, self.id)
        pointer = run.workspace / "models" / "selected" / run.run_id / "master.json"
        atomic_write_json(pointer, {"artifact_id": manifest["artifact_id"], "path": str(dest), "manifest": str(dest) + ".manifest.json"})
        decision.update(promoted=True, master=str(dest), artifact_id=manifest["artifact_id"], storage_semantics=semantics, ungated=ungated)
        out = ctx.metrics_path("selection.json")
        atomic_write_json(out, decision)
        ctx.add_output(out, "selection")
        ctx.add_output(pointer, "master_pointer")
        for entry in files:
            ctx.add_output(dest / entry["relative_path"], "master_file", sha256=entry["sha256"], size=entry["size_bytes"])
        ctx.input_artifacts.append(source["artifact_id"])
        ctx.output_artifacts.append(manifest["artifact_id"])
        ctx.diagnostics.update(
            master=str(dest), artifact_id=manifest["artifact_id"], ungated=ungated, promotion=promotion["outcome"], precision=summary["precision"]
        )
        if not config.get("abliteration.save_intermediate") and semantics == "copy":
            remove_tree(candidate["path"])
            ctx.info("Removed candidate checkpoint (save_intermediate=false and the master is an independent copy)")
