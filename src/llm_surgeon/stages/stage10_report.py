"""Stage 10 - report_export: run summary JSON, human-readable report, artifact usage examples.

``write_report`` is also called by the orchestrator after a failed run (best effort, marked
incomplete) so failures are documented even when stage 10 is never reached.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import __version__
from ..fsutil import atomic_write_json, atomic_write_text, human_bytes, read_json
from ..telemetry import utc_now
from .base import Stage, StageContext, load_receipt

if TYPE_CHECKING:  # pragma: no cover
    from ..orchestrator import Run


def _maybe(path: Path) -> Any:
    try:
        return read_json(path) if path.exists() else None
    except (OSError, ValueError):
        return None


def collect(run: Run) -> dict[str, Any]:
    from . import STAGES  # noqa: PLC0415

    receipts = {}
    for stage in STAGES:
        data = load_receipt(run.receipt_path(stage))
        if data:
            receipts[stage.id] = data
    measured = sum(r.get("duration_s") or 0 for r in receipts.values() if r.get("run_id") == run.run_id)
    rate = run.config.get("execution.hourly_rate_usd")
    artifacts = {}
    for role, entry in run.registry.roles().items():
        manifest = _maybe(Path(entry["manifest"]))
        if manifest:
            artifacts[role] = {
                "artifact_id": manifest["artifact_id"],
                "parent_artifact_id": manifest["parent_artifact_id"],
                "path": manifest["artifact"]["uri"],
                "manifest": entry["manifest"],
                "precision": manifest["precision"],
                "format": manifest["artifact"]["format"],
                "size_bytes": sum(f["size_bytes"] for f in manifest["artifact"]["files"]),
                "validation": manifest["validation"],
                "inference": manifest["inference"],
                "evaluation": manifest["evaluation"],
                "label": (manifest.get("extensions") or {}).get("label"),
                "promotion": (manifest.get("extensions") or {}).get("promotion"),
                "evidence_status": (manifest.get("extensions") or {}).get("evidence_status"),
                "derived_status": manifest.get("derived_status"),
                "assessments": len(manifest.get("assessments") or []),
            }
    for art in artifacts.values():
        if not art["evidence_status"]:
            art["evidence_status"] = derived_evidence(art, artifacts)
    metrics_dir = run.run_dir / "metrics"
    return {
        "schema_version": "1.0",
        "platform_version": __version__,
        "generated_utc": utc_now(),
        "run_id": run.run_id,
        "parent_run_id": run.parent_run_id,
        "config_hash": run.config.config_hash,
        "config_profiles": run.config.profiles,
        "model": {"ref": run.config.get("model.url") or run.config.get("model.local_path"), "family": run.config.get("model.family")},
        "stages": {
            sid: {
                "name": r.get("stage_name"),
                "outcome": r.get("outcome"),
                "partial": r.get("partial"),
                "duration_s": r.get("duration_s"),
                "warnings": r.get("warnings", []),
                "error": r.get("error"),
                "resources": r.get("resources"),
                "copied_from_run": r.get("copied_from_run"),
            }
            for sid, r in receipts.items()
        },
        "artifacts": artifacts,
        "environment": _maybe(metrics_dir / "preflight.json"),
        "components": _maybe(metrics_dir / "components.json"),
        "published_baselines": _maybe(metrics_dir / "published_baselines.json"),
        "candidate": _maybe(metrics_dir / "candidate.json"),
        "qualification": _maybe(metrics_dir / "qualification.json"),
        "selection": _maybe(metrics_dir / "selection.json"),
        "quantization": _maybe(metrics_dir / "quantization.json"),
        "verification": _maybe(metrics_dir / "verification.json"),
        "comparison": _maybe(metrics_dir / "comparison.json"),
        "time": {
            "measured_stage_seconds": round(measured, 1),
            "estimated_cost_usd": round(measured / 3600 * rate, 2) if rate is not None else None,
            "cost_note": "ESTIMATE: measured stage wall time x execution.hourly_rate_usd; excludes idle instance time."
            if rate is not None
            else "set execution.hourly_rate_usd for a cost estimate",
        },
    }


def derived_evidence(art: dict[str, Any], artifacts: dict[str, Any]) -> str:
    """Evidence status for artifacts that are not promoted masters (source, exports)."""
    loader = {True: "loader test passed", False: "loader test FAILED", None: "loader test not run"}[art["validation"]["inference_passed"]]
    parent = next((a for a in artifacts.values() if a["artifact_id"] == art["parent_artifact_id"]), None)
    if parent is None:
        return f"not promoted or qualified by this run; {loader}"
    origin = (parent.get("promotion") or {}).get("outcome") or "not promoted (source model)"
    return f"export of {parent['artifact_id'][:16]} ({origin}); this export: {loader}, not independently qualified"


def usage_examples(artifacts: dict[str, Any]) -> list[str]:
    lines = []
    for role, art in sorted(artifacts.items()):
        path = art["path"]
        if role == "master":
            lines += [
                f"### {role} ({art['precision']}, safetensors)",
                "```python",
                "from transformers import AutoModelForImageTextToText, AutoProcessor  # use AutoModelForCausalLM for text-only models",
                f'model = AutoModelForImageTextToText.from_pretrained(r"{path}", dtype="auto", device_map="auto")',
                f'processor = AutoProcessor.from_pretrained(r"{path}")',
                "```",
                "",
            ]
        elif role == "export:fp8":
            lines += [f"### {role} (compressed-tensors FP8)", "```bash", f"vllm serve {path} --max-model-len 8192", "```", ""]
        elif role.startswith("export:gguf:"):
            lines += [
                f"### {role}",
                "```bash",
                f"llama-cli -m {path}/<model>.gguf -c 8192 -ngl 99   # add --mmproj {path}/mmproj-*.gguf for images if present",
                "```",
                "Fit on a 16 GB GPU (e.g. RTX A4000) depends on context length, KV cache type and offload; it is not guaranteed.",
                "",
            ]
    return lines


def _with_delta(entry: dict[str, Any], name: str) -> str:
    pair = entry.get(name)
    if not pair:
        return "-"
    delta = ((entry.get("deltas") or {}).get(name) or {}).get("delta")
    return f"{pair['count']}/{pair['of']}" + (f" ({delta:+d})" if delta is not None else "")


def comparison_section(doc: dict[str, Any] | None) -> list[str]:
    if not doc:
        return []
    base = doc["baseline"]
    out = [
        "",
        "## Master vs export comparison",
        "",
        f"Exported runtime vs baseline runtime (transformers, {base['role']} `{base['artifact_id']}`): suite `{doc['suite']['id']}` "
        f"v{doc['suite']['version']}, the same rendered prompts on every backend; export gates: {(doc.get('export_gates') or {}).get('mode')}. "
        "Counts are correct/graded, refused/prompts and responses with integrity issues/responses; deltas are export minus baseline.",
        "",
        "| Artifact | Backend | Benign correct | Held-out refused | Integrity issues | Outcome | Contract gaps |",
        "|---|---|---|---|---|---|---|",
    ]
    if base.get("measured"):
        cells = " | ".join(_with_delta(base, k) for k in ("benign_correct", "heldout_refused", "integrity_issues"))
        out.append(f"| baseline ({base['role']}) | transformers | {cells} | - | - |")
    else:
        out.append(f"| baseline ({base['role']}) | transformers | unmeasured: {base.get('reason')} | | | | |")
    for role, entry in sorted((doc.get("exports") or {}).items()):
        gaps = ", ".join(g["field"] for g in entry.get("contract_gaps") or []) or "-"
        if entry.get("deltas") is None:
            out.append(f"| {role} | {entry.get('backend')} | unmeasured: {entry.get('unmeasured_reason')} | | | {entry['outcome']} | {gaps} |")
        else:
            cells = " | ".join(_with_delta(entry, k) for k in ("benign_correct", "heldout_refused", "integrity_issues"))
            out.append(f"| {role} | {entry.get('backend')} | {cells} | {entry['outcome']} | {gaps} |")
    if doc.get("redacted"):
        out += ["", "Prompts and responses are redacted (hashes only; exact text replay unavailable)."]
    return out


def render_markdown(summary: dict[str, Any], final: bool) -> str:
    out = [f"# Run report {summary['run_id']}", ""]
    if not final:
        out += ["> **Incomplete run** - generated after a failure or interruption; see failed stages below.", ""]
    out += [
        f"- Platform version: {summary['platform_version']}",
        f"- Model: `{summary['model']['ref']}` (family `{summary['model']['family']}`)",
        f"- Config hash: `{summary['config_hash']}`",
        *[f"- Profile: `{p['path']}` (sha256 `{p['sha256'][:16]}`)" for p in summary.get("config_profiles") or []],
        f"- Generated: {summary['generated_utc']}",
        "",
        "## Stages",
        "",
        "| Stage | Outcome | Duration (s, measured) | Warnings |",
        "|---|---|---:|---|",
    ]
    for sid, st in sorted(summary["stages"].items()):
        outcome = st["outcome"] + (" (partial)" if st.get("partial") else "") + (f" (reused from {st['copied_from_run']})" if st.get("copied_from_run") else "")
        out.append(f"| {sid} {st['name']} | {outcome} | {st['duration_s'] if st['duration_s'] is not None else '-'} | {len(st['warnings'])} |")
    errors = [(sid, st["error"]) for sid, st in sorted(summary["stages"].items()) if st.get("error")]
    if errors:
        out += ["", "## Failures", ""]
        for sid, err in errors:
            out.append(f"- **{sid}** `{err['error_id']}` (exit {err['exit_code']}): {err['message']}")
            if err.get("hint"):
                out.append(f"  - hint: {err['hint']}")
    out += [
        "",
        "## Artifacts",
        "",
        "| Role | Artifact ID | Precision | Size | Integrity | Loader test | Modalities tested | Metrics provenance |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for role, art in sorted(summary["artifacts"].items()):
        v = art["validation"]
        loader = {True: "passed", False: "FAILED", None: "not run (unverified)"}[v["inference_passed"]]
        out.append(
            f"| {role} | `{art['artifact_id']}` | {art['precision']} | {human_bytes(art['size_bytes'])} | {'ok' if v['integrity_passed'] else 'FAILED'} | "
            f"{loader} | {', '.join(v['modalities_tested']) or '-'} | {art['evaluation']['metrics_provenance']} |"
        )
    out += ["", "### Evidence status", ""]
    for role, art in sorted(summary["artifacts"].items()):
        out.append(f"- **{role}**: {art.get('evidence_status') or '-'}")
        status = art.get("derived_status")
        if status:
            parts = ", ".join(f"{part} `{aid}`" if aid else f"{part}: registration" for part, aid in status.items())
            out.append(f"  - status from assessment records ({art['assessments']} in `assessments/{art['artifact_id']}/`): {parts}")
    out += comparison_section(summary.get("comparison"))
    cand = summary.get("candidate")
    if cand:
        s = cand["summary"]
        out += ["", "## Abliteration", "", f"- Label: **{cand['label']}** (candidate `{cand['candidate_id']}`, seed {cand.get('seed')})"]
        out.append(f"- Optimizer metrics ({s.get('provenance')}): refusal rate {s.get('optimizer_refusal_rate')}, KL {s.get('optimizer_kl')}")
        chosen = (cand.get("selection") or {}).get("chosen")
        if chosen:
            how = "by policy `" + cand["selection"]["policy"] + "`" if chosen["by"] == "policy" else f"manually ({chosen['by']})"
            out.append(
                f"- Selected study trial **{chosen['index']}** {how}; Pareto position {chosen['pareto_position']} (informational); "
                f"constraints {cand['selection']['constraints']}"
                + ("; **exported although ineligible** (selection.allow_ineligible_choice)" if chosen.get("ineligible_override") else "")
            )
            out.append(f"- All trials and the ranking: `{cand['selection']['candidates']}`")
        for item in cand.get("limitations", []):
            out.append(f"- Limitation: {item}")
        diff = cand.get("tensor_diff_vs_source") or {}
        if diff.get("missing_count"):
            out.append(f"- Candidate lacks {diff['missing_count']} source tensors: {diff.get('missing_prefixes')}")
    qual = summary.get("qualification")
    if qual:
        m = qual["independent"]
        out += ["", f"## Qualification ({qual['mode']}, independent, held-out)", ""]
        out.append(f"- Benign pass rate: {m['benign_pass_rate']} over {m['benign_graded']} graded prompts")
        out.append(f"- Held-out refusals ({m['scorer']}): {m['heldout_refusal_count']}/{m['heldout_refusal_prompts']}")
        out.append(f"- Integrity: {'passed' if m['integrity_passed'] else 'FAILED'}; reload: {m['reload_ok']}; extended KL: {m.get('extended_kl')}")
        for c in qual.get("caveats", []):
            out.append(f"- Caveat: {c}")
    sel = summary.get("selection")
    if sel:
        out += ["", "## Selection gates", ""]
        if sel.get("outcome"):
            out += [f"- Outcome: **{sel['outcome']}** (reason `{sel['reason']}`, deciding gate `{sel.get('deciding_gate')}`, profile {sel['profile']})", ""]
        out += ["| Gate | Required | Value | Threshold | Status | Provenance |", "|---|---|---|---|---|---|"]
        for g in sel["gates"]:
            status = g.get("status") or g.get("result") or g.get("passed")
            if g.get("override_key"):
                status = f"{status} ({g['override_key']})"
            out.append(f"| {g['gate']} | {g.get('required', True)} | {g['value']} | {g['op']} {g['threshold']} | {status} | {g['provenance']} |")
    pub = summary.get("published_baselines")
    if pub:
        out += ["", "## Published baselines (not measured locally)", ""]
        for ref in pub["references"]:
            out.append(f"- {ref['repo']}@{str(ref['revision'])[:12]} ({ref['source_url']}, retrieved {ref['retrieved_utc']}); evaluator: {ref['evaluator']}")
            for metric in ref["metrics"]:
                denom = f"/{metric['denominator']}" if metric.get("denominator") else ""
                out.append(f"  - {metric['model_role']}: {metric['name']} = {metric['value']}{denom}")
            out.append(f"  - Caveat: {ref['comparability_caveat']}")
        for w in pub.get("comparability_warnings", []):
            out.append(f"- WARNING: {w}")
    t = summary["time"]
    out += ["", "## Time and cost", "", f"- Measured stage time: {t['measured_stage_seconds']} s", f"- Cost: {t['estimated_cost_usd']} USD ({t['cost_note']})"]
    comps = summary.get("components") or {}
    if comps:
        out += ["", "## Tool versions", ""]
        for name, rec in sorted(comps.items()):
            out.append(f"- {name}: {rec.get('versions')}")
    usage = usage_examples(summary["artifacts"])
    if usage:
        out += ["", "## Using the artifacts", ""] + usage
    out += ["", "Project B (deep capability evaluation) consumes the manifests listed above; it is not part of this repository.", ""]
    return "\n".join(out)


def write_report(run: Run, final: bool) -> Path:
    summary = collect(run)
    summary["complete"] = final
    reports = run.run_dir / "reports"
    atomic_write_json(reports / "run-summary.json", summary)
    atomic_write_json(
        reports / "manifest-index.json",
        {role: {k: a[k] for k in ("artifact_id", "parent_artifact_id", "path", "manifest")} for role, a in summary["artifacts"].items()},
    )
    path = reports / "report.md"
    atomic_write_text(path, render_markdown(summary, final))
    return path


class ReportExportStage(Stage):
    id = "10"
    name = "report_export"
    title = "Report and manifests"
    config_keys = ()

    def dependencies(self, config) -> tuple[str, ...]:
        return ("09",)

    def target_outputs(self, config) -> list[str]:
        return ["runs/<run>/reports/report.md", "runs/<run>/reports/run-summary.json", "runs/<run>/reports/manifest-index.json"]

    def extra_fingerprint(self, run) -> dict[str, Any]:
        return {"artifacts": sorted((r, e["artifact_id"]) for r, e in run.registry.roles().items())}

    def execute(self, ctx: StageContext) -> None:
        path = write_report(ctx.run, final=True)
        for name in ("report.md", "run-summary.json", "manifest-index.json"):
            ctx.add_output(path.parent / name, "report")
        ctx.info(f"Report written to {path}")
