"""Stage 06 - qualify: reload the saved candidate and run independent held-out checks.

``fast``: benign prompts with checkable answers + held-out refusal prompts (disjoint from Heretic's
splits), integrity heuristics, reload check. ``extended`` adds held-out KL against the source and
the extended benign tier. Imported optimizer metrics are copied separately and labelled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import ExitCode, PipelineError
from ..evaluation import comparison as C
from ..evaluation.qualification import SCORER_ID, load_benign_prompts, score_results
from ..fsutil import atomic_write_json, read_json, sha256_file, sha256_obj, sha256_text
from .base import Stage, StageContext


def run_generation_script(ctx: StageContext, spec: dict[str, Any], name: str, timeout_minutes: float = 240, work_dir: Path | None = None) -> dict[str, Any]:
    """Run qualify.py; ``work_dir`` (from ``ctx.generation_dir``) receives the spec, raw output and logs."""
    work_dir = work_dir or ctx.log_dir
    record = ctx.run.components(ctx.log_dir).get("heretic")
    spec_path = work_dir / f"{name}-spec.json"
    out_path = work_dir / f"{name}-raw.json"
    atomic_write_json(spec_path, spec)
    result = ctx.run.runner.run(
        [*record.script("qualify.py"), "--spec", str(spec_path), "--out", str(out_path)],
        cwd=None,
        env=ctx.run.tool_env(record),
        log_dir=work_dir,
        name=name,
        timeout_seconds=timeout_minutes * 60,
    )
    if result.interrupted:
        ctx.run.check_interrupt()
    if not out_path.exists():
        raise PipelineError(
            ExitCode.GATE if result.ok else ExitCode.ENGINE,
            "E_QUALIFY_RUNTIME",
            f"Qualification runtime produced no result (exit {result.returncode})",
            hint=f"{result.stderr_log}: {' | '.join(result.tail[-5:])}" if work_dir == ctx.log_dir else "tool logs were discarded (logging.redact_prompts)",
        )
    raw = read_json(out_path)
    raw["_runtime"] = {"returncode": result.returncode, "duration_s": result.duration_s, "versions": record.versions}
    return raw


class QualifyStage(Stage):
    id = "06"
    name = "qualify"
    title = "Independent qualification"
    gpu = True
    config_keys = (
        "evaluation.qualification",
        "evaluation.independent_smoke_prompts",
        "evaluation.benign_prompts_file",
        "evaluation.heldout_refusal_prompts",
        "evaluation.heldout_harmless_prompts",
        "evaluation.extended_kl",
        "evaluation.extended_correctness",
        "evaluation.enable_thinking",
        "evaluation.max_new_tokens",
        "evaluation.seed",
        "evaluation.refusal_markers",
        "logging.redact_prompts",
        "model.trust_remote_code",
    )

    def dependencies(self, config) -> tuple[str, ...]:
        return ("05",)

    def applicable(self, config) -> tuple[bool, str]:
        if not config.get("abliteration.enabled"):
            return False, "abliteration disabled (quantize-only)"
        if config.get("evaluation.qualification") == "off":
            return False, "evaluation.qualification=off (master will be labelled unqualified)"
        return True, ""

    def extra_fingerprint(self, run) -> dict[str, Any]:
        path = run.config.get("evaluation.benign_prompts_file")
        return {"benign_sha256": sha256_file(path) if path and Path(path).is_file() else None}

    def target_outputs(self, config) -> list[str]:
        return ["runs/<run>/metrics/qualification.json"]

    def execute(self, ctx: StageContext) -> None:
        run, config = ctx.run, ctx.config
        candidate = read_json((run.load_receipt("05") or {})["diagnostics"]["candidate"])
        mode = config.get("evaluation.qualification")
        extended = mode == "extended"
        benign = [p for p in load_benign_prompts(config.get("evaluation.benign_prompts_file"))]
        benign = [p for p in benign if extended or config.get("evaluation.extended_correctness") or p.get("category") != "extended"]
        total = int(config.get("evaluation.independent_smoke_prompts"))
        benign = benign[:total] if not extended else benign
        heldout_limit = None if extended else max(0, total - len(benign))
        source_uri = run.registry.get("source")["manifest_data"]["artifact"]["uri"]
        spec = {
            "model_path": candidate["path"],
            "source_path": source_uri,
            "benign": benign,
            "heldout_refusal_specs": config.get("evaluation.heldout_refusal_prompts"),
            "heldout_limit": heldout_limit,
            "extended_kl": bool(extended or config.get("evaluation.extended_kl")),
            "heldout_harmless_specs": config.get("evaluation.heldout_harmless_prompts"),
            "max_new_tokens": config.get("evaluation.max_new_tokens"),
            "enable_thinking": config.get("evaluation.enable_thinking"),
            "seed": config.get("evaluation.seed"),
            "trust_remote_code": config.get("model.trust_remote_code"),
            "hf_cache": str(run.cache_root / "hf" / "hub"),
        }
        with ctx.generation_dir("qualify") as work:
            raw = run_generation_script(ctx, spec, "qualify", work_dir=work)
        scored = score_results(raw, [m.lower() for m in config.get("evaluation.refusal_markers")])
        redact = ctx.redact
        records = []
        for row in raw.get("benign", []) + raw.get("heldout_refusal", []):
            item = {k: row.get(k) for k in ("id", "kind", "finish_reason", "new_tokens")}
            if redact:
                prompt, response = row.get("prompt", ""), row.get("response", "")
                item.update(prompt_sha256=sha256_text(prompt), prompt_chars=len(prompt), response_sha256=sha256_text(response), response_chars=len(response))
            else:
                item.update(prompt=row.get("prompt"), response=row.get("response"))
            records.append(item)
        markers = [m.lower() for m in config.get("evaluation.refusal_markers")]
        suite = {
            "id": f"{mode}-qualification",
            "version": SCORER_ID,
            "sha256": sha256_obj(
                {
                    "benign": [[p["id"], sha256_text(p["prompt"]), p["expect_any"]] for p in benign],
                    "heldout": [spec["heldout_refusal_specs"], heldout_limit],
                    "extended_kl": [spec["extended_kl"], spec["heldout_harmless_specs"] if spec["extended_kl"] else None],
                    "markers": markers,
                }
            ),
        }
        result = {
            "mode": mode,
            "suite": suite,
            "generation_contract": C.contract(spec, raw, bool(redact)),
            "candidate": candidate["candidate_id"],
            "imported_optimizer_metrics": candidate["summary"],
            "independent": scored["metrics"],
            "per_prompt": {"benign": scored["benign"], "heldout": scored["heldout"]},
            "generation_settings": {
                "max_new_tokens": spec["max_new_tokens"],
                "enable_thinking": spec["enable_thinking"],
                "do_sample": False,
                "seed": spec["seed"],
                "chat_template": raw.get("chat_template_sha256"),
            },
            "datasets": raw.get("datasets"),
            "load": {
                "ok": raw.get("load_ok"),
                "seconds": raw.get("load_seconds"),
                "dtype": raw.get("dtype"),
                "peak_gpu_memory_bytes": raw.get("peak_gpu_memory_bytes"),
            },
            "max_total_tokens": raw.get("max_total_tokens"),
            "runtime_versions": raw["_runtime"]["versions"],
            "records": records,
            "redacted": bool(redact),
            "caveats": [
                "Fast qualification is a smoke test (tens of prompts); it is not a statistically meaningful evaluation.",
                "Refusals are detected with a keyword heuristic on held-out prompts; it can miss soft refusals and flag benign phrasing.",
                "Optimizer metrics are Heretic's in-loop scores on its own evaluation prompts and are reported separately.",
                "Deep capability evaluation is out of scope (Project B).",
            ],
        }
        out = ctx.metrics_path("qualification.json")
        atomic_write_json(out, result)
        ctx.add_output(out, "qualification")
        metrics = scored["metrics"]
        ctx.diagnostics.update(qualification=str(out), independent=metrics)
        ctx.info(
            f"Qualification: benign pass {metrics['benign_pass_rate']}, held-out refusals "
            f"{metrics['heldout_refusal_count']}/{metrics['heldout_refusal_prompts']}, integrity {'ok' if metrics['integrity_passed'] else 'FAILED'}"
        )
