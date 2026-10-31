"""Stage 09 - verify_exports: re-hash every exported file, check tokenizer/config/processor
preservation, read GGUF metadata, and run real loader smoke tests where a backend is configured.

``inference_passed`` stays ``null`` when no loader test ran; modalities are only listed after a
real test of that modality. Results are assessment records (verification, and comparison of each
export with its baseline); the manifests derive their status from them (docs/assessments.md).
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

from ..adapters import llama_cpp as LC
from ..artifacts import assessments as A
from ..artifacts.gguf_meta import read_metadata
from ..artifacts.manifest import PROCESSOR_FILES, TOKENIZER_FILES
from ..environment.capabilities import native_fp8
from ..errors import ExitCode, Interrupted, PipelineError
from ..evaluation import comparison as C
from ..evaluation.qualification import integrity_issues
from ..fsutil import atomic_write_bytes, atomic_write_json, file_inventory, read_json, sha256_file, sha256_obj, sha256_text
from .base import Stage, StageContext
from .stage06_qualify import run_generation_script
from .stage08_quantize import input_role, is_projector

PRESERVED = TOKENIZER_FILES + PROCESSOR_FILES + ("generation_config.json",)


def tiny_png(path: Path, rgb: tuple[int, int, int] = (220, 20, 20), size: int = 64) -> None:
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    atomic_write_bytes(path, png)


# transformers 5 saves image/video processor settings inside processor_config.json (verified with
# 5.18.0 on Qwen/Qwen3.5-2B, docs/implementation-log.md): source file -> key in processor_config.json.
MERGED_PROCESSOR_KEYS = {
    "preprocessor_config.json": "image_processor",
    "image_processor_config.json": "image_processor",
    "video_preprocessor_config.json": "video_processor",
}
# Keys that move or are renamed by that merge, not settings: processor_class moves to the top level,
# and the "Fast" processor class names lose their suffix.
MERGE_BOOKKEEPING_KEYS = {"processor_class", "image_processor_type", "video_processor_type"}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _reserialized_basis(name: str, artifact_dir: Path, source_dir: Path) -> str | None:
    """Why a source file absent from the artifact is still represented in it, or None if it is not."""
    if name in MERGED_PROCESSOR_KEYS:
        merged = (_load_json(artifact_dir / "processor_config.json") or {}).get(MERGED_PROCESSOR_KEYS[name])
        source = _load_json(source_dir / name)
        if isinstance(merged, dict) and isinstance(source, dict):
            settings = {k: v for k, v in source.items() if k not in MERGE_BOOKKEEPING_KEYS}
            if all(merged.get(k) == v for k, v in settings.items()):
                return f"every setting of {name} is in processor_config.json['{MERGED_PROCESSOR_KEYS[name]}'] with the same value"
        return None
    if name in ("vocab.json", "merges.txt"):
        model = (_load_json(artifact_dir / "tokenizer.json") or {}).get("model") or {}
        if name == "vocab.json":
            same = model.get("vocab") is not None and model.get("vocab") == _load_json(source_dir / name)
        else:
            lines = (source_dir / name).read_text(encoding="utf-8").splitlines()
            merges = [m for m in lines if m and not m.startswith("#version")]
            embedded = [" ".join(m) if isinstance(m, list) else m for m in model.get("merges") or []]
            same = bool(merges) and merges == embedded
        return f"identical content embedded in tokenizer.json model.{name.split('.')[0]}" if same else None
    return None


def _tokenizer_differences(artifact_dir: Path, source_dir: Path) -> str:
    src, dst = _load_json(source_dir / "tokenizer.json") or {}, _load_json(artifact_dir / "tokenizer.json") or {}
    sm, dm = src.get("model") or {}, dst.get("model") or {}
    merges = [" ".join(m) if isinstance(m, list) else m for m in sm.get("merges") or []] == [
        " ".join(m) if isinstance(m, list) else m for m in dm.get("merges") or []
    ]
    parts = [f"vocab {'identical' if sm.get('vocab') == dm.get('vocab') else 'DIFFERS'}", f"merges {'identical' if merges else 'DIFFER'}"]
    added = {t.get("content") for t in dst.get("added_tokens") or []} - {t.get("content") for t in src.get("added_tokens") or []}
    removed = {t.get("content") for t in src.get("added_tokens") or []} - {t.get("content") for t in dst.get("added_tokens") or []}
    if added or removed:
        parts.append(f"added tokens +{sorted(added)[:8]} -{sorted(removed)[:8]}")
    changed = sorted(k for k in set(src) | set(dst) if k not in ("model", "added_tokens") and src.get(k) != dst.get(k))
    if changed:
        parts.append(f"changed sections: {', '.join(changed)}")
    return "; ".join(parts)


def compare_files(artifact_dir: Path, source_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Per preserved file: identical, json-equivalent, re-serialized, differs or missing; plus the basis."""
    result, basis = {}, {}
    for name in PRESERVED:
        src = source_dir / name
        if not src.is_file():
            continue
        dst = artifact_dir / name
        if not dst.is_file():
            reason = _reserialized_basis(name, artifact_dir, source_dir)
            result[name] = "re-serialized" if reason else "missing"
            if reason:
                basis[name] = reason
        elif sha256_file(src) == sha256_file(dst):
            result[name] = "identical"
        elif name.endswith(".json"):
            same = _load_json(src) == _load_json(dst) and _load_json(src) is not None
            result[name] = "json-equivalent" if same else "differs"
            if name == "tokenizer.json" and not same:
                basis[name] = _tokenizer_differences(artifact_dir, source_dir)
        else:
            result[name] = "differs"
    return result, basis


class VerifyExportsStage(Stage):
    id = "09"
    name = "verify_exports"
    title = "Verify exports"
    gpu = True
    config_keys = (
        "quantization.input",
        "quantization.smoke_prompts",
        "quantization.smoke_max_tokens",
        "quantization.fp8.verify_backend",
        "quantization.fp8.vllm_max_model_len",
        "quantization.fp8.vllm_gpu_memory_utilization",
        "quantization.gguf.smoke_context",
        "quantization.gguf.smoke_gpu_layers",
        "execution.allow_partial_exports",
        "evaluation.comparison",
        "evaluation.benign_prompts_file",
        "evaluation.heldout_refusal_prompts",
        "evaluation.enable_thinking",
        "evaluation.refusal_markers",
        "selection.export_gates",
        "logging.redact_prompts",
        "model.trust_remote_code",
    )

    def components_used(self, config) -> set[str]:
        from ..planning import components_for_config  # noqa: PLC0415

        return components_for_config(config) - {"fp8"}

    def dependencies(self, config) -> tuple[str, ...]:
        deps = ["01", "02"]
        if config.get("abliteration.enabled"):
            deps.append("07")
        if config.get("quantization.enabled") and config.get("quantization.formats"):
            deps.append("08")
        return tuple(deps)

    def extra_fingerprint(self, run) -> dict[str, Any]:
        path = run.config.get("evaluation.benign_prompts_file")
        return {"benign_sha256": sha256_file(path) if path and Path(path).is_file() else None}

    def target_outputs(self, config) -> list[str]:
        return [
            "runs/<run>/metrics/verification.json",
            "runs/<run>/metrics/comparison.json (with exports and evaluation.comparison.enabled)",
            "workspace/assessments/<artifact_id>/<assessment_id>.json (verification and comparison records)",
        ]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _texts(texts: list[str], redact: bool) -> dict[str, Any]:
        if redact:
            return {"samples_sha256": [sha256_text(t) for t in texts], "samples_chars": [len(t) for t in texts]}
        return {"samples": texts}

    @staticmethod
    def _runtime(ctx: StageContext, backend: str | None, version: str | None, versions: dict[str, Any] | None) -> dict[str, Any]:
        gpus = [g.get("name") for g in ctx.run.probe.gpus() or []]
        return {"backend": backend, "version": version, "environment_fingerprint": A.environment_fingerprint(versions, gpus)}

    # ------------------------------------------------------------------ smoke tests
    def _smoke_master(self, ctx: StageContext, path: str) -> dict[str, Any]:
        qual = ctx.run.load_receipt("06")
        if qual and qual.get("outcome") == "succeeded":
            data = read_json(qual["diagnostics"]["qualification"])
            ok = bool(data["independent"]["reload_ok"]) and bool(data["independent"]["integrity_passed"])
            return {
                "ran": True,
                "reference": "stage 06 qualification reload",
                "passed": ok,
                "backend": "transformers",
                "backend_revision": f"transformers {(data.get('runtime_versions') or {}).get('transformers')}",
                "versions": data.get("runtime_versions") or {},
                "tested_context": data.get("max_total_tokens"),
                "modalities": ["text"] if ok else [],
            }
        if not ctx.config.get("abliteration.enabled"):
            return {"ran": False}
        prompts = [{"id": f"smoke-{i}", "prompt": p, "expect_any": []} for i, p in enumerate(ctx.config.get("quantization.smoke_prompts"))]
        with ctx.generation_dir("master-smoke") as work:
            raw = run_generation_script(
                ctx,
                {
                    "model_path": path,
                    "benign": prompts,
                    "heldout_refusal_specs": [],
                    "heldout_limit": 0,
                    "extended_kl": False,
                    "max_new_tokens": ctx.config.get("quantization.smoke_max_tokens"),
                    "enable_thinking": False,
                    "seed": 0,
                    "trust_remote_code": ctx.config.get("model.trust_remote_code"),
                    "hf_cache": str(ctx.run.cache_root / "hf" / "hub"),
                },
                "master-smoke",
                work_dir=work,
            )
        outputs = [r.get("response", "") for r in raw.get("benign", [])]
        ok = bool(raw.get("load_ok")) and bool(outputs) and not any(integrity_issues(o, include_thinking=True) for o in outputs)
        return {
            "ran": True,
            "passed": ok,
            "backend": "transformers",
            "backend_revision": f"transformers {raw['_runtime']['versions'].get('transformers')}",
            "versions": raw["_runtime"]["versions"],
            "tested_context": raw.get("max_total_tokens"),
            "modalities": ["text"] if ok else [],
            **self._texts(outputs[:3], ctx.redact),
        }

    def _smoke_fp8(self, ctx: StageContext, path: str, suite: list[dict[str, Any]] | None) -> dict[str, Any]:
        """vLLM loader smoke test; with ``suite`` the comparison prompts run in the same load."""
        config = ctx.config
        if config.get("quantization.fp8.verify_backend") != "vllm":
            return {"ran": False, "reason": "verify_backend=none"}
        record = ctx.run.components(ctx.log_dir).get("vllm")
        with ctx.generation_dir("vllm-smoke") as work:
            out = work / "vllm-smoke.json"
            out.unlink(missing_ok=True)
            prompts_file = work / "smoke-prompts.json"
            atomic_write_json(prompts_file, config.get("quantization.smoke_prompts"))
            argv = [
                *record.script("vllm_smoke.py"),
                "--model",
                path,
                "--max-model-len",
                str(config.get("quantization.fp8.vllm_max_model_len")),
                "--gpu-memory-utilization",
                str(config.get("quantization.fp8.vllm_gpu_memory_utilization")),
                "--prompts",
                str(prompts_file),
                "--max-tokens",
                str(config.get("quantization.smoke_max_tokens")),
                "--out",
                str(out),
            ]
            if suite:
                suite_file = work / "comparison-suite.json"
                atomic_write_json(suite_file, [{"id": s["id"], "rendered": s["rendered"]} for s in suite])
                argv += ["--suite", str(suite_file), "--suite-max-tokens", str(config.get("evaluation.comparison.max_new_tokens"))]
            result = ctx.run.runner.run(argv, cwd=None, env=ctx.run.tool_env(record), log_dir=work, name="vllm-smoke", timeout_seconds=3600)
            if result.interrupted:
                raise Interrupted()
            data = read_json(out) if out.exists() else {"ok": False, "error": f"no result (exit {result.returncode})"}
        outputs = data.get("outputs") or []
        ok = bool(data.get("ok")) and bool(outputs) and not any(integrity_issues(o, include_thinking=True) for o in outputs)
        native = native_fp8(ctx.run.probe.gpus() or [])
        kinds = {s["id"]: s["kind"] for s in suite or []}
        suite_rows = data.get("suite")
        return {
            "ran": True,
            "passed": ok,
            # Without native FP8 vLLM runs a weight-only fallback (capabilities.VLLM_FP8_BASIS): the result
            # says nothing about the export on FP8 hardware, so it neither passes nor fails it.
            "representative": native is not False,
            "native_fp8": native,
            "backend": "vllm",
            "backend_revision": f"vllm {record.versions.get('vllm')}",
            "versions": record.versions,
            "tested_context": config.get("quantization.fp8.vllm_max_model_len") if ok else None,
            "modalities": ["text"] if ok else [],
            **self._texts(outputs[:3], ctx.redact),
            "error": data.get("error"),
            "load_seconds": data.get("load_seconds"),
            "suite_rows": [{**r, "kind": kinds.get(r["id"], "benign")} for r in suite_rows] if suite and suite_rows is not None else None,
        }

    def _gguf_layers(self, ctx: StageContext, record) -> str:
        layers = ctx.config.get("quantization.gguf.smoke_gpu_layers")
        return ("all" if record.build.get("cuda") else "0") if layers == "auto" else layers

    def _smoke_gguf(self, ctx: StageContext, art_dir: Path) -> dict[str, Any]:
        config = ctx.config
        record = ctx.run.components(ctx.log_dir).get("llama_cpp")
        ggufs = sorted(art_dir.glob("*.gguf"))
        mmproj = next((p for p in ggufs if is_projector(p)), None)
        model = next((p for p in ggufs if not is_projector(p)), None)
        if model is None:
            return {"ran": False, "reason": "no model gguf"}
        layers = self._gguf_layers(ctx, record)
        ctxlen = config.get("quantization.gguf.smoke_context")
        timeout = float(config.get("quantization.gguf.smoke_timeout_minutes")) * 60
        samples, ok = [], True
        with ctx.generation_dir(f"gguf-smoke-{art_dir.name}") as work:
            for i, prompt in enumerate(config.get("quantization.smoke_prompts")):
                res = ctx.run.runner.run(
                    LC.completion_argv(record.cmd("llama-completion"), model, prompt, config.get("quantization.smoke_max_tokens"), ctxlen, layers),
                    cwd=None,
                    env=ctx.run.tool_env(record),
                    log_dir=work,
                    name=f"gguf-smoke-{art_dir.name}-{i}",
                    timeout_seconds=timeout,
                )
                if res.interrupted:
                    raise Interrupted()
                text = Path(res.stdout_log).read_text(encoding="utf-8", errors="replace").split("\n", 1)[-1]
                generated = text.split(prompt, 1)[-1].strip() if prompt in text else text.strip()
                samples.append(generated[:500])
                if not res.ok or not generated or integrity_issues(generated, include_thinking=True):
                    ok = False
            result = {
                "ran": True,
                "passed": ok,
                "backend": "llama.cpp",
                "backend_revision": f"llama.cpp {record.versions.get('commit')}",
                "versions": record.versions,
                "tested_context": ctxlen if ok else None,
                "gpu_layers": layers,
                "modalities": ["text"] if ok else [],
                **self._texts(samples, ctx.redact),
            }
            if mmproj is not None:
                image = work / "smoke-red-square.png"
                tiny_png(image)
                res = ctx.run.runner.run(
                    LC.mtmd_argv(record.cmd("llama-mtmd-cli"), model, mmproj, image, "What color is this image? Answer in one word.", 16, ctxlen, layers),
                    cwd=None,
                    env=ctx.run.tool_env(record),
                    log_dir=work,
                    name=f"gguf-mtmd-{art_dir.name}",
                    timeout_seconds=timeout,
                )
                out = Path(res.stdout_log).read_text(encoding="utf-8", errors="replace")
                image_ok = res.ok and bool(out.strip().splitlines()[1:] if out.strip() else False)
                result["image_test"] = {"passed": image_ok, "mentions_red": "red" in out.lower(), "returncode": res.returncode}
                if image_ok and ok:
                    result["modalities"].append("image")
        return result

    def _suite_gguf(self, ctx: StageContext, art_dir: Path, suite: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """The comparison prompts through llama-completion as rendered by the baseline (one process per prompt).

        Returns None when the model did not load or a generation failed."""
        config = ctx.config
        record = ctx.run.components(ctx.log_dir).get("llama_cpp")
        model = next((p for p in sorted(art_dir.glob("*.gguf")) if not is_projector(p)), None)
        if model is None:
            return None
        layers = self._gguf_layers(ctx, record)
        timeout = float(config.get("quantization.gguf.smoke_timeout_minutes")) * 60
        rows = []
        with ctx.generation_dir(f"gguf-comparison-{art_dir.name}") as work:
            for i, item in enumerate(suite):
                prompt_file = work / f"comparison-{i}.txt"
                atomic_write_bytes(prompt_file, (item["rendered"] + "\n").encode("utf-8"))  # -f drops one trailing newline
                res = ctx.run.runner.run(
                    LC.rendered_completion_argv(
                        record.cmd("llama-completion"),
                        model,
                        prompt_file,
                        config.get("evaluation.comparison.max_new_tokens"),
                        config.get("quantization.gguf.smoke_context"),
                        layers,
                    ),
                    cwd=None,
                    env=ctx.run.tool_env(record),
                    log_dir=work,
                    name=f"gguf-comparison-{art_dir.name}-{i}",
                    timeout_seconds=timeout,
                )
                if res.interrupted:
                    raise Interrupted()
                if not res.ok:
                    return None
                stdout = Path(res.stdout_log).read_text(encoding="utf-8", errors="replace").split("\n", 1)[-1]  # first line: runner header
                response, finish = LC.generated_text(stdout)
                rows.append({"id": item["id"], "kind": item["kind"], "response": response, "finish_reason": finish})
        return rows

    # ------------------------------------------------------------------ comparison
    def _baseline(self, ctx: StageContext, spec: dict[str, Any]) -> dict[str, Any]:
        """Run the comparison suite with transformers on the artifact the exports were made from."""
        run, config = ctx.run, ctx.config
        role = input_role(config)
        entry = run.registry.get(role)
        if entry is None:
            return {"role": role, "entry": None, "raw": None, "scored": C.score(None, spec, False, ctx.redact, f"no {role} artifact registered")}
        try:
            with ctx.generation_dir("comparison-baseline") as work:
                raw = run_generation_script(
                    ctx,
                    {
                        "purpose": "comparison-baseline",
                        "model_path": entry["manifest_data"]["artifact"]["uri"],
                        "benign": spec["benign"],
                        "heldout_refusal_specs": spec["heldout_refusal_specs"],
                        "heldout_limit": spec["heldout_limit"],
                        "extended_kl": False,
                        "max_new_tokens": spec["max_new_tokens"],
                        "enable_thinking": spec["enable_thinking"],
                        "seed": 0,
                        "trust_remote_code": config.get("model.trust_remote_code"),
                        "hf_cache": str(run.cache_root / "hf" / "hub"),
                    },
                    "comparison-baseline",
                    work_dir=work,
                )
        except PipelineError as exc:
            if exc.error_id != "E_QUALIFY_RUNTIME":
                raise
            ctx.warn(f"Comparison baseline ({role}) produced no result; every export comparison is unmeasured")
            return {"role": role, "entry": entry, "raw": None, "scored": C.score(None, spec, False, ctx.redact, "baseline runtime produced no result")}
        rows = raw.get("benign", []) + raw.get("heldout_refusal", [])
        loaded = bool(raw.get("load_ok"))
        if not loaded:
            ctx.warn(f"Comparison baseline ({role}) did not load; every export comparison is unmeasured")
        return {"role": role, "entry": entry, "raw": raw, "scored": C.score(rows, spec, loaded, ctx.redact)}

    def _baseline_record(self, ctx: StageContext, spec: dict[str, Any], base: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any] | None:
        if base["entry"] is None:
            return None
        scored = base["scored"]
        versions = (base["raw"] or {}).get("_runtime", {}).get("versions")
        if scored.get("measured"):
            issues = scored["integrity_issues"]["count"]
            checks = [A.check("baseline_loaded", "passed", True), A.check("baseline_integrity", "failed" if issues else "passed", issues, 0)]
        else:
            checks = [A.check("baseline_loaded", "failed" if scored.get("reason") == "did not load" else "unmeasured", False, reason=scored.get("reason"))]
        record = A.new_record(
            artifact_id=base["entry"]["artifact_id"],
            role=base["role"],
            run_id=ctx.run.run_id,
            stage_id=self.id,
            kind="comparison",
            suite=C.suite_ref(spec),
            runtime=self._runtime(ctx, "transformers", (versions or {}).get("transformers"), versions),
            contract=contract,
            gaps=C.backend_gaps("transformers", None, None),
            redacted=ctx.redact,
            results={"side": "baseline", "baseline_role": base["role"], **scored},
            checks=checks,
        )
        ctx.run.registry.attach(base["role"], record)
        return record

    def _export_record(
        self, ctx: StageContext, role: str, artifact_id: str, cmp: dict[str, Any], smoke: dict[str, Any], scored: dict[str, Any]
    ) -> dict[str, Any]:
        base, contract = cmp["base"], cmp["contract"]
        backend = smoke.get("backend") or ("vllm" if role == "export:fp8" else "llama.cpp")
        limit = ctx.config.get("quantization.fp8.vllm_max_model_len") if backend == "vllm" else ctx.config.get("quantization.gguf.smoke_context")
        gaps = C.backend_gaps(backend, limit, contract["context"]["required_tokens"])
        if smoke.get("representative") is False:
            gaps.append(
                {
                    "field": "runtime",
                    "detail": "weight-only FP8 fallback (no native FP8 on this GPU): outputs reflect the FP8 weights, not native FP8 execution",
                    "basis": "environment.capabilities.VLLM_FP8_BASIS",
                }
            )
        result = C.compare(base["scored"], scored, ctx.config.get("selection.export_gates"))
        record = A.new_record(
            artifact_id=artifact_id,
            role=role,
            run_id=ctx.run.run_id,
            stage_id=self.id,
            kind="comparison",
            suite=C.suite_ref(cmp["spec"]),
            runtime=self._runtime(ctx, backend, smoke.get("backend_revision"), smoke.get("versions")),
            contract=contract,
            gaps=gaps,
            redacted=ctx.redact,
            results={
                "side": "export",
                "description": f"exported runtime ({backend}, {role.split(':')[-1]}) vs baseline runtime (transformers, {base['role']})",
                "baseline": {
                    "role": base["role"],
                    "artifact_id": (base["entry"] or {}).get("artifact_id"),
                    "assessment_id": cmp["record"]["assessment_id"] if cmp["record"] else None,
                },
                **scored,
                "deltas": result["deltas"],
                "new_integrity_issues": result.get("new_integrity_issues"),
                "unmeasured_reason": result.get("unmeasured_reason"),
            },
            checks=result["checks"],
            outcome=result["outcome"],
        )
        ctx.run.registry.attach(role, record)
        return record

    def _verification_record(
        self, ctx: StageContext, role: str, artifact_id: str, item: dict[str, Any], smoke: dict[str, Any], problems: list[str], unrepresentative: bool
    ) -> dict[str, Any]:
        checks = [A.check("integrity", "failed" if problems else "passed", problems or None, "files match the manifest")]
        inference = evaluation = None
        if not smoke.get("ran"):
            checks.append(A.check("loader", "unmeasured", reason=smoke.get("reason", "not configured")))
        elif unrepresentative:
            checks.append(A.check("loader", "unmeasured", smoke["passed"], reason="weight-only FP8 fallback; not representative of native FP8"))
        else:
            checks.append(A.check("loader", "passed" if smoke["passed"] else "failed", smoke["passed"]))
            inference = {
                "inference_passed": bool(smoke["passed"]),
                "modalities_tested": smoke.get("modalities", []),
                "backend": smoke.get("backend"),
                "backend_revision": smoke.get("backend_revision"),
                "tested_context": smoke.get("tested_context") or None,
            }
            if role.startswith("export:"):
                evaluation = {"qualification_run_id": None, "metrics_provenance": "integrity-only"}
        if "image_test" in smoke:
            checks.append(A.check("image", "passed" if smoke["image_test"]["passed"] else "failed", smoke["image_test"]["passed"]))
        ran_here = bool(smoke.get("ran")) and not smoke.get("reference")
        prompts = ctx.config.get("quantization.smoke_prompts")
        contract = None
        if ran_here:
            template = {"vllm": "vLLM chat (the model's template)", "llama.cpp": "none: raw prompt (-no-cnv)"}.get(
                smoke.get("backend"), "transformers apply_chat_template"
            )
            contract = {
                "prompts": [{"id": f"smoke-{i}", **C.text_fields("prompt", p, ctx.redact)} for i, p in enumerate(prompts)],
                "chat_template": template,
                "max_new_tokens": ctx.config.get("quantization.smoke_max_tokens"),
                "decoding": {"strategy": "greedy", "temperature": 0, "seed": 0},
            }
        record = A.new_record(
            artifact_id=artifact_id,
            role=role,
            run_id=ctx.run.run_id,
            stage_id=self.id,
            kind="verification",
            suite={"id": "loader-smoke", "version": "1", "sha256": sha256_obj([prompts, ctx.config.get("quantization.smoke_max_tokens")])}
            if ran_here
            else None,
            runtime=self._runtime(ctx, smoke.get("backend"), smoke.get("backend_revision"), smoke.get("versions")) if smoke.get("ran") else None,
            contract=contract,
            redacted=ctx.redact,
            results={k: v for k, v in item.items() if k not in ("artifact_id", "path")},
            checks=checks,
            derives={"validation": {"integrity_passed": not problems}, "inference": inference, "evaluation": evaluation},
        )
        ctx.run.registry.attach(role, record)
        return record

    # ------------------------------------------------------------------ execute
    def _check_files(self, ctx: StageContext, role: str, manifest: dict[str, Any], source_dir: Path, item: dict[str, Any]) -> list[str]:
        config = ctx.config
        art_dir = Path(manifest["artifact"]["uri"])
        expected = {f["relative_path"]: f for f in manifest["artifact"]["files"]}
        problems = []
        if config.get("storage.rehash_on_verify"):
            actual = {e.relative_path: e for e in file_inventory(art_dir)}
            for rel, f in expected.items():
                got = actual.get(rel)
                if got is None:
                    problems.append(f"missing {rel}")
                elif got.sha256 != f["sha256"] or got.size_bytes != f["size_bytes"]:
                    problems.append(f"hash/size mismatch {rel}")
            extra = sorted(set(actual) - set(expected))
            if extra:
                problems.append(f"unexpected files {extra[:5]}")
            item["rehash"] = "full"
        else:
            for rel, f in expected.items():
                p = art_dir / rel
                if not p.is_file() or p.stat().st_size != f["size_bytes"]:
                    problems.append(f"missing/size {rel}")
            item["rehash"] = "sizes-only"
        if manifest["artifact"]["format"] == "safetensors":
            item["preservation"], item["preservation_basis"] = compare_files(art_dir, source_dir)
            bad = {k: v for k, v in item["preservation"].items() if v in ("missing", "differs") and not (k == "config.json")}
            if bad:
                detail = f" (tokenizer.json: {item['preservation_basis']['tokenizer.json']})" if "tokenizer.json" in item["preservation_basis"] else ""
                ctx.warn(f"{role}: tokenizer/processor files not preserved verbatim: {bad}{detail}")
        else:
            gguf_checks = {}
            for f in expected:
                meta = read_metadata(art_dir / f)["kv"]
                gguf_checks[f] = {
                    "architecture": meta.get("general.architecture"),
                    "has_tokenizer": "tokenizer.ggml.model" in meta,
                    "has_chat_template": "tokenizer.chat_template" in meta,
                }
                projector = meta.get("general.architecture") == "clip" or meta.get("general.type") == "mmproj"
                gguf_checks[f]["role"] = "projector" if projector else "model"
                if not projector and (not meta.get("general.architecture") or "tokenizer.ggml.model" not in meta):
                    problems.append(f"{f}: missing architecture/tokenizer metadata")
            item["gguf_metadata"] = gguf_checks
        return problems

    def execute(self, ctx: StageContext) -> None:
        run, config = ctx.run, ctx.config
        roles = {r: e for r, e in run.registry.roles().items() if r == "master" or r.startswith("export:")}
        if not roles:
            raise PipelineError(ExitCode.VERIFY, "E_NOTHING_TO_VERIFY", "No master or exports are registered for this run")
        source_dir = Path(run.registry.get("source")["manifest_data"]["artifact"]["uri"])
        exports = sorted(r for r in roles if r.startswith("export:"))
        cmp: dict[str, Any] | None = None
        if config.get("evaluation.comparison.enabled") and exports:
            spec = C.suite_spec(config)
            base = self._baseline(ctx, spec)
            contract = C.contract(spec, base["raw"], ctx.redact)
            cmp = {"spec": spec, "base": base, "contract": contract, "record": self._baseline_record(ctx, spec, base, contract)}
            cmp["suite"] = C.rendered_prompts(base["raw"] or {}) if base["scored"].get("measured") else None
        report: dict[str, Any] = {}
        compared: dict[str, Any] = {}
        integrity_failures, inference_failures, comparison_failures = [], [], []
        for role in sorted(roles):
            run.check_interrupt()
            manifest = run.registry.get(role)["manifest_data"]
            art_dir = Path(manifest["artifact"]["uri"])
            item: dict[str, Any] = {"artifact_id": manifest["artifact_id"], "path": str(art_dir)}
            problems = self._check_files(ctx, role, manifest, source_dir, item)
            suite = cmp["suite"] if cmp else None
            suite_rows: list[dict[str, Any]] | None = None
            if role == "master":
                smoke = self._smoke_master(ctx, str(art_dir))
            elif role == "export:fp8":
                smoke = self._smoke_fp8(ctx, str(art_dir), suite)
                suite_rows = smoke.pop("suite_rows", None)
            else:
                smoke = self._smoke_gguf(ctx, art_dir)
                if suite:
                    suite_rows = self._suite_gguf(ctx, art_dir, suite)
            item["integrity_problems"] = problems
            item["smoke"] = {k: v for k, v in smoke.items() if k != "versions"}
            report[role] = item
            unrepresentative = bool(smoke.get("ran") and smoke.get("representative") is False)
            if unrepresentative:
                ctx.warn(
                    f"{role}: loader test ran on the weight-only FP8 fallback ({'passed' if smoke['passed'] else 'failed'}); "
                    "not representative of native FP8, so inference_passed stays null (unverified_execution)"
                )
            item["assessment_id"] = self._verification_record(ctx, role, manifest["artifact_id"], item, smoke, problems, unrepresentative)["assessment_id"]
            if problems:
                integrity_failures.append(role)
                ctx.warn(f"{role}: integrity problems {problems[:3]}")
            if smoke.get("ran") and not smoke["passed"] and not unrepresentative:
                inference_failures.append(role)
                ctx.warn(f"{role}: loader smoke test FAILED")
            elif not smoke.get("ran"):
                ctx.warn(f"{role}: no loader smoke test ran ({smoke.get('reason', 'not configured')}); inference_passed stays null")
            if cmp is None or role not in exports:
                continue
            if not suite:
                scored = C.score(None, cmp["spec"], False, ctx.redact, "no baseline measurement")
            elif not smoke.get("ran") and role == "export:fp8":
                scored = C.score(None, cmp["spec"], False, ctx.redact, f"export runtime did not run ({smoke.get('reason')})")
            else:
                scored = C.score(suite_rows or [], cmp["spec"], suite_rows is not None, ctx.redact)
            record = self._export_record(ctx, role, manifest["artifact_id"], cmp, smoke, scored)
            compared[role] = {
                "artifact_id": manifest["artifact_id"],
                "assessment_id": record["assessment_id"],
                "backend": record["runtime"]["backend"],
                **{
                    k: record["results"].get(k) for k in ("description", "benign_correct", "heldout_refused", "integrity_issues", "deltas", "unmeasured_reason")
                },
                "contract_gaps": record["contract_gaps"],
                "checks": record["checks"],
                "outcome": record["outcome"],
            }
            if record["outcome"] != "passed":
                mode = config.get("selection.export_gates.mode")
                ctx.warn(f"{role}: comparison with the {cmp['base']['role']} is {record['outcome']} (export gates: {mode})")
                if mode == "enforce":
                    comparison_failures.append(role)
        out = ctx.metrics_path("verification.json")
        atomic_write_json(out, report)
        ctx.add_output(out, "verification")
        if cmp is not None:
            base = cmp["base"]
            cmp_out = ctx.metrics_path("comparison.json")
            atomic_write_json(
                cmp_out,
                {
                    "description": "exported runtime vs baseline runtime; counts are correct/graded, refused/prompts, responses with integrity issues/responses",
                    "suite": C.suite_ref(cmp["spec"]),
                    "contract": cmp["contract"],
                    "export_gates": config.get("selection.export_gates"),
                    "baseline": {
                        "role": base["role"],
                        "artifact_id": (base["entry"] or {}).get("artifact_id"),
                        "assessment_id": cmp["record"]["assessment_id"] if cmp["record"] else None,
                        **{k: base["scored"].get(k) for k in ("measured", "reason", "benign_correct", "heldout_refused", "integrity_issues")},
                    },
                    "exports": compared,
                    "redacted": ctx.redact,
                },
            )
            ctx.add_output(cmp_out, "comparison")
        ctx.diagnostics.update(
            integrity_failures=integrity_failures,
            inference_failures=inference_failures,
            comparison_failures=comparison_failures,
            comparison={r: c["outcome"] for r, c in compared.items()},
        )
        if "master" in integrity_failures:
            raise PipelineError(ExitCode.VERIFY, "E_MASTER_CORRUPT", "The master checkpoint failed integrity verification")
        failures = sorted(set(integrity_failures + inference_failures + comparison_failures))
        if failures:
            if config.get("execution.allow_partial_exports") and len(failures) < len(roles):
                ctx.partial = True
            else:
                raise PipelineError(ExitCode.VERIFY, "E_VERIFY_FAILED", f"Verification failed for: {', '.join(failures)}")
