"""Stage 09 - verify_exports: re-hash every exported file, check tokenizer/config/processor
preservation, read GGUF metadata, and run real loader smoke tests where a backend is configured.

``inference_passed`` stays ``null`` when no loader test ran; modalities are only listed after a
real test of that modality.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

from ..adapters import llama_cpp as LC
from ..artifacts.gguf_meta import read_metadata
from ..artifacts.manifest import PROCESSOR_FILES, TOKENIZER_FILES
from ..errors import ExitCode, Interrupted, PipelineError
from ..evaluation.qualification import integrity_issues
from ..fsutil import atomic_write_bytes, atomic_write_json, file_inventory, read_json, sha256_file
from .base import Stage, StageContext
from .stage06_qualify import run_generation_script

PRESERVED = TOKENIZER_FILES + PROCESSOR_FILES + ("generation_config.json",)


def tiny_png(path: Path, rgb: tuple[int, int, int] = (220, 20, 20), size: int = 64) -> None:
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    atomic_write_bytes(path, png)


def compare_files(artifact_dir: Path, source_dir: Path) -> dict[str, str]:
    result = {}
    for name in PRESERVED:
        src = source_dir / name
        if not src.is_file():
            continue
        dst = artifact_dir / name
        if not dst.is_file():
            result[name] = "missing"
        elif sha256_file(src) == sha256_file(dst):
            result[name] = "identical"
        elif name.endswith(".json"):
            try:
                same = json.loads(src.read_text(encoding="utf-8")) == json.loads(dst.read_text(encoding="utf-8"))
            except ValueError:
                same = False
            result[name] = "json-equivalent" if same else "differs"
        else:
            result[name] = "differs"
    return result


class VerifyExportsStage(Stage):
    id = "09"
    name = "verify_exports"
    title = "Verify exports"
    gpu = True
    config_keys = (
        "quantization.smoke_prompts",
        "quantization.smoke_max_tokens",
        "quantization.fp8.verify_backend",
        "quantization.fp8.vllm_max_model_len",
        "quantization.fp8.vllm_gpu_memory_utilization",
        "quantization.gguf.smoke_context",
        "quantization.gguf.smoke_gpu_layers",
        "execution.allow_partial_exports",
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

    def target_outputs(self, config) -> list[str]:
        return ["runs/<run>/metrics/verification.json", "manifests updated with validation/inference results"]

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
                "tested_context": data.get("max_total_tokens"),
                "modalities": ["text"] if ok else [],
            }
        if not ctx.config.get("abliteration.enabled"):
            return {"ran": False}
        prompts = [{"id": f"smoke-{i}", "prompt": p, "expect_any": []} for i, p in enumerate(ctx.config.get("quantization.smoke_prompts"))]
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
        )
        outputs = [r.get("response", "") for r in raw.get("benign", [])]
        ok = bool(raw.get("load_ok")) and bool(outputs) and not any(integrity_issues(o, include_thinking=True) for o in outputs)
        return {
            "ran": True,
            "passed": ok,
            "backend": "transformers",
            "backend_revision": f"transformers {raw['_runtime']['versions'].get('transformers')}",
            "tested_context": raw.get("max_total_tokens"),
            "modalities": ["text"] if ok else [],
            "samples": outputs[:3],
        }

    def _smoke_fp8(self, ctx: StageContext, path: str) -> dict[str, Any]:
        config = ctx.config
        if config.get("quantization.fp8.verify_backend") != "vllm":
            return {"ran": False, "reason": "verify_backend=none"}
        record = ctx.run.components(ctx.log_dir).get("vllm")
        out = ctx.log_dir / "vllm-smoke.json"
        out.unlink(missing_ok=True)
        prompts_file = ctx.log_dir / "smoke-prompts.json"
        atomic_write_json(prompts_file, config.get("quantization.smoke_prompts"))
        result = ctx.run.runner.run(
            [
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
            ],
            cwd=None,
            env=ctx.run.child_env(),
            log_dir=ctx.log_dir,
            name="vllm-smoke",
            timeout_seconds=3600,
        )
        if result.interrupted:
            raise Interrupted()
        data = read_json(out) if out.exists() else {"ok": False, "error": f"no result (exit {result.returncode})"}
        outputs = data.get("outputs") or []
        ok = bool(data.get("ok")) and bool(outputs) and not any(integrity_issues(o, include_thinking=True) for o in outputs)
        return {
            "ran": True,
            "passed": ok,
            "backend": "vllm",
            "backend_revision": f"vllm {record.versions.get('vllm')}",
            "tested_context": config.get("quantization.fp8.vllm_max_model_len") if ok else None,
            "modalities": ["text"] if ok else [],
            "samples": outputs[:3],
            "error": data.get("error"),
            "load_seconds": data.get("load_seconds"),
        }

    def _smoke_gguf(self, ctx: StageContext, art_dir: Path) -> dict[str, Any]:
        config = ctx.config
        record = ctx.run.components(ctx.log_dir).get("llama_cpp")
        ggufs = sorted(art_dir.glob("*.gguf"))
        mmproj = next((p for p in ggufs if p.name.startswith("mmproj")), None)
        model = next((p for p in ggufs if not p.name.startswith("mmproj")), None)
        if model is None:
            return {"ran": False, "reason": "no model gguf"}
        layers = config.get("quantization.gguf.smoke_gpu_layers")
        if layers == "auto":
            layers = "all" if record.build.get("cuda") else "0"
        ctxlen = config.get("quantization.gguf.smoke_context")
        timeout = float(config.get("quantization.gguf.smoke_timeout_minutes")) * 60
        samples, ok = [], True
        for i, prompt in enumerate(config.get("quantization.smoke_prompts")):
            res = ctx.run.runner.run(
                LC.completion_argv(record.cmd("llama-completion"), model, prompt, config.get("quantization.smoke_max_tokens"), ctxlen, layers),
                cwd=None,
                env=ctx.run.child_env(),
                log_dir=ctx.log_dir,
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
            "tested_context": ctxlen if ok else None,
            "gpu_layers": layers,
            "modalities": ["text"] if ok else [],
            "samples": samples,
        }
        if mmproj is not None:
            image = ctx.log_dir / "smoke-red-square.png"
            tiny_png(image)
            res = ctx.run.runner.run(
                LC.mtmd_argv(record.cmd("llama-mtmd-cli"), model, mmproj, image, "What color is this image? Answer in one word.", 16, ctxlen, layers),
                cwd=None,
                env=ctx.run.child_env(),
                log_dir=ctx.log_dir,
                name=f"gguf-mtmd-{art_dir.name}",
                timeout_seconds=timeout,
            )
            out = Path(res.stdout_log).read_text(encoding="utf-8", errors="replace")
            image_ok = res.ok and bool(out.strip().splitlines()[1:] if out.strip() else False)
            result["image_test"] = {"passed": image_ok, "mentions_red": "red" in out.lower(), "returncode": res.returncode}
            if image_ok and ok:
                result["modalities"].append("image")
        return result

    # ------------------------------------------------------------------ execute
    def execute(self, ctx: StageContext) -> None:
        run, config = ctx.run, ctx.config
        roles = {r: e for r, e in run.registry.roles().items() if r == "master" or r.startswith("export:")}
        if not roles:
            raise PipelineError(ExitCode.VERIFY, "E_NOTHING_TO_VERIFY", "No master or exports are registered for this run")
        source = run.registry.get("source")["manifest_data"]
        source_dir = Path(source["artifact"]["uri"])
        report: dict[str, Any] = {}
        integrity_failures, inference_failures = [], []
        for role in sorted(roles):
            run.check_interrupt()
            entry = run.registry.get(role)
            manifest = entry["manifest_data"]
            art_dir = Path(manifest["artifact"]["uri"])
            item: dict[str, Any] = {"artifact_id": manifest["artifact_id"], "path": str(art_dir)}
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
            fmt = manifest["artifact"]["format"]
            if fmt == "safetensors":
                item["preservation"] = compare_files(art_dir, source_dir)
                bad = {k: v for k, v in item["preservation"].items() if v in ("missing", "differs") and not (k == "config.json")}
                if bad:
                    ctx.warn(f"{role}: tokenizer/processor files not preserved verbatim: {bad}")
            else:
                gguf_checks = {}
                for f in expected:
                    meta = read_metadata(art_dir / f)["kv"]
                    gguf_checks[f] = {
                        "architecture": meta.get("general.architecture"),
                        "has_tokenizer": "tokenizer.ggml.model" in meta,
                        "has_chat_template": "tokenizer.chat_template" in meta,
                    }
                    if not f.startswith("mmproj") and (not meta.get("general.architecture") or "tokenizer.ggml.model" not in meta):
                        problems.append(f"{f}: missing architecture/tokenizer metadata")
                item["gguf_metadata"] = gguf_checks
            if role == "master":
                smoke = self._smoke_master(ctx, str(art_dir))
            elif role == "export:fp8":
                smoke = self._smoke_fp8(ctx, str(art_dir))
            else:
                smoke = self._smoke_gguf(ctx, art_dir)
            item["integrity_problems"] = problems
            item["smoke"] = smoke
            report[role] = item
            manifest["validation"]["integrity_passed"] = not problems
            if smoke.get("ran"):
                manifest["validation"]["inference_passed"] = bool(smoke["passed"])
                manifest["validation"]["modalities_tested"] = smoke.get("modalities", [])
                manifest["inference"] = {
                    "backend": smoke.get("backend"),
                    "backend_revision": smoke.get("backend_revision"),
                    "tested_context": smoke.get("tested_context") if smoke.get("tested_context") else None,
                }
                if role.startswith("export:") and manifest["evaluation"]["metrics_provenance"] == "not-evaluated":
                    manifest["evaluation"]["metrics_provenance"] = "integrity-only"
            ext = manifest.setdefault("extensions", {})
            ext["verification"] = {k: v for k, v in item.items() if k not in ("artifact_id", "path")}
            run.registry.update(role, manifest)
            if problems:
                integrity_failures.append(role)
                ctx.warn(f"{role}: integrity problems {problems[:3]}")
            if smoke.get("ran") and not smoke["passed"]:
                inference_failures.append(role)
                ctx.warn(f"{role}: loader smoke test FAILED")
            elif not smoke.get("ran"):
                ctx.warn(f"{role}: no loader smoke test ran ({smoke.get('reason', 'not configured')}); inference_passed stays null")
        out = ctx.metrics_path("verification.json")
        atomic_write_json(out, report)
        ctx.add_output(out, "verification")
        ctx.diagnostics.update(integrity_failures=integrity_failures, inference_failures=inference_failures)
        if "master" in integrity_failures:
            raise PipelineError(ExitCode.VERIFY, "E_MASTER_CORRUPT", "The master checkpoint failed integrity verification")
        failures = sorted(set(integrity_failures + inference_failures))
        if failures:
            if config.get("execution.allow_partial_exports") and len(failures) < len(roles):
                ctx.partial = True
            else:
                raise PipelineError(ExitCode.VERIFY, "E_VERIFY_FAILED", f"Verification failed for: {', '.join(failures)}")
