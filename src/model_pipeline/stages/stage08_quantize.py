"""Stage 08 - quantize: independent, resumable FP8 / GGUF jobs from a preserved high-precision input.

Each format is its own job with its own sub-receipt. A failure in one format never touches other
outputs; with ``execution.allow_partial_exports`` the stage succeeds as *partial*. The input
checkpoint is fingerprinted before and after and must be unchanged.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import configspec
from ..adapters import fp8 as FP8
from ..adapters import llama_cpp as LC
from ..artifacts.gguf_meta import read_metadata
from ..artifacts.manifest import build_manifest
from ..errors import ExitCode, Interrupted, PipelineError
from ..fsutil import (
    atomic_write_json,
    disk_free_bytes,
    file_inventory,
    human_bytes,
    quick_fingerprint,
    read_json,
    remove_tree,
    safe_component,
    sha256_file,
    sha256_obj,
)
from ..telemetry import utc_now
from .base import Stage, StageContext


def input_role(config) -> str:
    choice = config.get("quantization.input")
    return {"selected_master": "master", "source": "source", "path": "quant_input"}[choice]


class QuantizeStage(Stage):
    id = "08"
    name = "quantize"
    title = "Deployment quantization"
    gpu = True
    config_keys = (
        "quantization.enabled",
        "quantization.input",
        "quantization.input_path",
        "quantization.formats",
        "quantization.fp8.scheme",
        "quantization.fp8.ignore",
        "quantization.fp8.backend",
        "quantization.fp8.llmcompressor_version",
        "quantization.fp8.compressed_tensors_version",
        "quantization.fp8.device",
        "quantization.gguf.llama_cpp_revision",
        "quantization.gguf.build_cuda",
        "quantization.gguf.intermediate_type",
        "quantization.gguf.imatrix_dataset",
        "quantization.gguf.imatrix_chunks",
        "quantization.gguf.preserve_multimodal",
        "quantization.gguf.mmproj_type",
        "execution.allow_partial_exports",
    )

    def dependencies(self, config) -> tuple[str, ...]:
        return ("01", "07") if input_role(config) == "master" else ("01", "02")

    def components_used(self, config) -> set[str]:
        formats = config.get("quantization.formats") or []
        used = {"fp8"} if "fp8" in formats else set()
        if any(f in configspec.GGUF_TYPES for f in formats):
            used.add("llama_cpp")
        return used

    def applicable(self, config) -> tuple[bool, str]:
        if not config.get("quantization.enabled"):
            return False, "quantization disabled"
        if not config.get("quantization.formats"):
            return False, "no quantization formats requested"
        return True, ""

    def target_outputs(self, config) -> list[str]:
        outs = []
        for fmt in config.get("quantization.formats"):
            outs.append(f"workspace/exports/{'fp8' if fmt == 'fp8' else 'gguf'}/<family>-<parent>-{fmt}/")
        return outs

    # ----------------------------------------------------------------- helpers
    def _sub_receipt_path(self, ctx: StageContext, fmt: str) -> Path:
        return ctx.run.receipts_dir / "08-quantize" / f"{fmt}.json"

    def _reusable(self, ctx: StageContext, fmt: str, job_key: str) -> dict[str, Any] | None:
        path = self._sub_receipt_path(ctx, fmt)
        if not path.exists():
            return None
        sub = read_json(path)
        if sub.get("outcome") != "succeeded" or sub.get("job_key") != job_key:
            return None
        entry = ctx.run.registry.get(sub["role"])
        if entry is None or entry["artifact_id"] != sub["artifact_id"]:
            return None
        for item in entry["manifest_data"]["artifact"]["files"]:
            target = Path(entry["manifest_data"]["artifact"]["uri"]) / item["relative_path"]
            if not target.is_file() or target.stat().st_size != item["size_bytes"]:
                return None
        return sub

    def _run_tool(self, ctx: StageContext, argv: list[str], name: str, hours: float = 8, cwd: Path | None = None) -> None:
        result = ctx.run.runner.run(argv, cwd=cwd, env=ctx.run.child_env(), log_dir=ctx.log_dir, name=name, timeout_seconds=hours * 3600)
        if result.interrupted:
            raise Interrupted()
        if not result.ok:
            raise PipelineError(
                ExitCode.QUANTIZE,
                "E_QUANT_TOOL_FAILED",
                f"{name} failed (exit {result.returncode})",
                hint=f"{result.stderr_log}: {' | '.join(result.tail[-6:])}",
            )

    def _finalize(self, ctx: StageContext, partial: Path, final: Path) -> list[dict[str, Any]]:
        files = [e.to_dict() for e in file_inventory(partial)]
        if final.exists():
            raise PipelineError(
                ExitCode.QUANTIZE,
                "E_EXPORT_EXISTS",
                f"Refusing to overwrite finalized export {final}",
                hint="It belongs to another recipe/run; remove it manually if it is obsolete.",
            )
        os.replace(partial, final)
        return files

    def _existing_export(self, ctx: StageContext, final: Path, role: str, parent_id: str, recipe_id: str) -> dict[str, Any] | None:
        """Reuse a finalized export from any run when parent and recipe match and files are intact."""
        sidecar = final.with_name(final.name + ".manifest.json")
        if not (final.is_dir() and sidecar.is_file()):
            return None
        manifest = read_json(sidecar)
        if manifest.get("parent_artifact_id") != parent_id or manifest["quantization"].get("recipe_id") != recipe_id:
            return None
        for item in manifest["artifact"]["files"]:
            path = final / item["relative_path"]
            if not path.is_file() or path.stat().st_size != item["size_bytes"]:
                return None
        ctx.run.registry.register(manifest, role, final, self.id)
        ctx.info(f"Reusing existing export {final} (same parent artifact and recipe {recipe_id})")
        return {"role": role, "artifact_id": manifest["artifact_id"], "path": str(final), "reused_existing_export": True}

    # --------------------------------------------------------------------- FP8
    def _fp8(self, ctx: StageContext, inp: dict[str, Any]) -> dict[str, Any]:
        run, config = ctx.run, ctx.config
        recipe = FP8.recipe(config)
        record = run.components(ctx.log_dir).get("fp8")
        in_dir = Path(inp["uri"])
        name = safe_component(f"{config.get('model.family')}-{inp['artifact_id'][4:12]}-{recipe['scheme']}")
        final = run.workspace / "exports" / "fp8" / name
        existing = self._existing_export(ctx, final, "export:fp8", inp["artifact_id"], recipe["recipe_id"])
        if existing:
            return existing
        partial = final.with_name(final.name + ".partial")
        remove_tree(partial)
        size = sum(f["size_bytes"] for f in inp["files"])
        if disk_free_bytes(run.workspace) < 0.7 * size:
            raise PipelineError(ExitCode.QUANTIZE, "E_DISK_INSUFFICIENT", f"FP8 export needs ~{human_bytes(0.6 * size)}")
        work = run.run_dir / "checkpoints" / "quantize"
        recipe_path, result_path = work / "fp8-recipe.json", work / "fp8-result.json"
        atomic_write_json(recipe_path, recipe)
        result_path.unlink(missing_ok=True)
        try:
            self._run_tool(
                ctx, FP8.quantize_argv(record.cmd("python"), record.scripts_dir / "fp8_quantize.py", in_dir, partial, recipe_path, result_path), "fp8-quantize"
            )
            result = read_json(result_path) if result_path.exists() else {}
            if not result.get("ok"):
                raise PipelineError(ExitCode.QUANTIZE, "E_FP8_FAILED", f"FP8 quantizer reported failure: {result.get('error')}")
            problems = FP8.check_output_config(read_json(partial / "config.json"), recipe["preset"])
            if not list(partial.glob("*.safetensors")):
                problems.append("no safetensors written")
            if problems:
                raise PipelineError(ExitCode.QUANTIZE, "E_FP8_OUTPUT_INVALID", f"FP8 output failed checks: {problems}")
            files = self._finalize(ctx, partial, final)
        finally:
            remove_tree(partial)
        qc = read_json(final / "config.json").get("quantization_config")
        manifest = build_manifest(
            run_id=run.run_id,
            model_family=config.get("model.family"),
            source_model=inp["source_model"],
            source_revision=inp["source_revision"],
            parent_artifact_id=inp["artifact_id"],
            kind="quantization",
            method=f"llmcompressor.model_free_ptq:{recipe['preset']}",
            config_hash=sha256_obj(recipe),
            tool_revision=f"llmcompressor {record.versions.get('llmcompressor')}; compressed-tensors {record.versions.get('compressed-tensors')}",
            precision="fp8",
            fmt="safetensors",
            uri=str(final),
            files=files,
            quantization={"recipe_id": recipe["recipe_id"], "calibration_revision": None, "imatrix_hash": None},
            validation={"integrity_passed": True, "inference_passed": None, "modalities_tested": []},
            extensions={
                "recipe": recipe,
                "quantization_config": qc,
                "note": "Data-free FP8 produced by this platform; not byte-identical to any official FP8 release.",
                "quantizer_result": {k: result.get(k) for k in ("duration_s", "peak_gpu_memory_bytes", "device")},
            },
        )
        run.registry.register(manifest, "export:fp8", final, self.id)
        return {"role": "export:fp8", "artifact_id": manifest["artifact_id"], "path": str(final), "recipe_id": recipe["recipe_id"]}

    # -------------------------------------------------------------------- GGUF
    def _gguf_intermediate(self, ctx: StageContext, inp: dict[str, Any], record) -> dict[str, Any]:
        run, config = ctx.run, ctx.config
        itype = config.get("quantization.gguf.intermediate_type")
        commit = record.versions.get("commit", "unknown")[:8]
        inter = run.workspace / "models" / "gguf-intermediate" / f"{inp['artifact_id']}-{commit}-{itype}"
        meta_path = inter / "intermediate.json"
        family = config.get("model.family")
        if meta_path.exists():
            meta = read_json(meta_path)
            if all(Path(p).is_file() and Path(p).stat().st_size == s for p, s in meta.get("sizes", {}).items()):
                ctx.info(f"Reusing GGUF intermediate {inter}")
                return meta
        remove_tree(inter)
        work = inter.with_name(inter.name + ".partial")
        remove_tree(work)
        work.mkdir(parents=True)
        model_file = work / f"{family}-{itype}.gguf"
        in_dir = Path(inp["uri"])
        self._run_tool(ctx, LC.convert_argv(record.cmd("convert"), in_dir, model_file, itype), "gguf-convert")
        if not model_file.is_file():
            raise PipelineError(ExitCode.QUANTIZE, "E_GGUF_CONVERT", f"Converter did not produce {model_file.name}")
        read_metadata(model_file)
        cfg = read_json(in_dir / "config.json")
        has_vision = "vision_config" in cfg
        mmproj = None
        mmproj_error = None
        if has_vision and config.get("quantization.gguf.preserve_multimodal"):
            mtype = config.get("quantization.gguf.mmproj_type")
            before = set(work.glob("*.gguf"))
            try:
                self._run_tool(ctx, LC.mmproj_argv(record.cmd("convert"), in_dir, work / f"{family}-{mtype}.gguf", mtype), "gguf-mmproj")
                produced = sorted(set(work.glob("*.gguf")) - before)
                produced = [p for p in produced if p.name.startswith("mmproj")] or produced
                if len(produced) != 1:
                    raise PipelineError(ExitCode.QUANTIZE, "E_MMPROJ_FAILED", f"Expected one mmproj file, found {[p.name for p in produced]}")
                read_metadata(produced[0])
                mmproj = produced[0].name
            except PipelineError as exc:
                mmproj_error = str(exc)
                ctx.warn(f"Multimodal projector conversion failed: {exc.message}")
        imatrix = None
        imatrix_data = (run.load_receipt("02") or {}).get("diagnostics", {}).get("imatrix_data")
        if imatrix_data:
            out = work / f"imatrix-{imatrix_data['sha256'][:12]}.gguf"
            gpu_layers = "all" if record.build.get("cuda") else "0"
            self._run_tool(
                ctx,
                LC.imatrix_argv(
                    record.cmd("llama-imatrix"), model_file, Path(imatrix_data["path"]), out, config.get("quantization.gguf.imatrix_chunks"), gpu_layers
                ),
                "gguf-imatrix",
                hours=12,
            )
            if not out.is_file():
                raise PipelineError(ExitCode.QUANTIZE, "E_IMATRIX_FAILED", "llama-imatrix produced no output")
            imatrix = {"file": out.name, "sha256": sha256_file(out), "calibration": imatrix_data}
        os.replace(work, inter)
        model_path = inter / model_file.name
        meta = {
            "dir": str(inter),
            "model": str(model_path),
            "model_sha256": sha256_file(model_path),
            "intermediate_type": itype,
            "has_vision": has_vision,
            "mmproj": str(inter / mmproj) if mmproj else None,
            "mmproj_error": mmproj_error,
            "imatrix": {**imatrix, "path": str(inter / imatrix["file"])} if imatrix else None,
            "converter": {"commit": record.versions.get("commit"), "gguf": record.versions.get("gguf"), "transformers": record.versions.get("transformers")},
            "created_utc": utc_now(),
        }
        meta["sizes"] = {p: Path(p).stat().st_size for p in [meta["model"], meta["mmproj"], (meta["imatrix"] or {}).get("path")] if p}
        atomic_write_json(meta_path, meta)
        return meta

    def _gguf_recipe_id(self, ctx: StageContext, record, fmt: str) -> str:
        config = ctx.config
        rid = f"llama.cpp-{record.versions.get('commit', 'unknown')[:8]}:{config.get('quantization.gguf.intermediate_type')}->{fmt.upper()}"
        imatrix_data = (ctx.run.load_receipt("02") or {}).get("diagnostics", {}).get("imatrix_data")
        if imatrix_data:
            rid += f":imatrix-data-{imatrix_data['sha256'][:12]}-chunks{config.get('quantization.gguf.imatrix_chunks') or 'all'}"
        return rid + (":mmproj" if config.get("quantization.gguf.preserve_multimodal") else ":text-only")

    def _gguf_target(self, ctx: StageContext, inp: dict[str, Any], fmt: str) -> Path:
        name = safe_component(f"{ctx.config.get('model.family')}-{inp['artifact_id'][4:12]}-{fmt}")
        return ctx.run.workspace / "exports" / "gguf" / name

    def _gguf(self, ctx: StageContext, inp: dict[str, Any], fmt: str, get_inter: Callable[[], dict[str, Any]], record) -> dict[str, Any]:
        run, config = ctx.run, ctx.config
        recipe_id = self._gguf_recipe_id(ctx, record, fmt)
        final = self._gguf_target(ctx, inp, fmt)
        existing = self._existing_export(ctx, final, f"export:gguf:{fmt}", inp["artifact_id"], recipe_id)
        if existing:
            return existing
        inter = get_inter()
        if inter["has_vision"] and config.get("quantization.gguf.preserve_multimodal") and not inter["mmproj"]:
            raise PipelineError(
                ExitCode.QUANTIZE,
                "E_MMPROJ_FAILED",
                f"preserve_multimodal=true but the vision projector could not be converted ({inter['mmproj_error']}); not producing a text-only package",
                hint="Set quantization.gguf.preserve_multimodal: false to export text-only GGUF files explicitly.",
            )
        imatrix = inter.get("imatrix")
        if fmt in configspec.GGUF_TYPES_REQUIRING_IMATRIX and not imatrix:
            raise PipelineError(
                ExitCode.QUANTIZE,
                "E_IMATRIX_REQUIRED",
                f"{fmt} requires an importance matrix (llama-quantize aborts without one)",
                hint="Set quantization.gguf.imatrix_dataset to a calibration text file.",
            )
        family = config.get("model.family")
        tag = inp["artifact_id"][4:12]
        partial = final.with_name(final.name + ".partial")
        remove_tree(partial)
        partial.mkdir(parents=True)
        out_file = partial / LC.gguf_name(family, tag, fmt)
        try:
            size = Path(inter["model"]).stat().st_size
            if disk_free_bytes(run.workspace) < size * 0.6:
                raise PipelineError(ExitCode.QUANTIZE, "E_DISK_INSUFFICIENT", f"Not enough space for {fmt}")
            self._run_tool(
                ctx,
                LC.quantize_argv(record.cmd("llama-quantize"), Path(inter["model"]), out_file, fmt, Path(imatrix["path"]) if imatrix else None),
                f"gguf-quantize-{fmt}",
            )
            if not out_file.is_file():
                raise PipelineError(ExitCode.QUANTIZE, "E_GGUF_QUANTIZE", f"llama-quantize produced no {out_file.name}")
            meta = read_metadata(out_file)
            modalities = ["text"]
            if inter["mmproj"]:
                target = partial / Path(inter["mmproj"]).name
                try:
                    os.link(inter["mmproj"], target)
                except OSError:
                    shutil.copy2(inter["mmproj"], target)
                modalities.append("image")
            files = self._finalize(ctx, partial, final)
        finally:
            remove_tree(partial)
        manifest = build_manifest(
            run_id=run.run_id,
            model_family=family,
            source_model=inp["source_model"],
            source_revision=inp["source_revision"],
            parent_artifact_id=inp["artifact_id"],
            kind="quantization",
            method=f"llama.cpp:convert_hf_to_gguf({inter['intermediate_type']})+llama-quantize:{fmt.upper()}",
            config_hash=sha256_obj(
                {"fmt": fmt, "intermediate": inter["intermediate_type"], "imatrix": (imatrix or {}).get("sha256"), "converter": inter["converter"]}
            ),
            tool_revision=f"llama.cpp {inter['converter']['commit']}",
            precision=fmt,
            fmt="gguf",
            uri=str(final),
            files=files,
            tokenizer={"revision": inp["source_revision"], "hash": None},
            processor={"revision": None, "hash": None},
            quantization={
                "recipe_id": recipe_id,
                "calibration_revision": (imatrix or {}).get("calibration", {}).get("origin"),
                "imatrix_hash": (imatrix or {}).get("sha256"),
            },
            validation={"integrity_passed": True, "inference_passed": None, "modalities_tested": []},
            extensions={
                "gguf_architecture": meta["kv"].get("general.architecture"),
                "intermediate": {"type": inter["intermediate_type"], "sha256": inter["model_sha256"]},
                "multimodal": {
                    "source_has_vision": inter["has_vision"],
                    "mmproj_included": bool(inter["mmproj"]),
                    "claimed_modalities_pending_test": modalities,
                },
                "note": "Tokenizer is embedded in the GGUF; tokenizer.hash is null because no standalone tokenizer file is shipped.",
            },
        )
        run.registry.register(manifest, f"export:gguf:{fmt}", final, self.id)
        return {"role": f"export:gguf:{fmt}", "artifact_id": manifest["artifact_id"], "path": str(final)}

    # ------------------------------------------------------------------ execute
    def execute(self, ctx: StageContext) -> None:
        run, config = ctx.run, ctx.config
        role = input_role(config)
        entry = run.registry.get(role)
        if entry is None:
            raise PipelineError(ExitCode.DEPENDENCY, "E_MISSING_DEPENDENCY", f"No '{role}' artifact registered for quantization input")
        man = entry["manifest_data"]
        inp = {
            "artifact_id": man["artifact_id"],
            "uri": man["artifact"]["uri"],
            "files": man["artifact"]["files"],
            "source_model": man["source_model"],
            "source_revision": man["source_revision"],
        }
        if role == "source" and config.get("abliteration.enabled"):
            ctx.warn("Quantizing the original source instead of an abliterated master (quantization.input=source)")
        before = quick_fingerprint(inp["uri"])
        ctx.input_artifacts.append(inp["artifact_id"])
        formats = config.get("quantization.formats")
        results: dict[str, Any] = {}
        errors: list[PipelineError] = []
        inter: dict[str, Any] | None = None
        llama = None

        def get_inter() -> dict[str, Any]:
            nonlocal inter
            if inter is None:
                inter = self._gguf_intermediate(ctx, inp, llama)
            return inter

        for fmt in formats:
            run.check_interrupt()
            job_key = sha256_obj(
                {
                    "fmt": fmt,
                    "input": inp["artifact_id"],
                    "config": config.subset(list(self.config_keys)),
                    "components": run.load_receipt("01")["diagnostics"].get("component_fingerprints"),
                }
            )
            reused = self._reusable(ctx, fmt, job_key)
            if reused:
                results[fmt] = {**reused, "reused": True}
                ctx.output_artifacts.append(reused["artifact_id"])
                ctx.info(f"{fmt}: reusing validated export {reused['path']}")
                continue
            started = utc_now()
            try:
                if fmt == "fp8":
                    job = self._fp8(ctx, inp)
                else:
                    if llama is None:
                        llama = run.components(ctx.log_dir).get("llama_cpp")
                    job = self._gguf(ctx, inp, fmt, get_inter, llama)
                sub = {"format": fmt, "outcome": "succeeded", "job_key": job_key, "started_utc": started, "ended_utc": utc_now(), **job}
                ctx.output_artifacts.append(job["artifact_id"])
                ctx.info(f"{fmt}: exported {job['path']}")
            except Interrupted:
                raise
            except PipelineError as exc:
                errors.append(exc)
                sub = {"format": fmt, "outcome": "failed", "job_key": job_key, "started_utc": started, "ended_utc": utc_now(), "error": exc.to_dict()}
                ctx.warn(f"{fmt}: FAILED - {exc.message}")
            atomic_write_json(self._sub_receipt_path(ctx, fmt), sub)
            results[fmt] = sub
        if quick_fingerprint(inp["uri"]) != before:
            raise PipelineError(ExitCode.VERIFY, "E_INPUT_MUTATED", f"Quantization input {inp['uri']} changed during stage 08 (immutability violated)")
        if inter is not None and not config.get("quantization.gguf.keep_intermediate"):
            remove_tree(inter["dir"])
            ctx.info("Removed GGUF intermediate (keep_intermediate=false)")
        summary_path = ctx.metrics_path("quantization.json")
        atomic_write_json(summary_path, {"input": inp, "results": results, "intermediate": inter})
        ctx.add_output(summary_path, "quantization_summary")
        ctx.diagnostics.update(results={k: v.get("outcome", "succeeded") for k, v in results.items()}, failed=[e.error_id for e in errors])
        if errors:
            succeeded = [k for k, v in results.items() if v.get("outcome") == "succeeded"]
            if config.get("execution.allow_partial_exports") and succeeded:
                ctx.partial = True
                ctx.warn(f"Partial export: {len(errors)} format(s) failed, {len(succeeded)} succeeded")
            else:
                first = errors[0]
                raise PipelineError(first.code, first.error_id, f"{len(errors)} quantization job(s) failed; first: {first.message}", hint=first.hint)
