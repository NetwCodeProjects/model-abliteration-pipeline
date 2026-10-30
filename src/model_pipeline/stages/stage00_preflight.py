"""Stage 00 - preflight: host facts, profile detection, GPU health, disk/VRAM warnings.

Always re-evaluated (host state changes between invocations). No architecture checks.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from ..environment.capabilities import assess, finding
from ..environment.detect import classify_storage, detect_profile, lambda_stack_markers
from ..errors import ExitCode, PipelineError
from ..fsutil import atomic_write_json, disk_free_bytes, human_bytes
from ..planning import components_for_config, estimate_disk, known_source_size, needs_gpu
from .base import Stage, StageContext


class PreflightStage(Stage):
    id = "00"
    name = "preflight"
    title = "Host preflight"
    config_keys = ("environment.profile", "storage.root")
    always_rerun = True

    def target_outputs(self, config) -> list[str]:
        return ["runs/<run>/metrics/preflight.json"]

    def execute(self, ctx: StageContext) -> None:
        run, config, probe = ctx.run, ctx.config, ctx.run.probe
        facts: dict[str, Any] = {
            "platform": probe.platform(),
            "python": platform.python_version(),
            "os_release": probe.os_release(),
            "kernel": probe.kernel(),
            "cpu": probe.cpu(),
            "memory_total_bytes": probe.memory_total_bytes(),
            "gpus": probe.gpus(),
            "cuda_driver_version": probe.cuda_driver_version(),
            "nvcc_version": probe.nvcc_version(),
            "lambda_stack_markers": lambda_stack_markers(probe),
            "hf_token_present": bool(run.hf_token()),
        }
        profile, reason = detect_profile(probe, config.get("environment.profile"))
        facts["profile"] = {"resolved": profile, "reason": reason}
        ctx.diagnostics.update(profile=profile, profile_reason=reason)
        ctx.info(f"Environment profile: {profile} ({reason})")
        if not probe.platform().startswith("linux"):
            raise PipelineError(
                ExitCode.PREFLIGHT,
                "E_ENV_UNSUPPORTED_PLATFORM",
                f"Execution stages require Linux (Ubuntu); this host is {probe.platform()}",
                hint="Use --dry-run/--help here; run the pipeline on the Lambda/Ubuntu host.",
            )
        components = components_for_config(config)
        facts["components_required"] = sorted(components)
        gpus = facts["gpus"] or []
        if needs_gpu(config) and not gpus:
            raise PipelineError(
                ExitCode.PREFLIGHT,
                "E_PREFLIGHT_NO_GPU",
                "The configured stages need an NVIDIA GPU but nvidia-smi reported none",
                hint="Check the driver (nvidia-smi). For CPU-only GGUF work use --quantize-only with only GGUF formats.",
            )
        for gpu in gpus:
            if gpu.get("ecc_uncorrected_volatile") not in (None, "0", 0):
                ctx.warn(f"GPU {gpu['index']} reports uncorrected ECC errors: {gpu['ecc_uncorrected_volatile']}")
        workspace = Path(config.get("storage.root"))
        cache = Path(config.get("environment.cache_root"))
        storage = {
            "workspace": {"path": str(workspace), "free_bytes": disk_free_bytes(workspace), **classify_storage(probe.mount_for(workspace))},
            "cache": {"path": str(cache), "free_bytes": disk_free_bytes(cache), **classify_storage(probe.mount_for(cache))},
        }
        facts["storage"] = storage
        if storage["workspace"]["kind"] in ("instance-local", "ephemeral"):
            ctx.warn(
                f"Workspace is on {storage['workspace']['kind']} storage ({storage['workspace'].get('fstype')}); "
                "copy finished exports to durable storage before terminating the instance."
            )
        floor = float(config.get("environment.min_free_disk_gb")) * 1024**3
        free = storage["workspace"]["free_bytes"]
        if free < floor:
            raise PipelineError(
                ExitCode.PREFLIGHT,
                "E_DISK_FLOOR",
                f"Only {human_bytes(free)} free under {workspace}; environment.min_free_disk_gb requires {human_bytes(floor)}",
                hint="Free space, mount a larger volume, or point --output-dir elsewhere.",
            )
        source_size = known_source_size(config)
        estimate = estimate_disk(config, source_size)
        facts["disk_estimate"] = estimate
        findings = assess(config, gpus, needs_gpu=needs_gpu(config), heretic="heretic" in components, source_size=source_size)
        if estimate["total_bytes"] is not None:
            needed = estimate["total_bytes"] + float(config.get("environment.disk_buffer_gb")) * 1024**3
            if free < needed:
                findings.append(
                    finding(
                        "estimated_resource_shortage",
                        "disk_estimate",
                        f"Estimated need {human_bytes(needed)} (incl. buffer) exceeds free space {human_bytes(free)}",
                        "Estimate: source + candidate + master + exports (planning.estimate_disk) + environment.disk_buffer_gb",
                    )
                )
        else:
            ctx.info("Disk requirement unknown until the model is resolved (stage 02 re-checks against real sizes).")
        facts["capabilities"] = findings
        for item in findings:
            ctx.warn(f"[{item['category']}] {item['message']}")
        unsupported = [f for f in findings if f["category"] == "unsupported_configuration"]
        if unsupported:
            atomic_write_json(ctx.metrics_path("preflight.json"), facts)
            raise PipelineError(
                ExitCode.PREFLIGHT,
                "E_CAPABILITY_UNSUPPORTED",
                "; ".join(f["message"] for f in unsupported),
                hint="Basis: " + " | ".join(f["basis"] for f in unsupported),
            )
        out = ctx.metrics_path("preflight.json")
        atomic_write_json(out, facts)
        ctx.add_output(out, "preflight_report")
        ctx.diagnostics["capabilities"] = {c: [f["rule"] for f in findings if f["category"] == c] for c in sorted({f["category"] for f in findings})}
        ctx.diagnostics["gpus"] = [{k: g.get(k) for k in ("name", "memory_total_mib", "driver_version", "compute_capability")} for g in gpus]
