"""Stage 01 - environment: host prerequisites (guarded) and pinned, validated tool components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..environment import get_profile
from ..environment.components import component_specs
from ..errors import ExitCode, PipelineError
from ..fsutil import FileLock, atomic_write_json
from ..planning import components_for_config
from .base import Stage, StageContext

PIN_KEYS = (
    "environment.install_missing",
    "environment.python_version",
    "environment.torch_backend",
    "environment.torch_source",
    "abliteration.engine_repo",
    "abliteration.engine_revision",
    "quantization.fp8.llmcompressor_version",
    "quantization.fp8.compressed_tensors_version",
    "quantization.fp8.verify_backend",
    "quantization.fp8.vllm_version",
    "quantization.gguf.llama_cpp_repo",
    "quantization.gguf.llama_cpp_revision",
    "quantization.gguf.build_cuda",
    "quantization.gguf.cuda_architectures",
    "quantization.gguf.converter_transformers",
)


class EnvironmentStage(Stage):
    id = "01"
    name = "environment"
    title = "Pinned tool environments"
    config_keys = PIN_KEYS

    def dependencies(self, config) -> tuple[str, ...]:
        return ("00",)

    def target_outputs(self, config) -> list[str]:
        comps = sorted(components_for_config(config))
        return [f"workspace/tools/components.json ({', '.join(comps) or 'no components'})"]

    def extra_fingerprint(self, run) -> dict[str, Any]:
        return {"components": sorted(components_for_config(run.config))}

    def validate_outputs(self, run, receipt) -> list[str]:
        problems = super().validate_outputs(run, receipt)
        manager = run.components(run.run_dir / "logs" / "01-environment")
        records = manager.load()
        for name, fingerprint in receipt.get("diagnostics", {}).get("component_fingerprints", {}).items():
            record = records.get(name)
            if record is None or record.fingerprint != fingerprint:
                problems.append(f"component {name} changed or missing since the receipt")
                continue
            for command in record.commands.values():
                if command and not Path(command[0]).exists():
                    problems.append(f"component {name}: {command[0]} no longer exists")
                    break
        return problems

    def _host_actions(self, ctx: StageContext, profile_name: str, components: set[str]) -> None:
        run, config = ctx.run, ctx.config.data
        profile = get_profile(profile_name, run.probe)
        assessment = profile.assess(components, config)
        ctx.diagnostics["host_assessment"] = assessment.to_dict()
        for note in assessment.notes:
            ctx.info(note)
        for req in assessment.optional_missing:
            ctx.warn(f"Optional host tool missing: {req.name} ({req.purpose})")
        if assessment.blockers:
            raise PipelineError(ExitCode.ENVIRONMENT, "E_HOST_PREREQ", "Host prerequisites not met:\n  - " + "\n  - ".join(assessment.blockers))
        if not assessment.missing and not assessment.actions:
            return
        allowed = [a for a in assessment.actions if config_flag(config, a.flag)]
        refused = [a for a in assessment.actions if not config_flag(config, a.flag)]
        if refused:
            commands = "\n  ".join(" ".join(a.argv) for a in refused)
            raise PipelineError(
                ExitCode.ENVIRONMENT,
                "E_HOST_PREREQ",
                f"Missing host prerequisites: {', '.join(r.name for r in assessment.missing)}",
                hint=f"Run manually:\n  {commands}\nor set {', '.join(sorted({a.flag for a in refused}))}: true and confirm with --yes.",
            )
        from ..orchestrator import confirm  # noqa: PLC0415

        for action in allowed:
            confirm(run.services, run.options, f"Privileged host step: {action.description}\n  $ {' '.join(action.argv)}")
            result = run.runner.run(
                action.argv,
                cwd=None,
                env=run.child_env(DEBIAN_FRONTEND="noninteractive"),
                log_dir=ctx.log_dir,
                name="host-" + action.argv[-1].replace("/", "_")[:40],
                timeout_seconds=3600,
            )
            ctx.diagnostics.setdefault("host_actions", []).append({"argv": action.argv, "returncode": result.returncode})
            if not result.ok:
                raise PipelineError(
                    ExitCode.ENVIRONMENT,
                    "E_HOST_ACTION_FAILED",
                    f"Host step failed: {' '.join(action.argv)}",
                    hint="'sudo -n' requires passwordless sudo; otherwise run the command yourself. See " + result.stderr_log,
                )
            if action.reboot_required:
                raise PipelineError(
                    ExitCode.ENVIRONMENT,
                    "E_REBOOT_REQUIRED",
                    "NVIDIA driver installed; a reboot is required before GPU stages can run",
                    hint=f"Reboot manually (sudo reboot), then resume: ./main.sh --resume {run.run_id}. v1 never reboots automatically.",
                )

    def execute(self, ctx: StageContext) -> None:
        run, config = ctx.run, ctx.config
        preflight = run.load_receipt("00") or {}
        profile_name = preflight.get("diagnostics", {}).get("profile")
        if not profile_name:
            raise PipelineError(ExitCode.DEPENDENCY, "E_MISSING_DEPENDENCY", "Preflight receipt has no resolved profile")
        components = components_for_config(config)
        ctx.diagnostics["components"] = sorted(components)
        self._host_actions(ctx, profile_name, components)
        if not components:
            ctx.info("No tool components needed for this configuration")
        profile = get_profile(profile_name, run.probe)
        torch_plan = profile.torch_plan(config.data)
        specs = component_specs(config.data, torch_plan, run.probe)
        manager = run.components(ctx.log_dir)
        fingerprints, actions = {}, {}
        with FileLock(run.tools_dir / ".lock", "tools directory"):
            for name in sorted(components):
                run.check_interrupt()
                record, action = manager.ensure(name, specs[name], allow_install=config.get("environment.install_missing"), gpu_smoke=name != "llama_cpp")
                fingerprints[name] = record.fingerprint
                actions[name] = action
                ctx.tool_revisions[name] = record.versions
                ctx.info(f"Component {name}: {action} ({record.fingerprint[:12]})")
        ctx.diagnostics.update(component_fingerprints=fingerprints, component_actions=actions, torch_plan=torch_plan)
        snapshot = ctx.metrics_path("components.json")
        records = manager.load()
        atomic_write_json(snapshot, {name: records[name].to_dict() for name in sorted(components)})
        ctx.add_output(snapshot, "installation_manifest")


def config_flag(config: dict[str, Any], dotted: str) -> bool:
    current: Any = config
    for part in dotted.split("."):
        current = current.get(part) if isinstance(current, dict) else None
    return bool(current)
