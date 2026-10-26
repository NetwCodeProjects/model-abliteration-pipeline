"""Lambda Stack profile: reuse the provider's driver/CUDA stack, isolate Python packages.

This profile never installs Lambda Stack and never upgrades the system. A missing NVIDIA
driver is a blocker (it should come with Lambda Stack); generic build tools may be installed
only with ``environment.allow_system_packages`` and confirmation.
"""

from __future__ import annotations

from typing import Any

from .base import GPU_COMPONENTS, EnvironmentProfile, HostAssessment
from .detect import lambda_stack_markers


class LambdaStackProfile(EnvironmentProfile):
    name = "lambda-stack"

    def profile_specific(self, assessment: HostAssessment, components: set[str], config: dict[str, Any]) -> None:
        markers = lambda_stack_markers(self.probe)
        if markers:
            assessment.notes.append(f"Lambda Stack markers: {', '.join(markers)}")
        else:
            assessment.notes.append(
                "No Lambda Stack packaging markers found; continuing because the profile was selected explicitly. Host GPU stack will be reused as-is."
            )
        if components & GPU_COMPONENTS and any(r.command == "nvidia-smi" for r in assessment.missing):
            assessment.blockers.append(
                "nvidia-smi not found: the Lambda Stack driver is missing or broken. This profile does not install drivers; "
                "fix the image (e.g. reinstall Lambda Stack per Lambda's documentation) or use --ubuntu with "
                "environment.allow_driver_changes."
            )
        if self.probe.which("nvcc"):
            assessment.notes.append(f"Reusing host CUDA toolkit (nvcc {self.probe.nvcc_version() or 'unknown version'}).")

    def torch_plan(self, config: dict[str, Any]) -> dict[str, Any]:
        source = config["environment"]["torch_source"]
        if source == "system":
            return {"source": "system", "backend": None, "note": "venv with --system-site-packages (experimental)"}
        return {"source": "pip", "backend": config["environment"]["torch_backend"]}
