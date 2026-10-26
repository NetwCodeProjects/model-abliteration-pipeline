"""Plain Ubuntu profile: report missing prerequisites and offer guarded, opt-in host steps.

Host-level changes (apt packages, NVIDIA driver) are only performed when the matching
config flag is true *and* the run is confirmed (--yes or an interactive 'y'). The driver step
never reboots: the stage stops and prints the reboot command.
"""

from __future__ import annotations

from typing import Any

from .base import GPU_COMPONENTS, EnvironmentProfile, HostAction, HostAssessment


class UbuntuProfile(EnvironmentProfile):
    name = "ubuntu"

    def profile_specific(self, assessment: HostAssessment, components: set[str], config: dict[str, Any]) -> None:
        release = self.probe.os_release()
        if release.get("ID") != "ubuntu":
            assessment.notes.append(f"Host OS is {release.get('PRETTY_NAME', 'unknown')}; the ubuntu profile is only tested on Ubuntu 22.04/24.04.")
        driver_missing = any(r.command == "nvidia-smi" for r in assessment.missing)
        if components & GPU_COMPONENTS and driver_missing:
            if config["environment"]["allow_driver_changes"]:
                assessment.actions.append(
                    HostAction(
                        "Install the recommended NVIDIA server driver (ubuntu-drivers --gpgpu); a reboot is required afterwards",
                        self._sudo(["ubuntu-drivers", "install", "--gpgpu"]),
                        privileged=True,
                        flag="environment.allow_driver_changes",
                        reboot_required=True,
                    )
                )
                if not self.probe.which("ubuntu-drivers"):
                    assessment.actions.insert(
                        0,
                        HostAction(
                            "Install ubuntu-drivers-common",
                            self._sudo(["apt-get", "install", "-y", "--no-install-recommends", "ubuntu-drivers-common"]),
                            privileged=True,
                            flag="environment.allow_driver_changes",
                        ),
                    )
            else:
                assessment.blockers.append(
                    "No NVIDIA driver (nvidia-smi missing). Install a datacenter driver supporting your GPU "
                    "(B200 needs the R570+ open kernel modules) — see docs/deployment-ubuntu.md — or set "
                    "environment.allow_driver_changes: true to let stage 01 run 'ubuntu-drivers install --gpgpu'."
                )
        if not self.probe.which("nvcc") and "llama_cpp" in components:
            assessment.notes.append("nvcc not found: llama.cpp will be built for CPU unless quantization.gguf.build_cuda=on (which then fails).")
