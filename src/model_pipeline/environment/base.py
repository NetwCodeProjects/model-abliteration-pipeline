"""Shared environment-profile machinery: host requirements and guarded host actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .detect import HostProbe

GPU_COMPONENTS = frozenset({"heretic", "fp8", "vllm"})


@dataclass(frozen=True)
class HostRequirement:
    name: str
    command: str
    apt_package: str | None
    needed_for: frozenset[str]
    optional: bool = False
    purpose: str = ""


@dataclass
class HostAction:
    description: str
    argv: list[str]
    privileged: bool
    flag: str  # config flag that must be true for this action
    reboot_required: bool = False


@dataclass
class HostAssessment:
    profile: str
    missing: list[HostRequirement] = field(default_factory=list)
    optional_missing: list[HostRequirement] = field(default_factory=list)
    actions: list[HostAction] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "missing": [r.name for r in self.missing],
            "optional_missing": [r.name for r in self.optional_missing],
            "actions": [{"description": a.description, "argv": a.argv, "privileged": a.privileged, "flag": a.flag} for a in self.actions],
            "blockers": self.blockers,
            "notes": self.notes,
        }


BASE_REQUIREMENTS = (
    HostRequirement("git", "git", "git", frozenset({"heretic", "llama_cpp"}), purpose="clone pinned engine sources"),
    HostRequirement("cmake", "cmake", "cmake", frozenset({"llama_cpp"}), purpose="build llama.cpp"),
    HostRequirement("c++ compiler", "g++", "build-essential", frozenset({"llama_cpp"}), purpose="build llama.cpp"),
    HostRequirement("make", "make", "build-essential", frozenset({"llama_cpp"}), purpose="build llama.cpp"),
    HostRequirement("nvidia-smi", "nvidia-smi", None, GPU_COMPONENTS, purpose="NVIDIA driver for GPU stages"),
    HostRequirement("nvcc", "nvcc", None, frozenset({"llama_cpp"}), optional=True, purpose="CUDA build of llama.cpp (CPU build is used otherwise)"),
)


class EnvironmentProfile:
    name = "base"

    def __init__(self, probe: HostProbe) -> None:
        self.probe = probe

    def requirements(self, components: set[str]) -> list[HostRequirement]:
        return [r for r in BASE_REQUIREMENTS if r.needed_for & components]

    def _sudo(self, argv: list[str]) -> list[str]:
        return argv if self.probe.is_root() else ["sudo", "-n", *argv]

    def assess(self, components: set[str], config: dict[str, Any]) -> HostAssessment:
        assessment = HostAssessment(self.name)
        for req in self.requirements(components):
            if self.probe.which(req.command):
                continue
            (assessment.optional_missing if req.optional else assessment.missing).append(req)
        apt_packages = sorted({r.apt_package for r in assessment.missing if r.apt_package})
        if apt_packages:
            assessment.actions.append(
                HostAction(
                    f"Install host packages: {' '.join(apt_packages)}",
                    self._sudo(["apt-get", "install", "-y", "--no-install-recommends", *apt_packages]),
                    privileged=True,
                    flag="environment.allow_system_packages",
                )
            )
        self.profile_specific(assessment, components, config)
        return assessment

    def profile_specific(self, assessment: HostAssessment, components: set[str], config: dict[str, Any]) -> None:
        raise NotImplementedError

    def torch_plan(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"source": "pip", "backend": config["environment"]["torch_backend"]}
