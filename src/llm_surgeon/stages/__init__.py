"""Stable stage registry. IDs 00-10 and names are public contracts (docs/stages.md)."""

from __future__ import annotations

from ..errors import ExitCode, PipelineError
from .base import Stage
from .stage00_preflight import PreflightStage
from .stage01_environment import EnvironmentStage
from .stage02_acquire import AcquireStage
from .stage03_baselines import BaselinesStage
from .stage04_prepare import PrepareEngineStage
from .stage05_abliterate import AbliterateStage
from .stage06_qualify import QualifyStage
from .stage07_select import SelectMasterStage
from .stage08_quantize import QuantizeStage
from .stage09_verify import VerifyExportsStage
from .stage10_report import ReportExportStage

STAGES: list[Stage] = [
    PreflightStage(),
    EnvironmentStage(),
    AcquireStage(),
    BaselinesStage(),
    PrepareEngineStage(),
    AbliterateStage(),
    QualifyStage(),
    SelectMasterStage(),
    QuantizeStage(),
    VerifyExportsStage(),
    ReportExportStage(),
]


def stage_by_ref(ref: str | int) -> Stage:
    text = str(ref).strip().lower()
    for stage in STAGES:
        if text in (stage.id, stage.name) or (text.isdigit() and int(text) == int(stage.id)):
            return stage
    raise PipelineError(
        ExitCode.USAGE,
        "E_UNKNOWN_STAGE",
        f"Unknown stage {ref!r}",
        hint="Use a number 0-10 or a name: " + ", ".join(f"{s.id}={s.name}" for s in STAGES),
    )
