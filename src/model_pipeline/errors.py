"""Stable exit codes and the single exception type used for expected failures.

Every failure the platform anticipates is raised as a ``PipelineError`` carrying a
stable ``error_id`` (documented in docs/cli.md) and an exit code. Anything else is a
bug and exits with ``ExitCode.INTERNAL``.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any


class ExitCode(IntEnum):
    OK = 0
    INTERNAL = 1
    USAGE = 2
    CONFIG = 3
    PREFLIGHT = 4
    ENVIRONMENT = 5
    ACQUIRE = 6
    DEPENDENCY = 7
    ENGINE = 8
    GATE = 9
    QUANTIZE = 10
    VERIFY = 11
    LOCKED = 12
    UNSUPPORTED = 14
    CONFIRMATION_REQUIRED = 15
    PARTIAL = 20
    INTERRUPTED = 130


class PipelineError(Exception):
    """An anticipated failure with a stable identifier and actionable hint."""

    def __init__(
        self,
        code: ExitCode,
        error_id: str,
        message: str,
        hint: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.error_id = error_id
        self.message = message
        self.hint = hint
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "exit_code": int(self.code),
            "error_id": self.error_id,
            "message": self.message,
            "hint": self.hint,
            "details": self.details,
        }

    def __str__(self) -> str:
        text = f"[{self.error_id}] {self.message}"
        if self.hint:
            text += f"\n  hint: {self.hint}"
        return text


class Interrupted(PipelineError):
    def __init__(self, message: str = "Run interrupted by signal") -> None:
        super().__init__(
            ExitCode.INTERRUPTED,
            "E_INTERRUPTED",
            message,
            hint="Resume later with ./main.sh --resume RUN_ID; partial outputs are never marked complete.",
        )
