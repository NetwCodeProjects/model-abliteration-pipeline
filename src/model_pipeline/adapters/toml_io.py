"""TOML read (tomllib / tomli) and a small, strict TOML writer for generated engine configs."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from ..errors import ExitCode, PipelineError

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def loads(text: str, source: str = "<string>") -> dict[str, Any]:
    try:
        import tomllib  # type: ignore[import-not-found]  # Python 3.11+
    except ImportError:  # pragma: no cover - Python 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError as exc:
            raise PipelineError(
                ExitCode.ENVIRONMENT,
                "E_CONTROL_ENV_INCOMPLETE",
                "Reading TOML on Python 3.10 needs the 'tomli' package",
                hint="Bootstrap the control environment (.venv).",
            ) from exc
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise PipelineError(ExitCode.CONFIG, "E_TOML_INVALID", f"Invalid TOML in {source}: {exc}") from exc


def load(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PipelineError(ExitCode.CONFIG, "E_TOML_NOT_FOUND", f"Cannot read TOML file {path}: {exc}") from exc
    return loads(text, str(path))


def _key(key: str) -> str:
    return key if _BARE_KEY.match(key) else json.dumps(key, ensure_ascii=False)


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("TOML writer refuses NaN/inf")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_value(v)}" for k, v in value.items() if v is not None) + " }"
    raise TypeError(f"Unsupported TOML value type {type(value).__name__}")


def dumps(data: dict[str, Any]) -> str:
    """Scalars and inline arrays first, then [tables] and [[arrays of tables]], recursively. None is omitted."""
    out: list[str] = []

    def emit(table: dict[str, Any], prefix: str) -> None:
        scalars, tables, table_arrays = [], [], []
        for key, value in table.items():
            if value is None:
                continue
            if isinstance(value, dict):
                tables.append((key, value))
            elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value) and prefix:
                table_arrays.append((key, value))
            else:
                scalars.append((key, value))
        for key, value in scalars:
            out.append(f"{_key(key)} = {_value(value)}")
        for key, value in tables:
            name = f"{prefix}.{_key(key)}" if prefix else _key(key)
            out.append("")
            out.append(f"[{name}]")
            emit(value, name)
        for key, items in table_arrays:
            name = f"{prefix}.{_key(key)}"
            for item in items:
                out.append("")
                out.append(f"[[{name}]]")
                emit(item, name)

    emit(data, "")
    return "\n".join(out).lstrip("\n") + "\n"
