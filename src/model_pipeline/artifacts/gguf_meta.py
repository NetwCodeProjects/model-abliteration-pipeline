"""Pure-Python GGUF metadata reader (header + key/value section only, tensors are skipped).

Used by stage 09 to check that converted files carry architecture, tokenizer and chat template
metadata without importing llama.cpp's gguf package into the control plane.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

from ..errors import ExitCode, PipelineError

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
STRING, ARRAY = 8, 9
ARRAY_PREVIEW = 8


def _read(handle: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = handle.read(size)
    if len(data) != size:
        raise PipelineError(ExitCode.VERIFY, "E_GGUF_CORRUPT", "Unexpected end of GGUF metadata")
    return struct.unpack(fmt, data)[0]


def _read_string(handle: BinaryIO) -> str:
    length = _read(handle, "<Q")
    if length > 512 * 1024 * 1024:
        raise PipelineError(ExitCode.VERIFY, "E_GGUF_CORRUPT", f"Implausible GGUF string length {length}")
    return handle.read(length).decode("utf-8", errors="replace")


def _read_value(handle: BinaryIO, vtype: int) -> Any:
    if vtype in _SCALARS:
        return _read(handle, _SCALARS[vtype])
    if vtype == STRING:
        return _read_string(handle)
    if vtype == ARRAY:
        etype = _read(handle, "<I")
        count = _read(handle, "<Q")
        preview = []
        for i in range(count):
            value = _read_value(handle, etype)
            if i < ARRAY_PREVIEW:
                preview.append(value)
        return {"array_length": count, "preview": preview}
    raise PipelineError(ExitCode.VERIFY, "E_GGUF_CORRUPT", f"Unknown GGUF value type {vtype}")


def read_metadata(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        if handle.read(4) != b"GGUF":
            raise PipelineError(ExitCode.VERIFY, "E_GGUF_CORRUPT", f"{path} is not a GGUF file (bad magic)")
        version = _read(handle, "<I")
        if version < 2:
            raise PipelineError(ExitCode.VERIFY, "E_GGUF_CORRUPT", f"{path}: unsupported GGUF version {version}")
        tensor_count = _read(handle, "<Q")
        kv_count = _read(handle, "<Q")
        kv: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_string(handle)
            vtype = _read(handle, "<I")
            kv[key] = _read_value(handle, vtype)
    return {"version": version, "tensor_count": tensor_count, "kv": kv}


def write_minimal_gguf(path: str | Path, kv: dict[str, Any]) -> None:
    """Tiny writer for tests (strings, ints, string arrays); writes zero tensors."""

    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    body = b""
    for key, value in kv.items():
        body += s(key)
        if isinstance(value, str):
            body += struct.pack("<I", STRING) + s(value)
        elif isinstance(value, bool):
            body += struct.pack("<I", 7) + struct.pack("<?", value)
        elif isinstance(value, int):
            body += struct.pack("<I", 4) + struct.pack("<I", value)
        elif isinstance(value, list):
            body += struct.pack("<I", ARRAY) + struct.pack("<I", STRING) + struct.pack("<Q", len(value))
            body += b"".join(s(str(v)) for v in value)
        else:
            raise TypeError(type(value))
    with open(path, "wb") as out:
        out.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv)) + body)
