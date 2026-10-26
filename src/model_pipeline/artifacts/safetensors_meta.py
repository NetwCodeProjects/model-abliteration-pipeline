"""Pure-Python safetensors header access (no torch): tensor inventories, dtypes, grafting.

Used to (a) record the *actual* saved precision of checkpoints instead of assuming BF16,
(b) diff tensor sets between the source and a transformed checkpoint (e.g. Heretic's merged
export drops ``mtp.*`` tensors of Qwen3.5-family models), and (c) optionally graft tensors that
are absent from a candidate verbatim from the untouched source.
"""

from __future__ import annotations

import json
import os
import struct
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..errors import ExitCode, PipelineError
from ..fsutil import atomic_write_json, read_json

INDEX_NAME = "model.safetensors.index.json"
FLOAT_DTYPES = {"BF16": "bf16", "F16": "f16", "F32": "f32", "F64": "f64", "F8_E4M3": "fp8_e4m3", "F8_E5M2": "fp8_e5m2"}


def read_header(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise PipelineError(ExitCode.VERIFY, "E_SAFETENSORS_CORRUPT", f"{path}: truncated header")
        (size,) = struct.unpack("<Q", raw)
        if size > 100 * 1024 * 1024:
            raise PipelineError(ExitCode.VERIFY, "E_SAFETENSORS_CORRUPT", f"{path}: implausible header size {size}")
        header = json.loads(handle.read(size))
    header["__header_size__"] = size
    return header


def tensor_map(model_dir: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """{tensor_name: {"file", "dtype", "shape", "data_offsets"}} across all shards."""
    root = Path(model_dir)
    files = sorted(p for p in root.glob("*.safetensors"))
    if not files:
        raise PipelineError(ExitCode.VERIFY, "E_NO_SAFETENSORS", f"No *.safetensors files in {root}")
    tensors: dict[str, dict[str, Any]] = {}
    for file in files:
        header = read_header(file)
        for name, meta in header.items():
            if name.startswith("__"):
                continue
            tensors[name] = {
                "file": file.name,
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "data_offsets": meta["data_offsets"],
                "header_size": header["__header_size__"],
            }
    return tensors


def dtype_summary(tensors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(meta["dtype"] for meta in tensors.values())
    params: Counter[str] = Counter()
    for meta in tensors.values():
        n = 1
        for dim in meta["shape"]:
            n *= dim
        params[meta["dtype"]] += n
    float_params = {k: v for k, v in params.items() if k in FLOAT_DTYPES}
    dominant = max(float_params, key=float_params.get) if float_params else None
    floats = sorted(k for k in counts if k in FLOAT_DTYPES)
    if len(floats) == 1:
        precision = FLOAT_DTYPES[floats[0]]
    elif dominant:
        precision = f"mixed:{'+'.join(FLOAT_DTYPES[f] for f in floats)}(dominant {FLOAT_DTYPES[dominant]})"
    else:
        precision = "unknown"
    return {"tensor_count": len(tensors), "dtype_tensor_counts": dict(counts), "dtype_param_counts": dict(params), "precision": precision}


def tensor_diff(source: dict[str, dict[str, Any]], other: dict[str, dict[str, Any]]) -> dict[str, Any]:
    missing = sorted(set(source) - set(other))
    extra = sorted(set(other) - set(source))
    shape_changes = sorted(n for n in set(source) & set(other) if source[n]["shape"] != other[n]["shape"])
    prefixes = Counter(name.split(".")[0] for name in missing)
    return {
        "missing_from_target": missing,
        "missing_count": len(missing),
        "missing_prefixes": dict(prefixes),
        "extra_in_target": extra,
        "shape_changes": shape_changes,
    }


def graft_tensors(source_dir: Path, target_dir: Path, prefixes: Iterable[str], shard_name: str = "model-grafted-source-tensors.safetensors") -> dict[str, Any]:
    """Copy tensors whose names start with ``prefixes`` and are absent in ``target_dir`` from the source.

    Bytes are copied verbatim (no dtype conversion). A new shard and a *new* index file are written
    (atomic replace creates new inodes, so hardlinked files shared with a candidate stay untouched).
    """
    prefixes = tuple(prefixes)
    src_map = tensor_map(source_dir)
    dst_map = tensor_map(target_dir)
    wanted = sorted(n for n in src_map if n.startswith(prefixes) and n not in dst_map)
    if not wanted:
        return {"grafted": [], "shard": None}
    header: dict[str, Any] = {"__metadata__": {"format": "pt", "grafted_from": "source"}}
    offset = 0
    for name in wanted:
        meta = src_map[name]
        start, end = meta["data_offsets"]
        header[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "data_offsets": [offset, offset + (end - start)]}
        offset += end - start
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    shard_path = target_dir / shard_name
    tmp = shard_path.with_name(shard_path.name + ".tmp")
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for name in wanted:
            meta = src_map[name]
            start, end = meta["data_offsets"]
            with open(source_dir / meta["file"], "rb") as handle:
                handle.seek(8 + meta["header_size"] + start)
                remaining = end - start
                while remaining:
                    block = handle.read(min(remaining, 64 * 1024 * 1024))
                    if not block:
                        raise PipelineError(ExitCode.VERIFY, "E_SAFETENSORS_CORRUPT", f"Unexpected EOF reading {name}")
                    out.write(block)
                    remaining -= len(block)
    os.replace(tmp, shard_path)
    index_path = target_dir / INDEX_NAME
    if index_path.exists():
        index = read_json(index_path)
    else:
        index = {"metadata": {}, "weight_map": {n: m["file"] for n, m in dst_map.items()}}
    index.setdefault("metadata", {})
    total = int(index["metadata"].get("total_size", 0)) + offset
    index["metadata"]["total_size"] = total
    for name in wanted:
        index["weight_map"][name] = shard_name
    if index_path.exists():
        index_path.unlink()  # break a potential hardlink before writing the new index
    atomic_write_json(index_path, index)
    return {"grafted": wanted, "shard": shard_name, "bytes": offset}


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    """Minimal writer used by tests/fixtures: {name: (dtype, shape, raw_bytes)}."""
    header: dict[str, Any] = {}
    offset = 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for _dtype, _shape, data in tensors.values():
            out.write(data)
