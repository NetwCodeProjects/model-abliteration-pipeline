"""Filesystem primitives: atomic writes, hashing, inventories, locks, path safety.

Everything here is stdlib-only so the control plane works before any environment
has been provisioned (and inside ``--dry-run``, where only read helpers are used).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import stat
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ExitCode, PipelineError

HASH_CHUNK = 8 * 1024 * 1024


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: str | os.PathLike[str]) -> str:
    """Git object id of a file's content, as reported by the Hugging Face API for non-LFS files."""
    size = os.path.getsize(path)
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode())
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike[str], obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def ensure_within(base: str | os.PathLike[str], candidate: str | os.PathLike[str]) -> Path:
    """Resolve ``candidate`` and refuse anything that escapes ``base`` (path traversal guard)."""
    base_path = Path(base).resolve()
    resolved = Path(candidate).resolve()
    if resolved != base_path and base_path not in resolved.parents:
        raise PipelineError(
            ExitCode.CONFIG,
            "E_PATH_ESCAPE",
            f"Path {candidate} resolves outside of {base_path}",
            hint="Use paths inside the workspace; '..' segments and symlinks out of the workspace are rejected.",
        )
    return resolved


def safe_component(name: str) -> str:
    """Make a string usable as a single path component."""
    cleaned = "".join(c if (c.isalnum() or c in "-_.@") else "-" for c in name)
    cleaned = cleaned.strip(".")
    return cleaned or "unnamed"


@dataclass(frozen=True)
class InventoryEntry:
    relative_path: str
    size_bytes: int
    sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"relative_path": self.relative_path, "size_bytes": self.size_bytes, "sha256": self.sha256}


def iter_files(root: str | os.PathLike[str], exclude_names: Iterable[str] = ()) -> list[Path]:
    root_path = Path(root)
    excluded = set(exclude_names)
    if root_path.is_file():
        return [root_path]
    files = []
    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in excluded and not d.startswith(".cache"))
        for name in sorted(filenames):
            if name in excluded:
                continue
            files.append(Path(dirpath) / name)
    return files


def file_inventory(
    root: str | os.PathLike[str],
    *,
    hash_files: bool = True,
    exclude_names: Iterable[str] = (),
) -> list[InventoryEntry]:
    """Inventory every file under ``root`` (symlinks are followed for content, e.g. HF cache snapshots)."""
    root_path = Path(root)
    entries = []
    for path in iter_files(root_path, exclude_names):
        rel = path.name if root_path.is_file() else path.relative_to(root_path).as_posix()
        size = path.stat().st_size
        entries.append(InventoryEntry(rel, size, sha256_file(path) if hash_files else None))
    return sorted(entries, key=lambda e: e.relative_path)


def inventory_digest(entries: Iterable[InventoryEntry | dict[str, Any]]) -> str:
    rows = []
    for entry in entries:
        data = entry.to_dict() if isinstance(entry, InventoryEntry) else entry
        rows.append([data["relative_path"], data["size_bytes"], data["sha256"]])
    return sha256_obj(sorted(rows))


def quick_fingerprint(root: str | os.PathLike[str]) -> dict[str, list[int]]:
    """Cheap immutability fingerprint: size and mtime_ns of every file."""
    root_path = Path(root)
    result: dict[str, list[int]] = {}
    for path in iter_files(root_path):
        st = path.stat()
        rel = path.name if root_path.is_file() else path.relative_to(root_path).as_posix()
        result[rel] = [st.st_size, st.st_mtime_ns]
    return result


def nearest_existing(path: str | os.PathLike[str]) -> Path:
    current = Path(path).absolute()
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def disk_free_bytes(path: str | os.PathLike[str]) -> int:
    return shutil.disk_usage(nearest_existing(path)).free


def human_bytes(value: float | int | None) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{amount:.1f} TiB"


def make_read_only(root: str | os.PathLike[str]) -> int:
    """Clear write bits on every file under ``root``; returns the number of files touched."""
    count = 0
    for path in iter_files(root):
        real = Path(os.path.realpath(path))
        mode = real.stat().st_mode
        real.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        count += 1
    return count


def link_or_copy_tree(src: str | os.PathLike[str], dst: str | os.PathLike[str], strategy: str = "auto") -> str:
    """Materialize ``src`` at ``dst`` file by file. Returns the storage semantics actually used.

    ``hardlink`` shares inodes (no extra space; the source must then never be modified in place,
    which the platform guarantees by only ever writing new files via atomic rename).
    """
    src_path, dst_path = Path(src), Path(dst)
    if dst_path.exists():
        raise PipelineError(ExitCode.INTERNAL, "E_DEST_EXISTS", f"Destination already exists: {dst_path}")
    used = set()
    for path in iter_files(src_path):
        rel = path.relative_to(src_path)
        target = dst_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        real = os.path.realpath(path)
        if strategy in ("auto", "hardlink"):
            try:
                os.link(real, target)
                used.add("hardlink")
                continue
            except OSError:
                if strategy == "hardlink":
                    raise
        shutil.copy2(real, target)
        used.add("copy")
    if not used:
        return "empty"
    return used.pop() if len(used) == 1 else "mixed(hardlink+copy)"


def remove_tree(path: str | os.PathLike[str]) -> None:
    def _onerror(func, failed_path, _exc):  # read-only files on some platforms
        os.chmod(failed_path, stat.S_IWUSR | stat.S_IRUSR)
        func(failed_path)

    if Path(path).exists():
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_onerror)
        else:  # pragma: no cover - Python 3.10/3.11
            shutil.rmtree(path, onerror=_onerror)


class LockHeld(PipelineError):
    pass


class FileLock:
    """Portable exclusive lock based on O_EXCL lock files with stale-owner detection.

    The lock file stores host and pid; a lock left behind by a dead process on the same
    host is reclaimed, a lock owned by another host is never broken automatically.
    """

    def __init__(self, path: str | os.PathLike[str], purpose: str) -> None:
        self.path = Path(path)
        self.purpose = purpose
        self._held = False

    def _owner(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if os.name == "nt":
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def acquire(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json({"host": socket.gethostname(), "pid": os.getpid(), "since": time.time(), "purpose": self.purpose}).encode()
        for _attempt in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                owner = self._owner()
                if owner and owner.get("host") == socket.gethostname() and not self._pid_alive(int(owner.get("pid", -1))):
                    self.path.unlink(missing_ok=True)
                    continue
                raise LockHeld(
                    ExitCode.LOCKED,
                    "E_LOCK_HELD",
                    f"{self.purpose} is locked by another process: {owner}",
                    hint=f"Wait for the other run to finish. If you are certain it is dead, delete {self.path}.",
                ) from None
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
            self._held = True
            return self
        raise LockHeld(ExitCode.LOCKED, "E_LOCK_HELD", f"Could not acquire lock {self.path}")

    def release(self) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False

    def __enter__(self) -> FileLock:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()
