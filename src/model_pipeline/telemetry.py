"""Structured run logging (JSONL + readable log + console), secret redaction, resource sampling.

Measurements are factual samples; the sampler records only what it could observe and leaves
fields as null otherwise. Cost figures elsewhere are always labelled as estimates.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .fsutil import disk_free_bytes

LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40}
SECRET_ENV_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|API_KEY|ACCESS_KEY|PRIVATE_KEY)", re.I)
HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Redactor:
    def __init__(self, extra_env_names: tuple[str, ...] = ()) -> None:
        secrets = set()
        for name, value in os.environ.items():
            if value and len(value) >= 8 and (SECRET_ENV_RE.search(name) or name in extra_env_names):
                secrets.add(value)
        self._secrets = sorted(secrets, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, "***REDACTED***")
        return HF_TOKEN_RE.sub("hf_***REDACTED***", text)


class EventLogger:
    """Writes ``events.jsonl`` and ``run.log`` under a run's logs directory and echoes to stderr."""

    def __init__(self, log_dir: Path | None, redactor: Redactor, console_level: str = "info", stream=None) -> None:
        self.log_dir = log_dir
        self.redact = redactor
        self.console_level = LEVELS.get(console_level, 20)
        self.stream = stream or sys.stderr
        self._lock = threading.Lock()
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)

    def attach(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

    def log(self, level: str, event: str, message: str, **fields: Any) -> None:
        stamp = utc_now()
        text = self.redact(message)
        record = {"ts": stamp, "level": level, "event": event, "message": text}
        if fields:
            record["fields"] = json.loads(self.redact(json.dumps(fields, default=str)))
        with self._lock:
            if self.log_dir is not None:
                with open(self.log_dir / "events.jsonl", "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                with open(self.log_dir / "run.log", "a", encoding="utf-8") as handle:
                    handle.write(f"{stamp} {level.upper():7s} {event}: {text}\n")
            if LEVELS.get(level, 20) >= self.console_level:
                prefix = {"warning": "WARN ", "error": "ERROR"}.get(level, "")
                self.stream.write(f"{prefix + ' ' if prefix else ''}{text}\n")
                self.stream.flush()

    def info(self, event: str, message: str, **fields: Any) -> None:
        self.log("info", event, message, **fields)

    def warning(self, event: str, message: str, **fields: Any) -> None:
        self.log("warning", event, message, **fields)

    def error(self, event: str, message: str, **fields: Any) -> None:
        self.log("error", event, message, **fields)

    def debug(self, event: str, message: str, **fields: Any) -> None:
        self.log("debug", event, message, **fields)


def query_gpus(timeout: float = 10.0) -> list[dict[str, Any]] | None:
    """Sample GPU memory/utilization via nvidia-smi; None when unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append({"index": int(parts[0]), "memory_used_mib": float(parts[1]), "memory_total_mib": float(parts[2]), "utilization_pct": float(parts[3])})
        except ValueError:
            continue
    return gpus


def host_memory_used_bytes() -> int | None:
    try:
        import psutil  # noqa: PLC0415

        vm = psutil.virtual_memory()
        return int(vm.total - vm.available)
    except ImportError:
        pass
    try:
        info = {}
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                info[key] = int(value.strip().split()[0]) * 1024
        return info["MemTotal"] - info.get("MemAvailable", info.get("MemFree", 0))
    except (OSError, KeyError, ValueError):
        return None


class ResourceSampler:
    """Background sampler writing JSONL samples and keeping peaks for stage receipts."""

    def __init__(self, out_file: Path | None, disk_path: Path, interval: float = 15.0, gpu_query=query_gpus) -> None:
        self.out_file = out_file
        self.disk_path = disk_path
        self.interval = interval
        self.gpu_query = gpu_query
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples = 0
        self.peak_gpu_mem_mib: float | None = None
        self.peak_gpu_util_pct: float | None = None
        self.peak_host_mem_bytes: int | None = None
        self.min_disk_free_bytes: int | None = None
        self.started = time.monotonic()

    def _sample(self) -> None:
        gpus = self.gpu_query()
        host = host_memory_used_bytes()
        try:
            disk = disk_free_bytes(self.disk_path)
        except OSError:
            disk = None
        if gpus:
            used = max(g["memory_used_mib"] for g in gpus)
            util = max(g["utilization_pct"] for g in gpus)
            self.peak_gpu_mem_mib = max(self.peak_gpu_mem_mib or 0.0, used)
            self.peak_gpu_util_pct = max(self.peak_gpu_util_pct or 0.0, util)
        if host is not None:
            self.peak_host_mem_bytes = max(self.peak_host_mem_bytes or 0, host)
        if disk is not None:
            self.min_disk_free_bytes = disk if self.min_disk_free_bytes is None else min(self.min_disk_free_bytes, disk)
        self.samples += 1
        if self.out_file is not None:
            self.out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"ts": utc_now(), "gpus": gpus, "host_mem_used_bytes": host, "disk_free_bytes": disk}) + "\n")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:  # noqa: BLE001 - sampling must never break a stage
                pass
            self._stop.wait(self.interval)

    def start(self) -> ResourceSampler:
        self._thread = threading.Thread(target=self._loop, name="resource-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 5)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            "measured": True,
            "samples": self.samples,
            "interval_seconds": self.interval,
            "peak_gpu_memory_mib": self.peak_gpu_mem_mib,
            "peak_gpu_utilization_pct": self.peak_gpu_util_pct,
            "peak_host_memory_bytes": self.peak_host_mem_bytes,
            "min_disk_free_bytes": self.min_disk_free_bytes,
            "note": "Sampled values; peaks between samples may be missed. null = not observable on this host.",
        }
