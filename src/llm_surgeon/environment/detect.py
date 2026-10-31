"""Read-only host probing and environment-profile detection.

``HostProbe(allow_exec=False)`` only reads files (used by --dry-run). With ``allow_exec`` it
also runs ``nvidia-smi``/``nvcc`` queries. Lambda Stack is detected from its own packaging
markers — the presence of CUDA alone never implies Lambda Stack.
"""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..errors import ExitCode, PipelineError


class HostProbe:
    def __init__(self, allow_exec: bool = True, root: str = "/") -> None:
        self.allow_exec = allow_exec
        self.root = Path(root)

    def _read(self, rel: str) -> str | None:
        try:
            return (self.root / rel.lstrip("/")).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def _run(self, argv: list[str], timeout: float = 20.0) -> str | None:
        if not self.allow_exec:
            return None
        exe = shutil.which(argv[0])
        if exe is None:
            return None
        try:
            return subprocess.run([exe, *argv[1:]], capture_output=True, text=True, timeout=timeout, check=True, stdin=subprocess.DEVNULL).stdout
        except (subprocess.SubprocessError, OSError):
            return None

    def platform(self) -> str:
        return sys.platform

    def os_release(self) -> dict[str, str]:
        text = self._read("/etc/os-release") or ""
        data = {}
        for line in text.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                data[key.strip()] = value.strip().strip('"')
        return data

    def kernel(self) -> str:
        return platform.release()

    def cpu(self) -> dict[str, Any]:
        model = None
        text = self._read("/proc/cpuinfo") or ""
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
        return {"model": model or platform.processor() or None, "logical_cpus": os.cpu_count()}

    def memory_total_bytes(self) -> int | None:
        text = self._read("/proc/meminfo")
        if text:
            match = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, re.M)
            if match:
                return int(match.group(1)) * 1024
        try:
            import psutil  # noqa: PLC0415

            return int(psutil.virtual_memory().total)
        except ImportError:
            return None

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def gpus(self) -> list[dict[str, Any]] | None:
        out = self._run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,compute_cap,uuid,ecc.errors.uncorrected.volatile.total",
                "--format=csv,noheader,nounits",
            ]
        )
        if out is None:
            return None
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            ecc = parts[6] if len(parts) > 6 else None
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": float(parts[2]) if parts[2].replace(".", "", 1).isdigit() else None,
                    "driver_version": parts[3],
                    "compute_capability": parts[4],
                    "uuid": parts[5],
                    "ecc_uncorrected_volatile": None if ecc in (None, "[N/A]", "N/A") else ecc,
                }
            )
        return gpus

    def cuda_driver_version(self) -> str | None:
        out = self._run(["nvidia-smi"])
        if out:
            match = re.search(r"CUDA Version:\s*([0-9.]+)", out)
            if match:
                return match.group(1)
        return None

    def nvcc_version(self) -> str | None:
        out = self._run(["nvcc", "--version"])
        if out:
            match = re.search(r"release ([0-9.]+)", out)
            if match:
                return match.group(1)
        return None

    def dpkg_packages(self, prefixes: tuple[str, ...]) -> list[str]:
        text = self._read("/var/lib/dpkg/status") or ""
        found = []
        for block in text.split("\n\n"):
            name = re.search(r"^Package:\s*(\S+)", block, re.M)
            status = re.search(r"^Status:\s*(.+)$", block, re.M)
            if name and status and "installed" in status.group(1) and "not-installed" not in status.group(1):
                if name.group(1).startswith(prefixes):
                    found.append(name.group(1))
        return sorted(found)

    def apt_source_files(self) -> list[str]:
        base = self.root / "etc/apt/sources.list.d"
        return sorted(glob.glob(str(base / "*")))

    def mounts(self) -> list[dict[str, str]]:
        text = self._read("/proc/mounts") or ""
        rows = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                rows.append({"device": parts[0], "mountpoint": parts[1].replace("\\040", " "), "fstype": parts[2]})
        return rows

    def mount_for(self, path: str | Path) -> dict[str, str] | None:
        target = str(Path(path).absolute())
        best = None
        for mount in self.mounts():
            mp = mount["mountpoint"]
            if target == mp or target.startswith(mp.rstrip("/") + "/") or mp == "/":
                if best is None or len(mp) > len(best["mountpoint"]):
                    best = mount
        return best

    def is_root(self) -> bool:
        return hasattr(os, "geteuid") and os.geteuid() == 0


NETWORK_FS = {"nfs", "nfs4", "cifs", "smb3", "fuse.sshfs", "virtiofs", "9p", "lustre", "beegfs", "gpfs", "ceph", "fuse.juicefs", "fuse.s3fs"}


def classify_storage(mount: dict[str, str] | None) -> dict[str, Any]:
    if mount is None:
        return {"fstype": None, "kind": "unknown", "note": "Mount information unavailable."}
    fstype = mount["fstype"]
    if fstype in NETWORK_FS or fstype.startswith("fuse."):
        kind, note = "network-or-persistent-candidate", "Network/attached filesystem; often survives instance termination (verify with your provider)."
    elif fstype in {"tmpfs", "ramfs", "overlay"}:
        kind, note = "ephemeral", "Memory/overlay filesystem; contents are lost on restart."
    else:
        kind, note = "instance-local", "Local disk; on cloud instances this is usually lost when the instance is terminated."
    return {"fstype": fstype, "mountpoint": mount["mountpoint"], "kind": kind, "note": note}


def lambda_stack_markers(probe: HostProbe) -> list[str]:
    markers = [f"dpkg:{p}" for p in probe.dpkg_packages(("lambda-stack", "lambda-server"))]
    for source in probe.apt_source_files():
        if "lambda" in Path(source).name.lower():
            markers.append(f"apt-source:{Path(source).name}")
    return markers


def detect_profile(probe: HostProbe, requested: str) -> tuple[str, str]:
    """Return (profile, reason). ``requested`` is auto | lambda-stack | ubuntu."""
    release = probe.os_release()
    is_ubuntu = release.get("ID") == "ubuntu"
    if requested != "auto":
        reason = f"selected explicitly ({requested})"
        if not is_ubuntu:
            reason += f"; WARNING: host OS is {release.get('PRETTY_NAME') or probe.platform()}, profiles target Ubuntu"
        return requested, reason
    if not probe.platform().startswith("linux"):
        raise PipelineError(
            ExitCode.PREFLIGHT,
            "E_ENV_UNSUPPORTED_PLATFORM",
            f"Automatic profile detection needs Linux; this host is {probe.platform()}",
            hint="Run on Ubuntu (Lambda Stack or plain). --help and --dry-run work everywhere.",
        )
    markers = lambda_stack_markers(probe)
    if markers and is_ubuntu:
        return "lambda-stack", f"Lambda Stack markers found: {', '.join(markers)}"
    if is_ubuntu:
        return "ubuntu", "Ubuntu without Lambda Stack packaging markers (CUDA alone does not imply Lambda Stack)"
    raise PipelineError(
        ExitCode.PREFLIGHT,
        "E_ENV_AMBIGUOUS",
        f"Cannot choose an environment profile automatically for {release.get('PRETTY_NAME', 'this OS')}",
        hint="Pass --lambda-stack or --ubuntu explicitly (both target Ubuntu hosts).",
    )


def cuda_arch_from_capability(capability: str | None) -> str | None:
    """'10.0' -> '100' (B200), '9.0' -> '90' (H100)."""
    if not capability:
        return None
    match = re.match(r"^(\d+)\.(\d+)$", capability.strip())
    if not match:
        return None
    return f"{match.group(1)}{match.group(2)}"
