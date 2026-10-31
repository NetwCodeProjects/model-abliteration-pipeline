"""Bounded, logged, signal-aware execution of external tools (never through a shell).

Children run in their own process group with stdin closed (an unexpected interactive prompt
fails fast instead of hanging). A *budget* deadline sends SIGINT first — Heretic treats that as
"stop optimizing and continue to export" — then SIGTERM after a grace period, then SIGKILL.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ExitCode, PipelineError
from .telemetry import Redactor, utc_now


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int | None
    started_utc: str
    ended_utc: str
    duration_s: float
    stdout_log: str
    stderr_log: str
    budget_stopped: bool = False
    interrupted: bool = False
    timed_out: bool = False
    killed: bool = False
    tail: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not (self.interrupted or self.timed_out or self.killed)

    def to_dict(self) -> dict:
        return {
            "argv": self.argv,
            "returncode": self.returncode,
            "started_utc": self.started_utc,
            "ended_utc": self.ended_utc,
            "duration_s": round(self.duration_s, 3),
            "stdout_log": self.stdout_log,
            "stderr_log": self.stderr_log,
            "budget_stopped": self.budget_stopped,
            "interrupted": self.interrupted,
            "timed_out": self.timed_out,
            "killed": self.killed,
        }


class CommandRunner:
    def __init__(self, redactor: Redactor, interrupt_event: threading.Event, echo: Callable[[str], None] | None = None) -> None:
        self.redact = redactor
        self.interrupt_event = interrupt_event
        self.echo = echo

    @staticmethod
    def _signal_group(proc: subprocess.Popen, sig: int) -> None:
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                if sig == signal.SIGINT:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                elif sig == getattr(signal, "SIGKILL", None):
                    proc.kill()
                else:
                    proc.terminate()
            else:
                os.killpg(proc.pid, sig)
        except (ProcessLookupError, OSError):
            pass

    def run(
        self,
        argv: Sequence[str | os.PathLike[str]],
        *,
        cwd: str | os.PathLike[str] | None,
        env: dict[str, str] | None,
        log_dir: Path,
        name: str,
        budget_seconds: float | None = None,
        grace_seconds: float = 600.0,
        kill_after_seconds: float = 60.0,
        timeout_seconds: float | None = None,
        on_line: Callable[[str, str], None] | None = None,
        echo: bool = False,
    ) -> CommandResult:
        args = [str(a) for a in argv]
        log_dir.mkdir(parents=True, exist_ok=True)
        out_path, err_path = log_dir / f"{name}.stdout.log", log_dir / f"{name}.stderr.log"
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            kwargs["start_new_session"] = True
        started_wall, started = utc_now(), time.monotonic()
        try:
            proc = subprocess.Popen(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
        except FileNotFoundError as exc:
            raise PipelineError(
                ExitCode.ENVIRONMENT,
                "E_EXECUTABLE_MISSING",
                f"Executable not found: {args[0]}",
                hint="Re-run stage 01 (environment) to (re)provision tools.",
            ) from exc
        tail: list[str] = []
        tail_lock = threading.Lock()

        def pump(stream, path: Path, label: str) -> None:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(f"# {utc_now()} $ {self.redact(' '.join(args))}\n")
                for raw in iter(stream.readline, b""):
                    line = self.redact(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
                    handle.write(line + "\n")
                    handle.flush()
                    with tail_lock:
                        tail.append(line)
                        del tail[:-60]
                    if on_line is not None:
                        try:
                            on_line(label, line)
                        except Exception:  # noqa: BLE001 - parsing callbacks must not kill the pump
                            pass
                    if echo and self.echo:
                        self.echo(line)
            stream.close()

        threads = [
            threading.Thread(target=pump, args=(proc.stdout, out_path, "stdout"), daemon=True),
            threading.Thread(target=pump, args=(proc.stderr, err_path, "stderr"), daemon=True),
        ]
        for thread in threads:
            thread.start()

        budget_stopped = interrupted = timed_out = killed = False
        soft_deadline = started + budget_seconds if budget_seconds else None
        hard_deadline = started + timeout_seconds if timeout_seconds else None
        sigint_at: float | None = None
        sigterm_at: float | None = None
        poll_interval = 0.01
        while proc.poll() is None:
            now = time.monotonic()
            if sigint_at is None:
                if self.interrupt_event.is_set():
                    interrupted = True
                    sigint_at = now
                    self._signal_group(proc, signal.SIGINT)
                elif soft_deadline is not None and now >= soft_deadline:
                    budget_stopped = True
                    sigint_at = now
                    self._signal_group(proc, signal.SIGINT)
            elif sigterm_at is None and (now - sigint_at >= grace_seconds or (interrupted and now - sigint_at >= 30)):
                sigterm_at = now
                self._signal_group(proc, signal.SIGTERM)
            elif sigterm_at is not None and now - sigterm_at >= kill_after_seconds:
                killed = True
                self._signal_group(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
            if hard_deadline is not None and now >= hard_deadline and not timed_out:
                timed_out = True
                self._signal_group(proc, signal.SIGTERM)
                sigterm_at = sigterm_at or now
            if self.interrupt_event.is_set() and budget_stopped and not interrupted:
                # A user interrupt during a budget-stop shutdown escalates immediately.
                interrupted = True
                sigterm_at = now
                self._signal_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=poll_interval)
            except subprocess.TimeoutExpired:
                pass
            poll_interval = min(0.5, poll_interval * 1.5)
        for thread in threads:
            thread.join(timeout=10)
        ended = time.monotonic()
        return CommandResult(
            argv=[self.redact(a) for a in args],
            returncode=proc.returncode,
            started_utc=started_wall,
            ended_utc=utc_now(),
            duration_s=ended - started,
            stdout_log=str(out_path),
            stderr_log=str(err_path),
            budget_stopped=budget_stopped,
            interrupted=interrupted,
            timed_out=timed_out,
            killed=killed,
            tail=list(tail),
        )


def child_env(base: dict[str, str] | None = None, **overrides: str | None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def python_in_venv(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def bin_in_venv(venv: Path, name: str) -> Path:
    return venv / ("Scripts" if sys.platform == "win32" else "bin") / (name + (".exe" if sys.platform == "win32" else ""))
