"""Run lifecycle: planning, resume/fork, reuse decisions, locking, signals, dry-run, status."""

from __future__ import annotations

import os
import secrets
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .adapters.hf_hub import HubClient
from .artifacts.manifest import ArtifactRegistry
from .config import (
    COMPATIBILITY_NOTICE,
    PROJECT_ROOT,
    ResolvedConfig,
    deep_merge,
    diff_configs,
    dump_yaml,
    finalize,
    overrides_to_tree,
)
from .environment.components import ComponentManager, find_uv
from .environment.detect import HostProbe
from .errors import ExitCode, Interrupted, PipelineError
from .fsutil import FileLock, atomic_write_json, atomic_write_text, read_json, remove_tree, sha256_obj
from .stages import STAGES, stage_by_ref
from .stages.base import (
    BLOCKED,
    FAILED,
    INTERRUPTED,
    RUNNING,
    SKIPPED,
    SUCCESS,
    Stage,
    StageContext,
    base_receipt,
    load_receipt,
    write_receipt,
)
from .subprocesses import CommandRunner, child_env
from .telemetry import EventLogger, Redactor, ResourceSampler, query_gpus, utc_now


@dataclass
class Services:
    """Injection points for things that touch the outside world (replaced by fakes in tests)."""

    probe_factory: Callable[[bool], HostProbe] = lambda allow_exec: HostProbe(allow_exec=allow_exec)
    hub_factory: Callable[[str, str | None], Any] = lambda cache, token: HubClient(cache, token)
    scripts_dir: Path = PROJECT_ROOT / "scripts" / "runtime"
    gpu_query: Callable[[], Any] = query_gpus
    confirm: Callable[[str], bool] | None = None
    uv_executable: str | None = None
    stdout: Any = None


@dataclass
class RunOptions:
    from_stage: str | None = None
    through_stage: str | None = None
    only_stage: str | None = None
    setup_only: bool = False
    rerun: bool = False
    dry_run: bool = False
    resume_run_id: str | None = None
    fork: bool = False
    new_run: bool = False
    yes: bool = False
    config_path: str | None = None
    cli_overrides: dict[str, Any] = field(default_factory=dict)


class Run:
    def __init__(
        self,
        run_id: str,
        workspace: Path,
        config: ResolvedConfig,
        logger: EventLogger,
        services: Services,
        options: RunOptions,
        interrupt_event: threading.Event,
        parent_run_id: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.workspace = workspace
        self.run_dir = workspace / "runs" / run_id
        self.config = config
        self.logger = logger
        self.services = services
        self.options = options
        self.interrupt_event = interrupt_event
        self.parent_run_id = parent_run_id
        self.redactor = logger.redact
        self.runner = CommandRunner(self.redactor, interrupt_event, echo=lambda line: logger.debug("subprocess", line))
        self.registry = ArtifactRegistry(self.run_dir, workspace)
        self._components: ComponentManager | None = None
        self._probe: HostProbe | None = None
        self.state: dict[str, Any] = {}

    # ------------------------------------------------------------------ layout
    @property
    def receipts_dir(self) -> Path:
        return self.run_dir / "receipts"

    @property
    def cache_root(self) -> Path:
        return Path(self.config.get("environment.cache_root"))

    @property
    def tools_dir(self) -> Path:
        return self.workspace / "tools"

    def path(self, *parts: str) -> Path:
        return self.workspace.joinpath(*parts)

    def receipt_path(self, stage: Stage) -> Path:
        return self.receipts_dir / stage.receipt_name

    def load_receipt(self, stage_id: str) -> dict[str, Any] | None:
        return load_receipt(self.receipt_path(stage_by_ref(stage_id)))

    # ---------------------------------------------------------------- services
    @property
    def probe(self) -> HostProbe:
        if self._probe is None:
            self._probe = self.services.probe_factory(True)
        return self._probe

    def hf_token(self) -> str | None:
        return os.environ.get(self.config.get("model.hf_token_env")) or None

    def hub(self) -> HubClient:
        return self.services.hub_factory(str(self.cache_root / "hf" / "hub"), self.hf_token())

    def components(self, log_dir: Path) -> ComponentManager:
        uv = self.services.uv_executable or find_uv(self.config.get("environment.uv_executable"))
        return ComponentManager(
            tools_dir=self.tools_dir,
            runner=self.runner,
            logger=self.logger,
            log_dir=log_dir,
            uv_executable=uv,
            scripts_dir=self.services.scripts_dir,
            cache_root=self.cache_root,
            probe=self.probe,
        )

    def child_env(self, **extra: str | None) -> dict[str, str]:
        overrides: dict[str, str | None] = {
            "HF_HOME": str(self.cache_root / "hf"),
            "HF_HUB_CACHE": str(self.cache_root / "hf" / "hub"),
            "UV_CACHE_DIR": str(self.cache_root / "uv"),
        }
        token = self.hf_token()
        if token:
            overrides["HF_TOKEN"] = token
        device = self.config.get("environment.gpu_device")
        if device and device != "auto":
            overrides["CUDA_VISIBLE_DEVICES"] = device
        overrides.update(extra)
        return child_env(**overrides)

    def tool_env(self, record, **extra: str | None) -> dict[str, str]:
        """Environment for a process of the tool component ``record`` (its venv activated)."""
        return record.activate(self.child_env(**extra))

    def check_interrupt(self) -> None:
        if self.interrupt_event.is_set():
            raise Interrupted()

    # ------------------------------------------------------------------ status
    def status_path(self) -> Path:
        return self.run_dir / "status.json"

    def load_status(self) -> dict[str, Any]:
        if self.status_path().exists():
            return read_json(self.status_path())
        return {
            "run_id": self.run_id,
            "created_utc": utc_now(),
            "platform_version": __version__,
            "parent_run_id": self.parent_run_id,
            "config_hash": self.config.config_hash,
            "stages": {},
            "outcome": RUNNING,
        }

    def update_status(self, stage: Stage | None = None, **fields: Any) -> None:
        status = self.load_status()
        status["updated_utc"] = utc_now()
        if stage is not None:
            entry = status["stages"].setdefault(stage.id, {"name": stage.name})
            entry.update(fields)
        else:
            status.update(fields)
        atomic_write_json(self.status_path(), status)


# --------------------------------------------------------------------------- planning
def selected_stage_ids(options: RunOptions) -> list[str]:
    ids = [s.id for s in STAGES]
    if options.setup_only:
        return ["00", "01"]
    if options.only_stage is not None:
        return [stage_by_ref(options.only_stage).id]
    start = ids.index(stage_by_ref(options.from_stage).id) if options.from_stage is not None else 0
    end = ids.index(stage_by_ref(options.through_stage).id) if options.through_stage is not None else len(ids) - 1
    if start > end:
        raise PipelineError(ExitCode.USAGE, "E_STAGE_RANGE", f"--from {options.from_stage} is after --through {options.through_stage}")
    return ids[start : end + 1]


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def find_resumable_run(workspace: Path, config_hash: str) -> str | None:
    runs_dir = workspace / "runs"
    if not runs_dir.is_dir():
        return None
    candidates = []
    for status_file in runs_dir.glob("*/status.json"):
        try:
            status = read_json(status_file)
        except (OSError, ValueError):
            continue
        if status.get("config_hash") == config_hash and status.get("outcome") not in (SUCCESS, "partial"):
            candidates.append((status.get("created_utc", ""), status["run_id"]))
    return sorted(candidates)[-1][1] if candidates else None


def affected_stages(changed_keys: list[str]) -> list[str]:
    affected: list[str] = []
    for stage in STAGES:
        if any(ck == key or ck.startswith(key + ".") or key.startswith(ck + ".") for ck in changed_keys for key in stage.config_keys):
            affected.append(stage.id)
    if affected:
        first = min(affected)
        affected = [s.id for s in STAGES if s.id >= first]
    return affected


def confirm(services: Services, options: RunOptions, message: str) -> None:
    if options.yes:
        return
    if services.confirm is not None:
        if services.confirm(message):
            return
        raise PipelineError(ExitCode.CONFIRMATION_REQUIRED, "E_NOT_CONFIRMED", "Confirmation declined")
    if sys.stdin is not None and sys.stdin.isatty():
        sys.stderr.write(message + "\nProceed? [y/N] ")
        sys.stderr.flush()
        if sys.stdin.readline().strip().lower() in {"y", "yes"}:
            return
        raise PipelineError(ExitCode.CONFIRMATION_REQUIRED, "E_NOT_CONFIRMED", "Confirmation declined")
    raise PipelineError(
        ExitCode.CONFIRMATION_REQUIRED,
        "E_CONFIRMATION_REQUIRED",
        "This run needs confirmation but stdin is not a terminal",
        hint="Re-run with --yes to acknowledge the compatibility notice and planned downloads/installs.",
    )


# --------------------------------------------------------------------------- execution
class Orchestrator:
    def __init__(self, services: Services | None = None, stream=None) -> None:
        self.services = services or Services()
        self.stream = stream or sys.stdout

    def _print(self, text: str = "") -> None:
        self.stream.write(text + "\n")
        self.stream.flush()

    # -- run setup ------------------------------------------------------------
    def _prepare_run(self, config: ResolvedConfig, options: RunOptions, logger: EventLogger) -> tuple[str, ResolvedConfig, str | None, str]:
        """Return (run_id, effective_config, parent_run_id, how)."""
        workspace = Path(config.get("storage.root"))
        if options.resume_run_id:
            run_dir = workspace / "runs" / options.resume_run_id
            stored_path = run_dir / "configs" / "resolved-config.json"
            if not stored_path.exists():
                raise PipelineError(
                    ExitCode.DEPENDENCY,
                    "E_RUN_NOT_FOUND",
                    f"No resumable run {options.resume_run_id} under {workspace / 'runs'}",
                    hint="Check --output-dir/storage.root; list runs with ./main.sh --list-runs.",
                )
            stored = read_json(stored_path)
            # The stored resolved config is authoritative; only command-line overrides are applied on top
            # (the current YAML is ignored). Operational keys (budgets, --yes, ...) may change freely.
            candidate = finalize(
                deep_merge(stored["config"], overrides_to_tree(options.cli_overrides)),
                stored.get("sources", []) + (["resume overrides"] if options.cli_overrides else []),
                profiles=stored.get("profiles", []),
            )
            changed = diff_configs(stored["config"], candidate.data)
            if changed:
                if not options.fork:
                    raise PipelineError(
                        ExitCode.DEPENDENCY,
                        "E_RESUME_CONFIG_CONFLICT",
                        f"Overrides change the stored configuration of run {options.resume_run_id}: {', '.join(changed)}",
                        hint=f"Stages that would be invalidated: {', '.join(affected_stages(changed)) or 'none'}. "
                        "Drop the overrides to resume as-is, or add --fork to start a new run that reuses still-valid receipts.",
                        details={"changed_keys": changed},
                    )
                return new_run_id(), candidate, options.resume_run_id, "fork"
            return options.resume_run_id, candidate, stored.get("parent_run_id"), "resume"
        if config.get("execution.resume") and not options.new_run:
            found = find_resumable_run(workspace, config.config_hash)
            if found:
                return found, config, None, "auto-resume"
        return new_run_id(), config, None, "new"

    def execute(self, config: ResolvedConfig, options: RunOptions) -> int:
        redactor = Redactor((config.get("model.hf_token_env"),))
        logger = EventLogger(None, redactor, config.get("logging.console_level"))
        if options.dry_run:
            return self.dry_run(config, options)
        run_id, config, parent, how = self._prepare_run(config, options, logger)
        workspace = Path(config.get("storage.root"))
        involves_model = not options.setup_only
        notice = (COMPATIBILITY_NOTICE + "\n\n") if involves_model else ""
        confirm(
            self.services,
            options,
            f"{notice}Run {run_id} ({how}): stages {', '.join(selected_stage_ids(options))}. "
            f"This may download models/tools and install project-local environments under {workspace}.",
        )
        interrupt = threading.Event()
        run = Run(run_id, workspace, config, logger, self.services, options, interrupt, parent_run_id=parent)
        for sub in ("logs", "receipts", "configs", "metrics", "checkpoints", "reports"):
            (run.run_dir / sub).mkdir(parents=True, exist_ok=True)
        logger.attach(run.run_dir / "logs")
        lock = FileLock(run.run_dir / ".lock", f"run {run_id}")
        lock.acquire()
        previous_handlers = self._install_signals(interrupt, logger)
        try:
            self._persist_config(run, how)
            if how == "fork" and parent:
                self._seed_fork(run, workspace / "runs" / parent)
            logger.info("run.start", f"Run {run_id} ({how}); workspace {workspace}", config_hash=config.config_hash)
            return self._run_stages(run, options)
        finally:
            self._restore_signals(previous_handlers)
            lock.release()

    def _persist_config(self, run: Run, how: str) -> None:
        cfg_dir = run.run_dir / "configs"
        stored = cfg_dir / "resolved-config.json"
        if stored.exists() and how in ("resume", "auto-resume"):
            return
        atomic_write_json(
            stored, {"config": run.config.data, "sources": run.config.sources, "profiles": run.config.profiles, "parent_run_id": run.parent_run_id}
        )
        atomic_write_text(cfg_dir / "resolved-config.yaml", dump_yaml(run.config.data))
        atomic_write_text(cfg_dir / "resolved-config.sha256", run.config.config_hash + "\n")
        run.update_status(config_hash=run.config.config_hash, parent_run_id=run.parent_run_id)

    def _seed_fork(self, run: Run, parent_dir: Path) -> None:
        for receipt in sorted((parent_dir / "receipts").glob("*.json")):
            data = load_receipt(receipt)
            if data and data.get("outcome") == SUCCESS:
                data["copied_from_run"] = data.get("copied_from_run") or data["run_id"]
                write_receipt(run.receipts_dir / receipt.name, data)
        run.registry.import_from(parent_dir / "artifacts.json")
        run.logger.info("run.fork", f"Forked from {parent_dir.name}; still-valid receipts will be reused")

    def _install_signals(self, interrupt: threading.Event, logger: EventLogger):
        if threading.current_thread() is not threading.main_thread():
            return None
        previous = {}

        def handler(signum, _frame):
            if interrupt.is_set():
                raise KeyboardInterrupt
            interrupt.set()
            logger.warning("run.signal", f"Received signal {signum}; stopping after cleanup (send again to abort immediately)")

        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, handler)
        return previous

    @staticmethod
    def _restore_signals(previous) -> None:
        if previous:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    # -- main loop --------------------------------------------------------------
    def _run_stages(self, run: Run, options: RunOptions) -> int:
        selected = selected_stage_ids(options)
        forced = set(selected) if options.rerun else set()
        first_error: PipelineError | None = None
        partial = False
        blocked_by: dict[str, str] = {}
        summary_rows = []
        for stage in STAGES:
            if stage.id not in selected:
                continue
            if run.interrupt_event.is_set():
                first_error = first_error or Interrupted()
                break
            outcome, error, was_partial, duration = self._run_one(run, stage, forced, blocked_by)
            summary_rows.append((stage, outcome, duration))
            partial = partial or was_partial
            if outcome in (FAILED, BLOCKED, INTERRUPTED):
                blocked_by[stage.id] = outcome
                first_error = first_error or error
                if outcome == INTERRUPTED or run.config.get("execution.stop_on_failure"):
                    break
        final = INTERRUPTED if isinstance(first_error, Interrupted) else (FAILED if first_error else ("partial" if partial else SUCCESS))
        run.update_status(outcome=final, ended_utc=utc_now())
        report_receipt = run.load_receipt("10")
        if first_error is not None and not (report_receipt and report_receipt.get("outcome") == SUCCESS):
            try:
                from .stages.stage10_report import write_report  # noqa: PLC0415

                write_report(run, final=False)
            except Exception as exc:  # noqa: BLE001 - reporting must never mask the real failure
                run.logger.warning("run.report", f"Could not write failure report: {exc}")
        self._print_summary(run, summary_rows, final, first_error)
        if first_error is not None:
            return int(first_error.code)
        return int(ExitCode.PARTIAL) if partial else int(ExitCode.OK)

    def _run_one(self, run: Run, stage: Stage, forced: set[str], blocked_by: dict[str, str]):
        ctx = StageContext(run, stage, run.run_dir / "logs" / f"{stage.id}-{stage.name}")
        started = utc_now()
        applicable, reason = stage.applicable(run.config)
        if not applicable:
            receipt = base_receipt(ctx, "", "", {}, started)
            receipt.update(outcome=SKIPPED, diagnostics={"not_applicable": True, "reason": reason})
            write_receipt(run.receipt_path(stage), receipt)
            run.update_status(stage, status=SKIPPED, reason=reason)
            run.logger.info(f"stage.{stage.id}", f"[{stage.id} {stage.name}] skipped: {reason}")
            return SKIPPED, None, False, 0.0
        dep_keys: dict[str, str | None] = {}
        for dep in stage.dependencies(run.config):
            dep_receipt = run.load_receipt(dep)
            satisfied = dep_receipt is not None and (
                dep_receipt.get("outcome") == SUCCESS or (dep_receipt.get("outcome") == SKIPPED and dep_receipt.get("diagnostics", {}).get("not_applicable"))
            )
            if not satisfied:
                dep_stage = stage_by_ref(dep)
                error = PipelineError(
                    ExitCode.DEPENDENCY,
                    "E_MISSING_DEPENDENCY",
                    f"Stage {stage.id} ({stage.name}) needs a valid receipt from stage {dep} ({dep_stage.name})"
                    + (f"; it {blocked_by[dep]} in this invocation" if dep in blocked_by else ""),
                    hint=f"Run it first, e.g. ./main.sh --resume {run.run_id} --from {dep}.",
                )
                return self._finish(run, stage, ctx, started, "", "", dep_keys, BLOCKED, error, None)
            used = stage.components_used(run.config) if dep == "01" else None
            if used is not None:
                fingerprints = dep_receipt.get("diagnostics", {}).get("component_fingerprints", {})
                missing = sorted(c for c in used if c not in fingerprints)
                if missing:
                    error = PipelineError(
                        ExitCode.DEPENDENCY,
                        "E_COMPONENT_NOT_PROVISIONED",
                        f"Stage {stage.id} needs components {missing} that stage 01 did not provision in this run",
                        hint=f"Re-run stage 01: ./main.sh --resume {run.run_id} --from 1",
                    )
                    return self._finish(run, stage, ctx, started, "", "", dep_keys, BLOCKED, error, None)
                dep_keys[dep] = sha256_obj({c: fingerprints[c] for c in sorted(used)})
            else:
                dep_keys[dep] = dep_receipt.get("stage_key")
        stage_key, stage_cfg_hash = stage.stage_key(run, dep_keys)
        existing = load_receipt(run.receipt_path(stage))
        if existing and existing.get("outcome") == SUCCESS and existing.get("stage_key") == stage_key and stage.id not in forced and not stage.always_rerun:
            problems = stage.validate_outputs(run, existing)
            if not problems:
                run.update_status(stage, status="reused", receipt=str(run.receipt_path(stage)))
                run.logger.info(f"stage.{stage.id}", f"[{stage.id} {stage.name}] reused valid receipt from {existing.get('ended_utc')}")
                return SUCCESS, None, bool(existing.get("partial")), 0.0
            run.logger.warning(f"stage.{stage.id}", f"Receipt for {stage.id} no longer valid: {problems[:3]}; re-running")
        if existing:
            history = run.receipts_dir / "history"
            history.mkdir(exist_ok=True)
            os.replace(run.receipt_path(stage), history / f"{stage.receipt_name}.{int(time.time())}")
        run.update_status(stage, status=RUNNING, started_utc=started)
        run.logger.info(f"stage.{stage.id}", f"[{stage.id} {stage.name}] starting")
        sampler = ResourceSampler(
            run.run_dir / "metrics" / f"resources-{stage.id}.jsonl",
            run.workspace,
            float(run.config.get("logging.resource_sample_seconds")),
            gpu_query=self.services.gpu_query,
        ).start()
        error: PipelineError | None = None
        outcome = SUCCESS
        try:
            stage.check_prerequisites(ctx)
            run.check_interrupt()
            stage.execute(ctx)
            run.check_interrupt()
            draft = {"outputs": ctx.outputs}
            problems = stage.validate_outputs(run, draft)
            if problems:
                raise PipelineError(ExitCode.VERIFY, "E_STAGE_OUTPUT_INVALID", f"Stage {stage.id} outputs failed validation: {problems[:5]}")
        except Interrupted as exc:
            outcome, error = INTERRUPTED, exc
        except KeyboardInterrupt:
            outcome, error = INTERRUPTED, Interrupted("Aborted by repeated signal")
        except PipelineError as exc:
            outcome, error = (INTERRUPTED if exc.code == ExitCode.INTERRUPTED else FAILED), exc
        except Exception as exc:  # noqa: BLE001 - convert bugs into a failed receipt, keep traceback
            outcome = FAILED
            error = PipelineError(ExitCode.INTERNAL, "E_INTERNAL", f"{type(exc).__name__}: {exc}", details={"traceback": traceback.format_exc()})
            run.logger.error(f"stage.{stage.id}", traceback.format_exc())
        resources = sampler.stop()
        return self._finish(run, stage, ctx, started, stage_key, stage_cfg_hash, dep_keys, outcome, error, resources)

    def _finish(self, run, stage, ctx, started, stage_key, cfg_hash, dep_keys, outcome, error, resources):
        receipt = base_receipt(ctx, stage_key, cfg_hash, dep_keys, started)
        duration = (datetime.fromisoformat(receipt["ended_utc"].rstrip("Z")) - datetime.fromisoformat(started.rstrip("Z"))).total_seconds()
        receipt.update(
            outcome=outcome,
            duration_s=round(duration, 3),
            error=error.to_dict() if error else None,
            resources=resources,
            resume={"attempt_utc": started, "rerun_forced": stage.id in selected_stage_ids(run.options) and run.options.rerun},
        )
        write_receipt(run.receipt_path(stage), receipt)
        run.update_status(
            stage, status=outcome, started_utc=started, ended_utc=receipt["ended_utc"], duration_s=receipt["duration_s"], receipt=str(run.receipt_path(stage))
        )
        if error is not None:
            run.logger.error(f"stage.{stage.id}", f"[{stage.id} {stage.name}] {outcome}: {error}")
        else:
            run.logger.info(f"stage.{stage.id}", f"[{stage.id} {stage.name}] {outcome}{' (partial)' if ctx.partial else ''} in {duration:.1f}s")
        return outcome, error, ctx.partial, duration

    def _print_summary(self, run: Run, rows, final: str, error: PipelineError | None) -> None:
        self._print("")
        self._print(f"Run {run.run_id}: {final}")
        for stage, outcome, duration in rows:
            self._print(f"  {stage.id} {stage.name:20s} {outcome:12s} {duration:9.1f}s")
        report = run.run_dir / "reports" / "report.md"
        if report.exists():
            self._print(f"Report: {report}")
        if error is not None:
            self._print(f"Error: {error}")
            self._print(f"Resume: ./main.sh --resume {run.run_id}")

    # -- dry run ------------------------------------------------------------------
    def dry_run(self, config: ResolvedConfig, options: RunOptions) -> int:
        """Describe the plan without network, installs, GPU work, model loading or writes."""
        from .planning import describe_plan  # noqa: PLC0415 - keeps import graph small

        probe = self.services.probe_factory(False)
        text = describe_plan(config, options, probe)
        self._print(text)
        return int(ExitCode.OK)


def run_status(workspace: Path, run_id: str) -> dict[str, Any]:
    path = workspace / "runs" / run_id / "status.json"
    if not path.exists():
        raise PipelineError(ExitCode.DEPENDENCY, "E_RUN_NOT_FOUND", f"Run {run_id} not found under {workspace / 'runs'}")
    return read_json(path)


def list_runs(workspace: Path) -> list[dict[str, Any]]:
    rows = []
    for status_file in sorted((workspace / "runs").glob("*/status.json")):
        try:
            status = read_json(status_file)
        except (OSError, ValueError):
            continue
        rows.append({k: status.get(k) for k in ("run_id", "created_utc", "updated_utc", "outcome", "config_hash", "parent_run_id")})
    return rows


# --------------------------------------------------------------------------- cleanup
def cleanup_plan(workspace: Path) -> list[dict[str, Any]]:
    """Items safe to delete: never sources, selected masters or finalized exports."""
    items = []
    for base in ("models", "exports", "tools"):
        root = workspace / base
        if root.is_dir():
            for partial in sorted(root.rglob("*.partial")):
                items.append({"path": str(partial), "reason": "incomplete output (.partial)"})
    promoted = set()
    for index in (workspace / "runs").glob("*/artifacts.json"):
        try:
            for entry in read_json(index)["artifacts"].values():
                promoted.add(str(Path(entry["path"]).resolve()))
        except (OSError, ValueError, KeyError):
            continue
    cand_root = workspace / "models" / "candidates"
    if cand_root.is_dir():
        for cand in sorted(p for p in cand_root.glob("*/*") if p.is_dir() and not p.name.endswith(".partial")):
            if str(cand.resolve()) not in promoted:
                items.append({"path": str(cand), "reason": "candidate checkpoint not registered as an artifact"})
    inter = workspace / "models" / "gguf-intermediate"
    if inter.is_dir():
        for path in sorted(inter.iterdir()):
            items.append({"path": str(path), "reason": "GGUF conversion intermediate (re-creatable)"})
    return items


def cleanup(workspace: Path, token: str | None, stream=sys.stdout) -> int:
    items = cleanup_plan(workspace)
    plan_token = sha256_obj(items)[:16]
    if token is None:
        stream.write(f"Cleanup plan for {workspace} (dry run, nothing deleted):\n")
        for item in items:
            stream.write(f"  {item['path']}  [{item['reason']}]\n")
        if not items:
            stream.write("  nothing to clean\n")
        stream.write(f"To delete exactly these items run: ./main.sh --cleanup --confirm-cleanup {plan_token}\n")
        return 0
    if token != plan_token:
        raise PipelineError(
            ExitCode.USAGE,
            "E_CLEANUP_TOKEN",
            "Cleanup token does not match the current plan (the workspace changed)",
            hint="Run ./main.sh --cleanup again to review the current plan.",
        )
    protected = [workspace / "models" / "source", workspace / "models" / "selected", workspace / "exports"]
    for item in items:
        path = Path(item["path"]).resolve()
        if any(p.resolve() in path.parents for p in protected) and not path.name.endswith(".partial"):
            raise PipelineError(ExitCode.INTERNAL, "E_CLEANUP_PROTECTED", f"Refusing to delete protected path {path}")
        remove_tree(path)
        stream.write(f"deleted {path}\n")
    return 0
