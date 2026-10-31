"""Command-line interface behind ./main.sh (stdlib-only at import time so --help always works)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .adapters.hf_hub import normalize_model_ref, validate_revision
from .config import PROJECT_ROOT, build_config, dump_yaml
from .errors import ExitCode, PipelineError

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pipeline.yaml"

EPILOG = """\
examples:
  ./main.sh --config configs/pipeline.yaml --dry-run
  ./main.sh --model-url Qwen/Qwen3.8-27B --lambda-stack --yes
  ./main.sh --setup-only --lambda-stack --yes
  ./main.sh --through 2 --yes
  ./main.sh --resume RUN_ID --from 5 --through 6
  ./main.sh --model-url org/model --quantize-only --quantizations fp8,iq4_xs,iq3_s --yes
  ./main.sh --experiment optimize --max-trials 50 --max-runtime-minutes 120 --yes

stages: 00 preflight, 01 environment, 02 acquire, 03 published_baselines, 04 prepare_engine,
        05 abliterate, 06 qualify, 07 select_master, 08 quantize, 09 verify_exports, 10 report_export
exit codes: 0 ok, 1 internal, 2 usage, 3 config, 4 preflight, 5 environment, 6 acquire,
        7 dependency/resume, 8 engine, 9 gates, 10 quantize, 11 verify, 12 locked,
        14 unsupported, 15 confirmation required, 20 partial success, 130 interrupted
See docs/cli.md for details.
"""


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.sh",
        description="Project A - reproducible abliteration (Heretic) and quantization (FP8 / GGUF) pipeline.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="store_true", help="print version and exit")
    p.add_argument(
        "--config", metavar="PATH", help=f"pipeline YAML (default: {DEFAULT_CONFIG.relative_to(PROJECT_ROOT)} if present; 'none' = built-in defaults)"
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--model-url", metavar="REF", help="Hugging Face repo ID or https://huggingface.co/<org>/<name> URL")
    src.add_argument("--model-path", metavar="DIR", help="local Hugging Face-format model directory")
    p.add_argument("--revision", metavar="REV", help="branch/tag/commit for --model-url (always pinned to a commit SHA)")
    env = p.add_mutually_exclusive_group()
    env.add_argument("--lambda-stack", action="store_true", help="reuse an already-provisioned Lambda Stack host (never installs Lambda Stack)")
    env.add_argument("--ubuntu", action="store_true", help="plain Ubuntu host; host-level steps are opt-in and guarded")
    env.add_argument("--env", choices=["auto", "lambda-stack", "ubuntu"], help="environment profile (default from config: auto)")
    p.add_argument("--from", dest="from_stage", metavar="STAGE", help="first stage to run (number or name)")
    p.add_argument("--through", dest="through_stage", metavar="STAGE", help="last stage to run")
    p.add_argument("--only", dest="only_stage", metavar="STAGE", help="run exactly one stage (dependencies must have valid receipts)")
    res = p.add_mutually_exclusive_group()
    res.add_argument("--resume", metavar="RUN_ID", help="resume a run with its stored resolved config")
    res.add_argument("--new-run", action="store_true", help="never auto-resume; always start a new run")
    p.add_argument("--fork", action="store_true", help="with --resume: accept config-changing overrides by forking a new run (valid receipts reused)")
    p.add_argument("--rerun", action="store_true", help="re-execute the selected stages even if valid receipts exist")
    p.add_argument("--dry-run", action="store_true", help="print the plan; no network, installs, GPU work, model loading or file writes")
    p.add_argument("--setup-only", action="store_true", help="run only stages 00-01 (provision components the config needs)")
    p.add_argument("--quantize-only", action="store_true", help="skip abliteration; quantize the source model (or --model-path)")
    p.add_argument("--quantizations", metavar="LIST", help="comma-separated formats, e.g. fp8,iq4_xs,iq3_s")
    p.add_argument("--experiment", choices=["reproduce", "optimize"], help="Heretic experiment profile")
    p.add_argument("--allow-approximate-reproduction", action="store_true", help="permit labelled approximate reproduction when exact inputs are unavailable")
    p.add_argument("--qualification", choices=["off", "fast", "extended"], help="qualification depth (stage 06)")
    p.add_argument("--max-trials", type=positive_int, metavar="N", help="Heretic trial budget")
    p.add_argument("--max-runtime-minutes", type=positive_int, metavar="N", help="wall-clock budget for Heretic optimization")
    choice = p.add_mutually_exclusive_group()
    choice.add_argument("--trial-id", type=positive_int, metavar="N", help="export Heretic trial N instead of the policy choice (gates still apply)")
    choice.add_argument("--pareto-index", type=non_negative_int, metavar="N", help="export the trial at Pareto position N (0 = fewest refusals)")
    p.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config key (repeatable; VALUE is YAML, e.g. --set selection.max_optimizer_kl=0.08 or =null)",
    )
    p.add_argument("--yes", "-y", action="store_true", help="acknowledge the compatibility notice and confirm downloads/installs non-interactively")
    p.add_argument("--output-dir", metavar="DIR", help="workspace root (storage.root)")
    p.add_argument("--log-level", choices=["debug", "info", "warning"], help="console verbosity")
    info = p.add_argument_group("inspection and maintenance (no pipeline execution)")
    info.add_argument("--print-config", action="store_true", help="print the resolved configuration and exit")
    info.add_argument("--list-stages", action="store_true", help="list stage IDs and exit")
    info.add_argument("--list-runs", action="store_true", help="list runs in the workspace and exit")
    info.add_argument("--status", metavar="RUN_ID", help="print a run's status.json and exit")
    info.add_argument("--cleanup", action="store_true", help="show (dry-run) what can be deleted safely; never sources or finalized exports")
    info.add_argument("--confirm-cleanup", metavar="TOKEN", help="with --cleanup: delete exactly the reviewed plan")
    return p


def check_combinations(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.setup_only and (args.from_stage or args.through_stage or args.only_stage):
        parser.error("--setup-only cannot be combined with --from/--through/--only")
    if args.only_stage and (args.from_stage or args.through_stage):
        parser.error("--only cannot be combined with --from/--through")
    if args.fork and not args.resume:
        parser.error("--fork requires --resume RUN_ID")
    if args.revision and args.model_path:
        parser.error("--revision applies to --model-url, not --model-path")
    if args.quantize_only and (args.experiment or args.allow_approximate_reproduction or args.max_trials or args.pareto_index is not None or args.trial_id):
        parser.error("--experiment/--max-trials/--trial-id/--pareto-index/--allow-approximate-reproduction have no effect with --quantize-only")
    if args.confirm_cleanup and not args.cleanup:
        parser.error("--confirm-cleanup requires --cleanup")


def to_overrides(args: argparse.Namespace) -> dict[str, Any]:
    o: dict[str, Any] = {}
    try:
        if args.model_url:
            o["model.url"] = normalize_model_ref(args.model_url)
            o["model.local_path"] = None
            o["model.family"] = None
        if args.model_path:
            o["model.local_path"] = str(Path(args.model_path).expanduser().resolve())
            o["model.url"] = None
            o["model.revision"] = None
            o["model.family"] = None
        if args.revision:
            o["model.revision"] = validate_revision(args.revision)
    except PipelineError as exc:
        raise PipelineError(ExitCode.USAGE, exc.error_id, exc.message, hint=exc.hint) from exc
    if args.lambda_stack:
        o["environment.profile"] = "lambda-stack"
    elif args.ubuntu:
        o["environment.profile"] = "ubuntu"
    elif args.env:
        o["environment.profile"] = args.env
    if args.quantizations is not None:
        o["quantization.formats"] = [f.strip() for f in args.quantizations.split(",") if f.strip()]
        o["quantization.enabled"] = bool(o["quantization.formats"])
    if args.quantize_only:
        o["abliteration.enabled"] = False
        o["quantization.enabled"] = True
    if args.experiment:
        o["abliteration.experiment"] = args.experiment
    if args.allow_approximate_reproduction:
        o["abliteration.reproduce.allow_approximate"] = True
    if args.qualification:
        o["evaluation.qualification"] = args.qualification
    if args.max_trials:
        o["abliteration.max_trials"] = args.max_trials
    if args.max_runtime_minutes:
        o["abliteration.max_runtime_minutes"] = args.max_runtime_minutes
    if args.trial_id:
        o["abliteration.trial_id"] = args.trial_id
        o["abliteration.pareto_trial_index"] = None
    if args.pareto_index is not None:
        o["abliteration.pareto_trial_index"] = args.pareto_index
        o["abliteration.trial_id"] = None
    if args.yes:
        o["execution.yes"] = True
    if args.dry_run:
        o["execution.dry_run"] = True
    if args.output_dir:
        o["storage.root"] = str(Path(args.output_dir).expanduser().resolve())
    if args.log_level:
        o["logging.console_level"] = args.log_level
    o.update(parse_set_values(args.set_values, flag_keys=set(o)))
    return o


NUMBER_RE = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")


def parse_set_values(items: list[str], flag_keys: set[str] = frozenset()) -> dict[str, Any]:
    """``--set KEY=VALUE`` pairs. Key names and value types are checked by config validation (exit 3)."""
    import yaml  # noqa: PLC0415 - keeps --help stdlib-only

    from .config import PATH_KEYS  # noqa: PLC0415

    out: dict[str, Any] = {}
    for item in items:
        key, sep, text = item.partition("=")
        key = key.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", key) or not text.strip():
            raise PipelineError(
                ExitCode.USAGE, "E_SET_INVALID", f"--set expects KEY=VALUE with a dotted config key, got {item!r}", hint="Use null for an empty value."
            )
        if key in out or key in flag_keys:
            raise PipelineError(
                ExitCode.USAGE, "E_SET_INVALID", f"{key} is set twice on the command line", hint="Give each key once, either with --set or its dedicated flag."
            )
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise PipelineError(ExitCode.USAGE, "E_SET_INVALID", f"--set {key}: value is not valid YAML: {exc}") from exc
        if isinstance(value, str) and NUMBER_RE.fullmatch(value):
            value = float(value)  # YAML 1.1 reads 1e-3 as a string
        if key in PATH_KEYS and isinstance(value, str):
            value = str(Path(value).expanduser().resolve())
        out[key] = value
    return out


def config_path_from(args: argparse.Namespace) -> Path | None:
    if args.config and args.config.lower() == "none":
        return None
    if args.config:
        return Path(args.config).expanduser().resolve()
    return DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None


def main(argv: list[str] | None = None, services: Any = None, stream=None) -> int:
    stream = stream or sys.stdout
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        stream.write(f"model-abliteration-pipeline {__version__}\n")
        return int(ExitCode.OK)
    check_combinations(args, parser)
    try:
        if args.list_stages:
            from .stages import STAGES  # noqa: PLC0415

            for stage in STAGES:
                stream.write(f"{stage.id}  {stage.name:20s} {stage.title}\n")
            return int(ExitCode.OK)
        overrides = to_overrides(args)
        maintenance = args.list_runs or args.status or args.cleanup
        config = build_config(
            config_path_from(args),
            overrides,
            require_model=not (args.setup_only or args.resume or maintenance or args.print_config),
        )
        workspace = Path(config.get("storage.root"))
        if args.print_config:
            stream.write(dump_yaml(config.data))
            stream.write(f"# config_hash: {config.config_hash}\n# sources: {', '.join(config.sources)}\n")
            return int(ExitCode.OK)
        from .orchestrator import Orchestrator, RunOptions, cleanup, list_runs, run_status  # noqa: PLC0415

        if args.list_runs:
            for row in list_runs(workspace):
                stream.write(json.dumps(row) + "\n")
            return int(ExitCode.OK)
        if args.status:
            stream.write(json.dumps(run_status(workspace, args.status), indent=2) + "\n")
            return int(ExitCode.OK)
        if args.cleanup:
            return cleanup(workspace, args.confirm_cleanup, stream)
        options = RunOptions(
            from_stage=args.from_stage,
            through_stage=args.through_stage,
            only_stage=args.only_stage,
            setup_only=args.setup_only,
            rerun=args.rerun,
            dry_run=args.dry_run,
            resume_run_id=args.resume,
            fork=args.fork,
            new_run=args.new_run,
            yes=args.yes,
            config_path=str(config_path_from(args)) if config_path_from(args) else None,
            cli_overrides=overrides,
        )
        from .orchestrator import selected_stage_ids  # noqa: PLC0415

        selected_stage_ids(options)  # validate stage references early (usage error)
        return Orchestrator(services, stream).execute(config, options)
    except PipelineError as exc:
        sys.stderr.write(f"ERROR {exc}\n")
        return int(exc.code)
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted\n")
        return int(ExitCode.INTERRUPTED)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
