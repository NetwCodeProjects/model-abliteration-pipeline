"""Config loading, precedence, cross-field validation and fingerprints.

Precedence for a new run: CLI > YAML > profiles listed in the YAML > built-in defaults.
Relative paths inside config files resolve against the project root; relative paths given
on the command line resolve against the invoking working directory (the CLI makes them
absolute before they reach this module).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import configspec
from .adapters.hf_hub import default_family, normalize_model_ref, validate_revision
from .errors import ExitCode, PipelineError
from .fsutil import sha256_file, sha256_obj

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PATH_KEYS = (
    "model.local_path",
    "environment.cache_root",
    "abliteration.heretic_config",
    "evaluation.published_baselines",
    "evaluation.benign_prompts_file",
    "quantization.input_path",
    "storage.root",
)

COMPATIBILITY_NOTICE = (
    "This platform does not automatically verify that your model architecture is supported by the "
    "selected abliteration engine or quantization backend. Verify compatibility with the engines' "
    "official documentation before execution. Unsupported models or formats may fail during model "
    "loading, transformation, or export. The platform manages execution, dependencies, logs, and "
    "artifacts but does not guarantee compatibility with every model."
)


_LOADER = None


def _yaml12_loader():
    """SafeLoader where only true/false are booleans (YAML 1.2 core schema).

    PyYAML follows YAML 1.1, which turns unquoted ``off``/``no`` values and even the key ``yes``
    (``execution.yes``) into booleans - a silent trap for this config format.
    """
    global _LOADER
    if _LOADER is None:
        import re  # noqa: PLC0415

        import yaml  # noqa: PLC0415

        class Yaml12SafeLoader(yaml.SafeLoader):
            pass

        resolvers = {k: [r for r in v if r[0] != "tag:yaml.org,2002:bool"] for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()}
        Yaml12SafeLoader.yaml_implicit_resolvers = resolvers
        Yaml12SafeLoader.add_implicit_resolver("tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))
        _LOADER = Yaml12SafeLoader
    return _LOADER


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:
        raise PipelineError(
            ExitCode.ENVIRONMENT,
            "E_CONTROL_ENV_INCOMPLETE",
            "PyYAML is required to read configuration files",
            hint="Install python3-yaml, or let ./main.sh bootstrap the control environment (.venv).",
        ) from exc
    try:
        with open(path, encoding="utf-8") as handle:
            data = yaml.load(handle, Loader=_yaml12_loader())  # noqa: S506 - SafeLoader subclass
    except FileNotFoundError as exc:
        raise PipelineError(ExitCode.CONFIG, "E_CONFIG_NOT_FOUND", f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise PipelineError(ExitCode.CONFIG, "E_CONFIG_YAML", f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise PipelineError(ExitCode.CONFIG, "E_CONFIG_YAML", f"{path}: top level must be a mapping")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_dotted(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = data
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def overrides_to_tree(overrides: dict[str, Any]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for dotted, value in overrides.items():
        set_dotted(tree, dotted, value)
    return tree


def flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(data, dict) and data:
        out: dict[str, Any] = {}
        for key, value in data.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else key))
        return out
    return {prefix: data}


@dataclass
class ResolvedConfig:
    data: dict[str, Any]
    sources: list[str] = field(default_factory=list)
    profiles: list[dict[str, str]] = field(default_factory=list)  # merged profile files: path + sha256 (provenance)

    def get(self, dotted: str, default: Any = None) -> Any:
        return get_dotted(self.data, dotted, default)

    @property
    def config_hash(self) -> str:
        return sha256_obj(strip_operational(self.data))

    def subset(self, keys: list[str]) -> dict[str, Any]:
        stripped = strip_operational(self.data)
        return {key: get_dotted(stripped, key) for key in keys}


def strip_operational(data: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(data)
    for dotted in configspec.operational_keys():
        parts = dotted.split(".")
        current = result
        for part in parts[:-1]:
            current = current.get(part, {}) if isinstance(current, dict) else {}
        if isinstance(current, dict):
            current.pop(parts[-1], None)
    return result


def _resolve_path(value: str | None, base: Path) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def build_config(
    config_path: str | Path | None,
    cli_overrides: dict[str, Any] | None = None,
    *,
    require_model: bool = True,
) -> ResolvedConfig:
    sources = ["built-in defaults"]
    profiles: list[dict[str, str]] = []
    merged = configspec.defaults()
    if config_path is not None:
        config_path = Path(config_path)
        user = load_yaml(config_path)
        for profile in user.get("profiles", []) or []:
            profile_path = Path(profile)
            if not profile_path.is_absolute():
                profile_path = (config_path.parent / profile_path).resolve()
            profile_data = load_yaml(profile_path)
            if "profiles" in profile_data:
                raise PipelineError(ExitCode.CONFIG, "E_CONFIG_INVALID", f"Profile {profile_path} must not nest 'profiles'")
            merged = deep_merge(merged, profile_data)
            sources.append(f"profile {profile_path}")
            profiles.append({"path": str(profile_path), "sha256": sha256_file(profile_path)})
        merged = deep_merge(merged, user)
        sources.append(f"yaml {config_path}")
    errors = configspec.validate(merged)
    if errors:
        raise PipelineError(ExitCode.CONFIG, "E_CONFIG_INVALID", "Configuration is invalid:\n  - " + "\n  - ".join(errors), details={"errors": errors})
    for dotted in PATH_KEYS:
        set_dotted(merged, dotted, _resolve_path(get_dotted(merged, dotted), PROJECT_ROOT))
    if cli_overrides:
        merged = deep_merge(merged, overrides_to_tree(cli_overrides))
        sources.append("command line")
    return finalize(merged, sources, require_model=require_model, profiles=profiles)


def finalize(data: dict[str, Any], sources: list[str], *, require_model: bool = True, profiles: list[dict[str, str]] | None = None) -> ResolvedConfig:
    errors = configspec.validate(data)
    if errors:
        raise PipelineError(ExitCode.CONFIG, "E_CONFIG_INVALID", "Configuration is invalid:\n  - " + "\n  - ".join(errors), details={"errors": errors})
    data = copy.deepcopy(data)
    _cross_validate(data, require_model=require_model)
    return ResolvedConfig(data, sources, list(profiles or []))


def _cross_validate(data: dict[str, Any], *, require_model: bool) -> None:
    model = data["model"]
    if model["url"] and model["local_path"]:
        raise PipelineError(
            ExitCode.USAGE,
            "E_MODEL_SOURCE_CONFLICT",
            "model.url and model.local_path are mutually exclusive",
            hint="Use either --model-url or --model-path.",
        )
    if model["url"]:
        model["url"] = normalize_model_ref(model["url"])
    if model["revision"]:
        if model["local_path"]:
            raise PipelineError(ExitCode.USAGE, "E_MODEL_SOURCE_CONFLICT", "model.revision applies only to model.url")
        validate_revision(model["revision"])
    if require_model and not (model["url"] or model["local_path"]):
        raise PipelineError(ExitCode.CONFIG, "E_MODEL_MISSING", "No model selected", hint="Set model.url in the YAML or pass --model-url/--model-path.")
    if not model["family"]:
        model["family"] = default_family(model["url"], model["local_path"])

    quant = data["quantization"]
    formats: list[str] = []
    for raw in quant["formats"]:
        name = raw.strip().lower()
        if name not in configspec.FP8_FORMATS and name not in configspec.GGUF_TYPES:
            raise PipelineError(
                ExitCode.UNSUPPORTED,
                "E_UNSUPPORTED_FORMAT",
                f"Unsupported quantization format {raw!r}",
                hint=f"Supported: fp8, {', '.join(configspec.GGUF_TYPES)} (verified against pinned llama.cpp).",
            )
        if name not in formats:
            formats.append(name)
    quant["formats"] = formats
    if quant["fp8"]["calibration_manifest"]:
        raise PipelineError(
            ExitCode.UNSUPPORTED,
            "E_FP8_CALIBRATION_UNSUPPORTED",
            "quantization.fp8.calibration_manifest is set, but the FP8 schemes in v1 (FP8_BLOCK, FP8_DYNAMIC) are data-free",
            hint="Remove calibration_manifest; calibrated static FP8 is not implemented.",
        )
    if quant["input"] == "path" and not quant["input_path"]:
        raise PipelineError(ExitCode.CONFIG, "E_CONFIG_INVALID", "quantization.input=path requires quantization.input_path")
    if quant["input"] == "selected_master" and not data["abliteration"]["enabled"] and quant["enabled"]:
        quant["input"] = "source"

    sel = data["selection"]
    for gate, key in configspec.GATE_CONFIG_KEYS.items():
        if gate not in sel["optional_gates"] and sel[key] in (None, False):
            raise PipelineError(
                ExitCode.CONFIG,
                "E_GATE_REQUIREMENT_MISSING",
                f"selection.{key} is {sel[key]!r}, which would disable the required gate '{gate}'",
                hint=f"Set a threshold, or add '{gate}' to selection.optional_gates to make the gate optional (see docs/evaluation.md).",
            )

    if data["evaluation"]["baseline_mode"] == "local":
        raise PipelineError(
            ExitCode.UNSUPPORTED,
            "E_LOCAL_BASELINES_UNSUPPORTED",
            "evaluation.baseline_mode=local is not implemented in v1",
            hint="Use published_only; local reference evaluation belongs to a later version or Project B.",
        )
    abl = data["abliteration"]
    if abl["trial_id"] is not None and abl["pareto_trial_index"] is not None:
        raise PipelineError(
            ExitCode.CONFIG,
            "E_TRIAL_CHOICE_CONFLICT",
            f"abliteration.trial_id ({abl['trial_id']}) and abliteration.pareto_trial_index ({abl['pareto_trial_index']}) are both set",
            hint="Choose a trial by its Heretic index (trial_id) or by Pareto position, not both (see docs/study-lifecycle.md).",
        )
    if abl["enabled"] and abl["experiment"] == "reproduce":
        rep = abl["reproduce"]
        if not (rep["source"] or rep["reference_repo"]):
            raise PipelineError(
                ExitCode.CONFIG,
                "E_REPRODUCE_SOURCE_MISSING",
                "experiment=reproduce needs abliteration.reproduce.source or abliteration.reproduce.reference_repo",
            )
        if rep["reference_repo"]:
            rep["reference_repo"] = normalize_model_ref(rep["reference_repo"])
    if data["environment"]["torch_source"] == "system" and data["environment"]["profile"] == "ubuntu":
        raise PipelineError(ExitCode.CONFIG, "E_CONFIG_INVALID", "environment.torch_source=system is only supported with the lambda-stack profile")
    if not data["environment"]["cache_root"]:
        data["environment"]["cache_root"] = str(Path(data["storage"]["root"]) / "cache")


def baseline_file_hash(config: ResolvedConfig) -> str | None:
    path = config.get("evaluation.published_baselines")
    if path and Path(path).is_file():
        return sha256_file(path)
    return None


def diff_configs(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Dotted keys whose non-operational values differ."""
    a, b = flatten(strip_operational(old)), flatten(strip_operational(new))
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def dump_yaml(data: dict[str, Any]) -> str:
    try:
        import yaml  # noqa: PLC0415

        return yaml.safe_dump(data, sort_keys=True, allow_unicode=True)
    except ImportError:  # pragma: no cover - control env always has PyYAML for real runs
        import json

        return json.dumps(data, indent=2, sort_keys=True)
