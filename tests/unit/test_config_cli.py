"""Config parsing, defaults, schema, precedence, CLI contract and mutual exclusivity."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from conftest import ROOT

from model_pipeline import configspec
from model_pipeline.cli import build_parser, main, to_overrides
from model_pipeline.config import build_config, deep_merge, diff_configs, strip_operational
from model_pipeline.errors import ExitCode, PipelineError
from model_pipeline.fsutil import sha256_file


def write(tmp_path: Path, data: dict, name: str = "c.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_builtin_defaults_validate():
    assert configspec.validate(configspec.defaults()) == []


def test_shipped_configs_resolve():
    for name in ("pipeline.yaml", "pipeline.example.yaml"):
        cfg = build_config(ROOT / "configs" / name)
        assert cfg.get("model.url") == "Qwen/Qwen3.8-27B"
        assert cfg.get("quantization.formats") == ["fp8", "iq4_xs", "iq3_s"]


def test_examples_resolve():
    for path in (ROOT / "examples").glob("*.yaml"):
        build_config(path, require_model=True)


def test_schema_file_is_generated_from_spec():
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_schema.py"), "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


def test_jsonschema_agrees_with_typed_validator(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "schemas" / "pipeline.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    good = configspec.defaults()
    jsonschema.validate(good, schema)
    bad_cases = [
        deep_merge(good, {"environment": {"profile": "windows"}}),
        deep_merge(good, {"abliteration": {"max_trials": 0}}),
        deep_merge(good, {"model": {"unknown_key": 1}}),
        deep_merge(good, {"model": {"hf_token_env": "1-bad"}}),
        deep_merge(good, {"selection": {"max_heldout_refusal_rate": 1.5}}),
    ]
    for bad in bad_cases:
        assert configspec.validate(bad), bad
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)


def test_precedence_cli_over_yaml_over_profile_over_default(tmp_path):
    profile = write(tmp_path, {"abliteration": {"max_trials": 11, "max_runtime_minutes": 11}, "evaluation": {"max_new_tokens": 11}}, "profile.yaml")
    main_cfg = write(tmp_path, {"profiles": [str(profile)], "model": {"url": "org/m"}, "abliteration": {"max_trials": 22}})
    cfg = build_config(main_cfg, {"abliteration.max_trials": 33})
    assert cfg.get("abliteration.max_trials") == 33  # CLI
    assert cfg.get("abliteration.max_runtime_minutes") == 11  # profile
    assert cfg.get("evaluation.max_new_tokens") == 11  # profile
    assert cfg.get("evaluation.qualification") == "fast"  # default
    assert "command line" in cfg.sources and any("profile" in s for s in cfg.sources)


def test_unknown_and_mistyped_keys_are_rejected(tmp_path):
    with pytest.raises(PipelineError) as err:
        build_config(write(tmp_path, {"model": {"url": "org/m"}, "abliteration": {"trials": 5, "max_trials": "many"}}))
    assert err.value.code == ExitCode.CONFIG
    assert "abliteration.trials: unknown key" in err.value.message and "max_trials: expected integer" in err.value.message


@pytest.mark.parametrize(
    "data,error_id",
    [
        ({"model": {"url": "a/b", "local_path": "/x"}}, "E_MODEL_SOURCE_CONFLICT"),
        ({"model": {"url": None}}, "E_MODEL_MISSING"),
        ({"model": {"url": "a/b"}, "quantization": {"formats": ["fp4_magic"]}}, "E_UNSUPPORTED_FORMAT"),
        ({"model": {"url": "a/b"}, "quantization": {"fp8": {"calibration_manifest": "x.json"}}}, "E_FP8_CALIBRATION_UNSUPPORTED"),
        ({"model": {"url": "a/b"}, "evaluation": {"baseline_mode": "local"}}, "E_LOCAL_BASELINES_UNSUPPORTED"),
        ({"model": {"url": "a/b"}, "abliteration": {"experiment": "reproduce"}}, "E_REPRODUCE_SOURCE_MISSING"),
        ({"model": {"url": "https://evil.example.com/a/b"}}, "E_MODEL_REF_INVALID"),
        ({"model": {"url": "a/b"}, "selection": {"max_optimizer_kl": None}}, "E_GATE_REQUIREMENT_MISSING"),
        ({"model": {"url": "a/b"}, "selection": {"require_integrity": False}}, "E_GATE_REQUIREMENT_MISSING"),
        ({"model": {"url": "a/b"}, "selection": {"optional_gates": ["no_such_gate"]}}, "E_CONFIG_INVALID"),
    ],
)
def test_cross_field_validation(tmp_path, data, error_id):
    with pytest.raises(PipelineError) as err:
        build_config(write(tmp_path, data))
    assert err.value.error_id == error_id


QWEN35_FP8_IGNORE = [  # the global default before the Qwen findings moved into configs/profiles/qwen3_5.yaml
    "lm_head",
    "re:.*embed_tokens$",
    "re:.*visual\\..*",
    "re:.*vision_tower.*",
    "re:.*mlp\\.gate$",
    "re:.*in_proj_(a|b|ba)$",
    "re:.*mtp\\.fc$",
    "re:.*norm.*",
    "re:.*conv1d.*",
]


@pytest.mark.parametrize("config", ["configs/pipeline.yaml", "examples/qwen-b200.yaml", "examples/ubuntu-quantize-only.yaml"])
def test_qwen_configs_keep_their_effective_settings_via_the_profile(config):
    cfg = build_config(ROOT / config)
    assert cfg.get("quantization.fp8.ignore") == QWEN35_FP8_IGNORE
    assert cfg.get("quantization.gguf.converter_transformers") == "5.18.0"
    profile = ROOT / "configs" / "profiles" / "qwen3_5.yaml"
    assert {"path": str(profile.resolve()), "sha256": sha256_file(profile)} in cfg.profiles


def test_non_qwen_config_does_not_inherit_qwen_exclusions(tmp_path):
    cfg = build_config(write(tmp_path, {"model": {"url": "org/some-llama"}}))
    ignore = cfg.get("quantization.fp8.ignore")
    assert not {"re:.*norm.*", "re:.*conv1d.*", "re:.*in_proj_(a|b|ba)$", "re:.*mtp\\.fc$"} & set(ignore)
    assert cfg.get("quantization.gguf.converter_transformers") is None and cfg.profiles == []


def test_only_optional_gates_may_be_disabled(tmp_path):
    cfg = build_config(write(tmp_path, {"model": {"url": "a/b"}, "selection": {"max_optimizer_kl": None, "optional_gates": ["optimizer_kl"]}}))
    assert cfg.get("selection.max_optimizer_kl") is None and cfg.get("selection.profile") == "qualified"


def test_formats_are_normalized_and_deduplicated(tmp_path):
    cfg = build_config(write(tmp_path, {"model": {"url": "a/b"}, "quantization": {"formats": ["FP8", "IQ4_XS", "iq4_xs"]}}))
    assert cfg.get("quantization.formats") == ["fp8", "iq4_xs"]


def test_operational_keys_do_not_change_config_hash(tmp_path):
    path = write(tmp_path, {"model": {"url": "a/b"}})
    a = build_config(path)
    b = build_config(path, {"abliteration.max_runtime_minutes": 5, "execution.yes": True, "logging.console_level": "debug"})
    c = build_config(path, {"abliteration.max_trials": 7})
    assert a.config_hash == b.config_hash != c.config_hash
    assert diff_configs(a.data, c.data) == ["abliteration.max_trials"]
    assert "max_runtime_minutes" not in strip_operational(a.data)["abliteration"]


def test_relative_paths_resolve_against_project_root(tmp_path):
    cfg = build_config(write(tmp_path, {"model": {"url": "a/b"}, "evaluation": {"published_baselines": "configs/baselines/qwen3.8-27b.yaml"}}))
    assert Path(cfg.get("evaluation.published_baselines")) == ROOT / "configs" / "baselines" / "qwen3.8-27b.yaml"


def parse(*argv):
    parser = build_parser()
    return parser, parser.parse_args(list(argv))


def test_cli_overrides_map_to_config_keys(tmp_path):
    _, args = parse(
        "--model-url",
        "https://huggingface.co/Qwen/Qwen3.8-27B",
        "--ubuntu",
        "--quantizations",
        "fp8, q4_k_m",
        "--experiment",
        "reproduce",
        "--qualification",
        "extended",
        "--max-trials",
        "5",
        "--max-runtime-minutes",
        "9",
        "--yes",
        "--output-dir",
        str(tmp_path),
        "--pareto-index",
        "2",
    )
    o = to_overrides(args)
    assert o["model.url"] == "Qwen/Qwen3.8-27B" and o["model.local_path"] is None
    assert o["environment.profile"] == "ubuntu"
    assert o["quantization.formats"] == ["fp8", "q4_k_m"]
    assert o["abliteration.experiment"] == "reproduce" and o["evaluation.qualification"] == "extended"
    assert o["abliteration.max_trials"] == 5 and o["abliteration.max_runtime_minutes"] == 9 and o["abliteration.pareto_trial_index"] == 2
    assert o["storage.root"] == str(tmp_path.resolve())


def test_cli_flags_override_yaml(tmp_path):
    path = write(tmp_path, {"model": {"url": "a/b"}, "environment": {"profile": "ubuntu"}, "abliteration": {"max_trials": 3}})
    _, args = parse("--lambda-stack", "--max-trials", "8")
    cfg = build_config(path, to_overrides(args))
    assert cfg.get("environment.profile") == "lambda-stack" and cfg.get("abliteration.max_trials") == 8


@pytest.mark.parametrize(
    "argv",
    [
        ["--lambda-stack", "--ubuntu"],
        ["--model-url", "a/b", "--model-path", "/x"],
        ["--setup-only", "--from", "3"],
        ["--only", "5", "--through", "6"],
        ["--fork"],
        ["--resume", "x", "--new-run"],
        ["--model-path", "/x", "--revision", "main"],
        ["--quantize-only", "--experiment", "optimize"],
        ["--max-trials", "0"],
        ["--experiment", "maybe"],
        ["--confirm-cleanup", "abc"],
    ],
)
def test_invalid_flag_combinations_exit_2(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv + ["--dry-run"])
    assert exc.value.code == 2


def test_help_and_version_and_stage_list():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = io.StringIO()
    assert main(["--version"], stream=out) == 0 and "model-abliteration-pipeline" in out.getvalue()
    out = io.StringIO()
    assert main(["--list-stages"], stream=out) == 0
    assert out.getvalue().splitlines()[5].startswith("05  abliterate")


def test_invalid_model_url_is_a_usage_error(tmp_path):
    assert main(["--config", "none", "--model-url", "http://huggingface.co/a/b", "--dry-run", "--output-dir", str(tmp_path)]) == ExitCode.USAGE


def test_unknown_stage_is_a_usage_error(tmp_path):
    code = main(["--config", "none", "--model-url", "a/b", "--from", "abliterat", "--dry-run", "--output-dir", str(tmp_path)])
    assert code == ExitCode.USAGE


def test_print_config_reflects_overrides(tmp_path):
    out = io.StringIO()
    assert main(["--config", str(ROOT / "configs" / "pipeline.yaml"), "--max-trials", "7", "--print-config", "--output-dir", str(tmp_path)], stream=out) == 0
    printed = yaml.safe_load(out.getvalue().split("# config_hash")[0])
    assert printed["abliteration"]["max_trials"] == 7


def test_non_interactive_run_without_yes_requires_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    code = main(["--config", "none", "--model-url", "a/b", "--output-dir", str(tmp_path / "ws")])
    assert code == ExitCode.CONFIRMATION_REQUIRED
    assert not (tmp_path / "ws").exists()
