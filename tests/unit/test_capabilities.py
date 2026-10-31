"""Step 6: capability policy, tool-environment activation, preservation of re-serialized files."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from conftest import FIXTURES

from llm_surgeon.config import build_config
from llm_surgeon.environment.capabilities import CATEGORIES, assess, native_fp8
from llm_surgeon.environment.components import ComponentRecord
from llm_surgeon.stages.stage09_verify import compare_files

A100 = {"name": "NVIDIA A100-SXM4-40GB", "compute_capability": "8.0", "memory_total_mib": 40960}
B300 = {"name": "NVIDIA B300", "compute_capability": "10.3", "memory_total_mib": 270000}
T4 = {"name": "Tesla T4", "compute_capability": "7.5", "memory_total_mib": 15360}


def cfg(**changes):
    overrides = {"model.url": "org/m", **changes}
    return build_config(None, overrides)


def rules(findings, category=None):
    return [f["rule"] for f in findings if category is None or f["category"] == category]


def test_a100_fp8_loader_test_is_unverified_not_failed():
    findings = assess(cfg(), [A100], needs_gpu=True, heretic=True, source_size=None)
    assert rules(findings, "unverified_execution") == ["fp8_vllm_weight_only"], "8.0 is an exercised capability"
    assert all(f["category"] in CATEGORIES and f["basis"] for f in findings)
    assert native_fp8([A100]) is False and native_fp8([B300]) is True and native_fp8([]) is None


def test_unexercised_hardware_is_reported_but_allowed():
    findings = assess(cfg(), [B300], needs_gpu=True, heretic=True, source_size=None)
    assert rules(findings) == ["hardware_not_exercised"] and "10.3" in findings[0]["message"]


def test_fp8_on_pre_ampere_is_unsupported_unless_the_test_is_off():
    assert "fp8_vllm_compute" in rules(assess(cfg(), [T4], needs_gpu=True, heretic=True, source_size=None), "unsupported_configuration")
    off = assess(cfg(**{"quantization.fp8.verify_backend": "none"}), [T4], needs_gpu=True, heretic=True, source_size=None)
    assert rules(off, "unsupported_configuration") == []


def test_vram_estimate_is_a_resource_shortage_not_a_verdict():
    findings = assess(cfg(), [A100], needs_gpu=True, heretic=True, source_size=54 * 1024**3)
    assert "vram_vs_checkpoint" in rules(findings, "estimated_resource_shortage")


def test_activation_puts_venv_bin_and_nvidia_libs_first(tmp_path):
    venv = tmp_path / "venv"
    lib = venv / "lib" / "python3.12" / "site-packages" / "nvidia" / "cuda_nvrtc" / "lib"
    lib.mkdir(parents=True)
    record = ComponentRecord(name="vllm", fingerprint="f", spec={}, commands={}, paths={"venv": str(venv)})
    env = record.activate({"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/opt/lib"})
    assert env["PATH"].split(os.pathsep)[0].startswith(str(venv)) and env["PATH"].endswith("/usr/bin")
    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [str(lib), "/opt/lib"] and env["VIRTUAL_ENV"] == str(venv)
    plain = ComponentRecord(name="llama_cpp", fingerprint="f", spec={}, commands={}, paths={})
    assert plain.activate({"PATH": "/usr/bin"}) == {"PATH": "/usr/bin"}


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def transformers5_save(tmp_path):
    """Source files of Qwen/Qwen3.5-2B and what transformers 5.18.0 saved (real processor files, synthetic tokenizer)."""
    real = FIXTURES / "transformers5_qwen35"
    src, out = tmp_path / "src", tmp_path / "out"
    shutil.copytree(real / "source", src)
    shutil.copytree(real / "saved", out)
    vocab, merges = {"a": 0, "b": 1, "ab": 2}, ["a b"]
    write_json(src / "vocab.json", vocab)
    (src / "merges.txt").write_text("#version: 0.2\na b\n", encoding="utf-8")
    added = [{"id": 3, "content": "<|im_start|>"}]
    write_json(src / "tokenizer.json", {"model": {"vocab": vocab, "merges": merges}, "added_tokens": added, "decoder": {"trim_offsets": False}})
    write_json(
        out / "tokenizer.json",
        {
            "model": {"vocab": vocab, "merges": [["a", "b"]]},
            "added_tokens": added + [{"id": 4, "content": "<|audio_start|>"}],
            "decoder": {"trim_offsets": True},
        },
    )
    return src, out


def test_transformers5_merges_are_reported_as_reserialized(transformers5_save):
    src, out = transformers5_save
    status, basis = compare_files(out, src)
    assert status["preprocessor_config.json"] == status["video_preprocessor_config.json"] == "re-serialized"
    assert status["vocab.json"] == status["merges.txt"] == "re-serialized"
    assert "processor_config.json['image_processor']" in basis["preprocessor_config.json"]
    assert status["tokenizer.json"] == "differs", "added tokens change tokenization; never called equivalent"
    assert "vocab identical" in basis["tokenizer.json"] and "<|audio_start|>" in basis["tokenizer.json"] and "decoder" in basis["tokenizer.json"]


def test_changed_processor_setting_is_still_missing(transformers5_save):
    src, out = transformers5_save
    saved = json.loads((out / "processor_config.json").read_text())
    key = next(k for k in json.loads((src / "preprocessor_config.json").read_text()) if k not in ("processor_class", "image_processor_type"))
    saved["image_processor"][key] = "changed"
    write_json(out / "processor_config.json", saved)
    assert compare_files(out, src)[0]["preprocessor_config.json"] == "missing"
