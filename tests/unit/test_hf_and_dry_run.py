"""Hugging Face reference normalization and the zero-side-effect guarantee of --dry-run."""

from __future__ import annotations

import builtins
import io
import os
import socket
import subprocess
import sys

import pytest
from conftest import ROOT, FakeProbe

from llm_surgeon.adapters.hf_hub import RepoFile, RepoInfo, normalize_model_ref, validate_revision, verify_snapshot
from llm_surgeon.cli import main
from llm_surgeon.errors import PipelineError
from llm_surgeon.orchestrator import Services


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B"),
        ("  Qwen/Qwen3.8-27B  ", "Qwen/Qwen3.8-27B"),
        ("https://huggingface.co/Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B"),
        ("https://huggingface.co/Qwen/Qwen3.8-27B/", "Qwen/Qwen3.8-27B"),
        ("https://www.huggingface.co/coder3101/Qwen3.8-27B-heretic-ara", "coder3101/Qwen3.8-27B-heretic-ara"),
        ("gpt2", "gpt2"),
    ],
)
def test_accepted_model_refs(text, expected):
    assert normalize_model_ref(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "http://huggingface.co/a/b",
        "https://huggingface.co.evil.com/a/b",
        "https://evil.com/huggingface.co/a/b",
        "https://user:pw@huggingface.co/a/b",
        "https://huggingface.co:8443/a/b",
        "https://huggingface.co/a/b?x=1",
        "https://huggingface.co/a/b#frag",
        "https://huggingface.co/a/b/tree/main",
        "https://huggingface.co/datasets/a/b",
        "https://huggingface.co/",
        "file:///etc/passwd",
        "a/b; rm -rf /",
        "$(touch /tmp/x)/b",
        "a/../b",
        "a--b/c",
        "./local/model",
        "/abs/path",
        "C:\\models\\x",
        "a/b/c",
        "a b/c",
        "a/" + "x" * 200,
        "",
    ],
)
def test_rejected_model_refs(text):
    with pytest.raises(PipelineError):
        normalize_model_ref(text)


@pytest.mark.parametrize("rev", ["main", "v1.0", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "refs/pr/1"])
def test_valid_revisions(rev):
    assert validate_revision(rev) == rev


@pytest.mark.parametrize("rev", ["--upload", "../x", "a b", "", "x" * 200])
def test_invalid_revisions(rev):
    with pytest.raises(PipelineError):
        validate_revision(rev)


def test_verify_snapshot_detects_corruption(tmp_path):
    good = tmp_path / "a.safetensors"
    good.write_bytes(b"hello")
    import hashlib

    info = RepoInfo("a/b", "model", "f" * 40, None, files=[RepoFile("a.safetensors", 5, hashlib.sha256(b"hello").hexdigest(), None)])
    assert verify_snapshot(tmp_path, info)["a.safetensors"]
    good.write_bytes(b"jello")
    with pytest.raises(PipelineError) as err:
        verify_snapshot(tmp_path, info)
    assert err.value.error_id == "E_DOWNLOAD_CORRUPT"
    info.files.append(RepoFile("missing.json", 3, None, "0" * 40))
    with pytest.raises(PipelineError):
        verify_snapshot(tmp_path, info)


def test_verify_snapshot_checks_git_blob_ids(tmp_path):
    (tmp_path / "config.json").write_bytes(b"{}\n")
    blob = subprocess.run(["git", "hash-object", str(tmp_path / "config.json")], capture_output=True, text=True).stdout.strip()
    if not blob:
        pytest.skip("git not available")
    assert verify_snapshot(tmp_path, RepoInfo("a/b", "model", "f" * 40, None, files=[RepoFile("config.json", 3, None, blob)]))


def test_dry_run_has_no_side_effects(tmp_path, monkeypatch):
    """--dry-run: no network, no subprocesses, no writes, no GPU/ML imports, no downloads."""
    workspace = tmp_path / "ws"
    calls = []

    def forbid(name):
        def _raise(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} called during dry-run")

        return _raise

    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            calls.append(f"open({file}, {mode})")
            raise AssertionError(f"write to {file} during dry-run")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", forbid("socket.connect"))
    monkeypatch.setattr(subprocess, "Popen", forbid("subprocess.Popen"))
    monkeypatch.setattr(subprocess, "run", forbid("subprocess.run"))
    monkeypatch.setattr(os, "makedirs", forbid("os.makedirs"))
    monkeypatch.setattr(os, "mkdir", forbid("os.mkdir"))
    monkeypatch.setattr(builtins, "open", guarded_open)
    heavy = {"torch", "transformers", "huggingface_hub", "vllm", "llmcompressor"}
    before = heavy & set(sys.modules)
    hub_calls = []
    services = Services(probe_factory=lambda allow_exec: FakeProbe(), hub_factory=lambda *a: hub_calls.append(a))
    out = io.StringIO()
    code = main(["--config", str(ROOT / "configs" / "pipeline.yaml"), "--dry-run", "--output-dir", str(workspace)], services=services, stream=out)
    assert code == 0, out.getvalue()
    assert calls == [] and hub_calls == []
    assert not workspace.exists()
    assert (heavy & set(sys.modules)) == before
    text = out.getvalue()
    assert "DRY RUN" in text and "Nothing was downloaded" in text and "05 abliterate" in text
    assert "This platform does not automatically verify" in text


def test_dry_run_reports_plan_details_for_quantize_only(tmp_path):
    out = io.StringIO()
    services = Services(probe_factory=lambda allow_exec: FakeProbe(lambda_stack=True))
    code = main(
        [
            "--config",
            str(ROOT / "configs" / "pipeline.yaml"),
            "--quantize-only",
            "--quantizations",
            "iq2_xxs",
            "--dry-run",
            "--output-dir",
            str(tmp_path / "ws"),
        ],
        services=services,
        stream=out,
    )
    text = out.getvalue()
    assert code == 0
    assert "lambda-stack" in text and "quantize-only" in text
    assert "skip: abliteration disabled" in text
    assert "WILL FAIL without imatrix: iq2_xxs" in text
