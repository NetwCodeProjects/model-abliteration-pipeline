"""Shared fixtures: tiny model, fake host probe, fake HF hub, fake tool components."""

from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_surgeon import configspec  # noqa: E402
from llm_surgeon.adapters.hf_hub import RepoFile, RepoInfo  # noqa: E402
from llm_surgeon.artifacts.safetensors_meta import write_safetensors  # noqa: E402
from llm_surgeon.environment.components import ComponentRecord, component_specs  # noqa: E402
from llm_surgeon.environment.detect import HostProbe  # noqa: E402
from llm_surgeon.fsutil import atomic_write_json, sha256_file, sha256_obj  # noqa: E402
from llm_surgeon.orchestrator import Services  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
FAKE_RUNTIME = FIXTURES / "fake_runtime"
FAKE_BIN = FIXTURES / "fake_bin"
SOURCE_SHA = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
REFERENCE_SHA = "7904b4f289b67f48430ad27f9e46b41acb82f0e2"


def make_tiny_model(path: Path, vision: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    config = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "text_config": {"dtype": "bfloat16"}}
    if vision:
        config["vision_config"] = {"depth": 1}
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer.json").write_text(json.dumps({"version": "1.0", "model": {"type": "BPE"}}))
    (path / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 262144}))
    (path / "chat_template.jinja").write_text("{% for m in messages %}{{ m['content'] }}{% endfor %}")
    (path / "generation_config.json").write_text(json.dumps({"do_sample": True}))
    if vision:
        (path / "preprocessor_config.json").write_text(json.dumps({"patch_size": 16}))

    def bf16(elements: int) -> bytes:
        return bytes(i % 251 for i in range(elements * 2))

    write_safetensors(
        path / "model-00001-of-00002.safetensors",
        {
            "model.language_model.layers.0.mlp.down_proj.weight": ("BF16", [4, 4], bf16(16)),
            "model.language_model.layers.0.self_attn.o_proj.weight": ("BF16", [4, 4], bf16(16)),
        },
    )
    write_safetensors(
        path / "model-00002-of-00002.safetensors",
        {
            "lm_head.weight": ("BF16", [8, 4], bf16(32)),
            "mtp.fc.weight": ("BF16", [4, 8], bf16(32)),
            "mtp.layers.0.mlp.down_proj.weight": ("BF16", [4, 4], bf16(16)),
        },
    )
    weight_map = {
        "model.language_model.layers.0.mlp.down_proj.weight": "model-00001-of-00002.safetensors",
        "model.language_model.layers.0.self_attn.o_proj.weight": "model-00001-of-00002.safetensors",
        "lm_head.weight": "model-00002-of-00002.safetensors",
        "mtp.fc.weight": "model-00002-of-00002.safetensors",
        "mtp.layers.0.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
    }
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}))
    return path


class FakeProbe(HostProbe):
    def __init__(
        self,
        lambda_stack: bool = False,
        tools: set[str] | None = None,
        gpus: bool = True,
        platform: str = "linux",
        gpu_name: str = "NVIDIA B200",
        compute_capability: str = "10.0",
    ) -> None:
        super().__init__(allow_exec=False)
        self.gpu_name, self.compute_capability = gpu_name, compute_capability
        self.lambda_stack = lambda_stack
        self.tools = tools if tools is not None else {"git", "cmake", "g++", "make", "nvidia-smi", "nvcc"}
        self._gpus = gpus
        self._platform = platform

    def platform(self) -> str:
        return self._platform

    def os_release(self) -> dict[str, str]:
        return {"ID": "ubuntu", "VERSION_ID": "24.04", "PRETTY_NAME": "Ubuntu 24.04.3 LTS"}

    def kernel(self) -> str:
        return "6.8.0-fake"

    def cpu(self) -> dict[str, Any]:
        return {"model": "Fake CPU", "logical_cpus": 224}

    def memory_total_bytes(self) -> int:
        return 2 * 1024**4

    def which(self, name: str) -> str | None:
        return f"/usr/bin/{name}" if name in self.tools else None

    def gpus(self):
        if not self._gpus:
            return []
        return [
            {
                "index": 0,
                "name": self.gpu_name,
                "memory_total_mib": 183359.0,
                "driver_version": "580.95.05",
                "compute_capability": self.compute_capability,
                "uuid": "GPU-fake",
                "ecc_uncorrected_volatile": "0",
            }
        ]

    def cuda_driver_version(self) -> str:
        return "13.0"

    def nvcc_version(self) -> str | None:
        return "12.9" if "nvcc" in self.tools else None

    def dpkg_packages(self, prefixes):
        return ["lambda-stack-cuda"] if self.lambda_stack else []

    def apt_source_files(self):
        return ["/etc/apt/sources.list.d/lambda-repository.list"] if self.lambda_stack else []

    def mounts(self):
        return [{"device": "/dev/nvme0n1p1", "mountpoint": "/", "fstype": "ext4"}]

    def is_root(self) -> bool:
        return False


class FakeHub:
    """In-memory Hub: repo_id -> (local dir, sha). Records every call for assertions."""

    def __init__(self, repos: dict[str, tuple[Path, str]], cache_dir: Path) -> None:
        self.repos = repos
        self.cache_dir = cache_dir
        self.calls: list[tuple[str, str]] = []

    def resolve(self, repo_id: str, revision: str | None, repo_type: str = "model") -> RepoInfo:
        self.calls.append(("resolve", repo_id))
        if repo_type == "dataset":
            return RepoInfo(repo_id, repo_type, revision or "d" * 40, revision, files=[])
        if repo_id not in self.repos:
            from llm_surgeon.errors import ExitCode, PipelineError

            raise PipelineError(ExitCode.ACQUIRE, "E_HF_NOT_FOUND", f"{repo_id} not found (fake)")
        path, sha = self.repos[repo_id]
        files = [RepoFile(p.relative_to(path).as_posix(), p.stat().st_size, sha256_file(p), None) for p in sorted(path.rglob("*")) if p.is_file()]
        return RepoInfo(repo_id, repo_type, sha, revision, files=files, card_data={"license": "apache-2.0"})

    def snapshot(self, repo_id: str, sha: str, repo_type: str = "model", allow_patterns=None) -> Path:
        self.calls.append(("snapshot", repo_id))
        target = self.cache_dir / f"{repo_type}s--{repo_id.replace('/', '--')}" / "snapshots" / sha
        if repo_type == "dataset":
            target.mkdir(parents=True, exist_ok=True)
            return target
        if not target.exists():
            shutil.copytree(self.repos[repo_id][0], target)
        return target

    def file_exists(self, repo_id, filename, sha, repo_type="model") -> bool:
        return (self.repos[repo_id][0] / filename).exists()

    def download_file(self, repo_id, filename, sha, repo_type="model") -> Path:
        self.calls.append(("download_file", f"{repo_id}/{filename}"))
        return self.repos[repo_id][0] / filename


def fake_component_records(config, probe: HostProbe) -> dict[str, ComponentRecord]:
    py = sys.executable
    specs = component_specs(config.data, {"source": "pip", "backend": config.get("environment.torch_backend")}, probe)

    def check(name):
        return [{"argv": [py, str(FAKE_RUNTIME / "env_check.py"), "--component", name], "json": True}]

    def rec(name, commands, **kw):
        return ComponentRecord(
            name=name,
            fingerprint=sha256_obj({"name": name, "spec": specs[name]}),
            spec=specs[name],
            commands=commands,
            paths={"scripts_dir": str(FAKE_RUNTIME)},
            validation_commands=kw.pop("validation", check(name)),
            **kw,
        )

    llama = lambda tool: [py, str(FAKE_BIN / "fake_llama.py"), tool]  # noqa: E731
    return {
        "heretic": rec(
            "heretic",
            {"python": [py], "heretic": [py, str(FAKE_BIN / "fake_heretic.py")]},
            versions={"heretic_commit": configspec.HERETIC_PINNED_REVISION, "transformers": "5.18.0"},
        ),
        "fp8": rec("fp8", {"python": [py]}, versions={"llmcompressor": "0.14.0.1", "compressed-tensors": "0.19.0"}),
        "vllm": rec("vllm", {"python": [py]}, versions={"vllm": "0.32.0"}),
        "llama_cpp": rec(
            "llama_cpp",
            {
                "python": [py],
                "convert": llama("convert"),
                "llama-quantize": llama("quantize"),
                "llama-imatrix": llama("imatrix"),
                "llama-completion": llama("completion"),
                "llama-mtmd-cli": llama("mtmd"),
            },
            versions={"commit": configspec.LLAMA_CPP_PINNED_COMMIT, "gguf": "0.19.0"},
            build={"cuda": False},
            validation=[{"argv": [*llama("completion"), "--version"], "returncodes": [0]}],
        ),
    }


def provision(workspace: Path, config, probe: HostProbe) -> None:
    records = fake_component_records(config, probe)
    atomic_write_json(workspace / "tools" / "components.json", {"schema_version": "1.0", "components": {n: r.to_dict() for n, r in records.items()}})


def fake_hub(tmp_path: Path) -> FakeHub:
    hub = tmp_path / "hub"
    return FakeHub(
        {"Qwen/Qwen3.8-27B": (hub / "Qwen--Qwen3.8-27B", SOURCE_SHA), "coder3101/Qwen3.8-27B-heretic-ara": (hub / "coder3101--ref", REFERENCE_SHA)},
        tmp_path / "hfcache",
    )


def fake_services(hub: FakeHub, probe: HostProbe) -> Services:
    return Services(
        probe_factory=lambda allow_exec: probe,
        hub_factory=lambda cache, token: hub,
        scripts_dir=FAKE_RUNTIME,
        gpu_query=lambda: None,
        confirm=lambda message: True,
    )


class Env:
    """Test environment wiring for in-process CLI runs (and ``run_killable`` for a separate process)."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.tmp = tmp_path
        self.workspace = tmp_path / "ws"
        self.model = make_tiny_model(tmp_path / "hub" / "Qwen--Qwen3.8-27B")
        self.reference = make_tiny_model(tmp_path / "hub" / "coder3101--ref")
        self.counters = tmp_path / "counters"
        self.counters.mkdir()
        monkeypatch.setenv("FAKE_COUNTER_DIR", str(self.counters))
        self.hub = fake_hub(tmp_path)
        self.probe = FakeProbe()
        self.services = fake_services(self.hub, self.probe)
        self.config_path = self.write_config({})
        self.base_args = ["--config", str(self.config_path), "--output-dir", str(self.workspace), "--yes", "--log-level", "warning"]

    def write_config(self, changes: dict, name: str = "pipeline.test.yaml") -> Path:
        """configs/pipeline.yaml with absolute profile paths, tiny disk floors and ``changes`` deep-merged."""
        import yaml

        from llm_surgeon.config import deep_merge

        data = yaml.safe_load((ROOT / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
        data["profiles"] = [str(ROOT / "configs" / p) for p in data.get("profiles", [])]
        data = deep_merge(data, {"environment": {"min_free_disk_gb": 0, "disk_buffer_gb": 0}})
        data = deep_merge(data, changes)
        path = self.tmp / name
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    def config(self, extra: list[str] | None = None):
        from llm_surgeon.cli import build_parser, config_path_from, to_overrides
        from llm_surgeon.config import build_config

        args = build_parser().parse_args(self.base_args + (extra or []))
        return build_config(config_path_from(args), to_overrides(args))

    def provision(self, extra: list[str] | None = None) -> None:
        provision(self.workspace, self.config(extra), self.probe)

    def run(self, *extra: str) -> tuple[int, str]:
        from llm_surgeon.cli import main

        stream = io.StringIO()
        code = main(self.base_args + list(extra), services=self.services, stream=stream)
        return code, stream.getvalue()

    def run_killable(self, *extra: str, **env_vars: str) -> int:
        """Run the CLI in a child process, so a fake tool can hard-kill it (FAKE_KILL_PARENT_AT)."""
        import os
        import subprocess

        driver = ROOT / "tests" / "integration" / "pipeline_driver.py"
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), **env_vars}
        proc = subprocess.run([sys.executable, str(driver), str(self.tmp), *self.base_args, *extra], env=env, capture_output=True, text=True, timeout=600)
        return proc.returncode

    def count(self, tool: str) -> int:
        path = self.counters / f"{tool}.count"
        return int(path.read_text()) if path.exists() else 0

    def latest_run(self) -> Path:
        runs = sorted((self.workspace / "runs").iterdir(), key=lambda p: p.stat().st_mtime)
        return runs[-1]

    def receipt(self, run_dir: Path, stage: str) -> dict:
        match = next((run_dir / "receipts").glob(f"{stage}-*.json"))
        return json.loads(match.read_text())

    def artifacts(self, run_dir: Path) -> dict:
        return json.loads((run_dir / "artifacts.json").read_text())["artifacts"]


@pytest.fixture
def env(tmp_path, monkeypatch) -> Env:
    for var in (
        "FAKE_HERETIC_MODE",
        "FAKE_QUANT_FAIL",
        "FAKE_MMPROJ_FAIL",
        "FAKE_FP8_FAIL",
        "FAKE_QUALIFY_REFUSE_ALL",
        "FAKE_TRIAL_SECONDS",
        "FAKE_TRIALS_FILE",
        "FAKE_KILL_PARENT_AT",
        "FAKE_KL_STEP",
        "FAKE_DEGRADE",
        "FAKE_BASELINE_FAIL",
    ):
        monkeypatch.delenv(var, raising=False)
    return Env(tmp_path, monkeypatch)
