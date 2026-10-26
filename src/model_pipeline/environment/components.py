"""Project-local tool components: pinned, fingerprinted, validated, reusable across runs.

A component is reused only when its manifest record has the same fingerprint *and* its
validation commands succeed right now (file existence is never treated as proof). Missing or
broken components are (re)installed into fingerprint-specific directories.

Components:
  heretic   - venv with torch (CUDA wheels via uv --torch-backend) + Heretic at a pinned commit,
              installed with constraints exported from Heretic's own uv.lock.
  fp8       - venv with llm-compressor / compressed-tensors (conflicts with Heretic's transformers pin).
  vllm      - venv with vLLM for FP8 loader smoke tests.
  llama_cpp - llama.cpp source build at a pinned tag + converter venv.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .. import configspec
from ..errors import ExitCode, PipelineError
from ..fsutil import atomic_write_json, read_json, remove_tree, sha256_obj
from ..subprocesses import CommandRunner, bin_in_venv, child_env, python_in_venv
from ..telemetry import EventLogger, utc_now
from .detect import HostProbe, cuda_arch_from_capability

COMPONENTS_FILE = "components.json"
TORCH_FAMILY_RE = re.compile(r"^(torch|torchvision|torchaudio|triton|pytorch-triton|nvidia-[a-z0-9-]+)\s*(==|>=|<=|~=|@)", re.I)
LLAMA_TARGETS = ["llama-quantize", "llama-imatrix", "llama-completion", "llama-mtmd-cli"]


@dataclass
class ComponentRecord:
    name: str
    fingerprint: str
    spec: dict[str, Any]
    commands: dict[str, list[str]]
    paths: dict[str, str] = field(default_factory=dict)
    versions: dict[str, Any] = field(default_factory=dict)
    build: dict[str, Any] = field(default_factory=dict)
    validation_commands: list[dict[str, Any]] = field(default_factory=list)
    installed_utc: str | None = None
    validated_utc: str | None = None
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComponentRecord:
        return cls(**data)

    def cmd(self, name: str) -> list[str]:
        if name not in self.commands:
            raise PipelineError(ExitCode.ENVIRONMENT, "E_COMPONENT_COMMAND_MISSING", f"Component {self.name} has no command {name!r}")
        return list(self.commands[name])

    @property
    def scripts_dir(self) -> Path:
        return Path(self.paths["scripts_dir"])

    def script(self, script_name: str) -> list[str]:
        return [*self.cmd("python"), str(self.scripts_dir / script_name)]


def component_specs(config: dict[str, Any], torch_plan: dict[str, Any], probe: HostProbe | None) -> dict[str, dict[str, Any]]:
    """Desired pins/strategy for every component (the fingerprint input)."""
    env = config["environment"]
    abl = config["abliteration"]
    fp8 = config["quantization"]["fp8"]
    gguf = config["quantization"]["gguf"]
    platform_tag = f"{sys.platform}-{os.uname().machine if hasattr(os, 'uname') else 'unknown'}"
    common = {"python_version": env["python_version"], "torch": torch_plan, "platform": platform_tag}
    cuda_arch = gguf["cuda_architectures"]
    if cuda_arch == "auto" and probe is not None:
        gpus = probe.gpus() or []
        archs = sorted({a for a in (cuda_arch_from_capability(g.get("compute_capability")) for g in gpus) if a})
        cuda_arch = ";".join(archs) if archs else "auto"
    return {
        "heretic": {
            **common,
            "repo": abl["engine_repo"],
            "revision": abl["engine_revision"] or configspec.HERETIC_PINNED_REVISION,
        },
        "fp8": {
            **common,
            "packages": [f"llmcompressor=={fp8['llmcompressor_version']}", f"compressed-tensors=={fp8['compressed_tensors_version']}"],
        },
        "vllm": {**common, "torch": {"source": "pip", "backend": None}, "packages": [f"vllm=={fp8['vllm_version']}"]},
        "llama_cpp": {
            "platform": platform_tag,
            "python_version": env["python_version"],
            "repo": gguf["llama_cpp_repo"],
            "revision": gguf["llama_cpp_revision"] or configspec.LLAMA_CPP_PINNED_REVISION,
            "build_cuda": gguf["build_cuda"],
            "cuda_architectures": cuda_arch,
            "converter_transformers": gguf["converter_transformers"],
        },
    }


class ComponentManager:
    def __init__(
        self,
        tools_dir: Path,
        runner: CommandRunner,
        logger: EventLogger,
        log_dir: Path,
        uv_executable: str | None,
        scripts_dir: Path,
        cache_root: Path,
        probe: HostProbe,
    ) -> None:
        self.tools_dir = tools_dir
        self.runner = runner
        self.logger = logger
        self.log_dir = log_dir
        self.uv = uv_executable
        self.scripts_dir = scripts_dir
        self.cache_root = cache_root
        self.probe = probe
        self.manifest_path = tools_dir / COMPONENTS_FILE

    # ------------------------------------------------------------------ manifest
    def load(self) -> dict[str, ComponentRecord]:
        if not self.manifest_path.exists():
            return {}
        data = read_json(self.manifest_path)
        return {name: ComponentRecord.from_dict(rec) for name, rec in data.get("components", {}).items()}

    def _save(self, records: dict[str, ComponentRecord]) -> None:
        atomic_write_json(
            self.manifest_path,
            {"schema_version": "1.0", "updated_utc": utc_now(), "components": {n: r.to_dict() for n, r in records.items()}},
        )

    def get(self, name: str) -> ComponentRecord:
        records = self.load()
        if name not in records:
            raise PipelineError(
                ExitCode.DEPENDENCY,
                "E_COMPONENT_NOT_PROVISIONED",
                f"Component {name!r} is not provisioned",
                hint="Run stage 01 (environment), e.g. ./main.sh --resume RUN_ID --from 1, or ./main.sh --setup-only.",
            )
        return records[name]

    # ---------------------------------------------------------------- validation
    def _env(self) -> dict[str, str]:
        return child_env(UV_CACHE_DIR=str(self.cache_root / "uv"), HF_HOME=str(self.cache_root / "hf"))

    def validate(self, record: ComponentRecord, gpu_smoke: bool) -> dict[str, Any]:
        results = []
        for index, check in enumerate(record.validation_commands):
            argv = list(check["argv"])
            if check.get("gpu_flag") and gpu_smoke:
                argv.append(check["gpu_flag"])
            result = self.runner.run(argv, cwd=None, env=self._env(), log_dir=self.log_dir, name=f"validate-{record.name}-{index}", timeout_seconds=600)
            # Search the complete captured output: help texts print "usage" first, beyond the kept tail
            # (found with the real llama-quantize).
            text = "\n".join(Path(log).read_text(encoding="utf-8", errors="replace") for log in (result.stdout_log, result.stderr_log))
            ok = result.returncode in check.get("returncodes", [0])
            if ok and check.get("expect"):
                ok = check["expect"].lower() in text.lower()
            payload = None
            if ok and check.get("json"):
                payload = _last_json_line(result.tail)
                ok = bool(payload and payload.get("ok"))
            results.append({"argv": result.argv, "ok": ok, "returncode": result.returncode, "payload": payload})
            if not ok:
                return {"ok": False, "checks": results, "failed": result.argv, "tail": result.tail[-15:]}
        return {"ok": True, "checks": results}

    # ------------------------------------------------------------------ ensure
    def ensure(self, name: str, spec: dict[str, Any], *, allow_install: bool, gpu_smoke: bool) -> tuple[ComponentRecord, str]:
        """Return (record, action) where action is 'reused' or 'installed'."""
        fingerprint = sha256_obj({"name": name, "spec": spec})
        records = self.load()
        existing = records.get(name)
        if existing is not None and existing.fingerprint == fingerprint:
            outcome = self.validate(existing, gpu_smoke)
            if outcome["ok"]:
                existing.validated_utc = utc_now()
                existing.validation = outcome
                records[name] = existing
                self._save(records)
                return existing, "reused"
            self.logger.warning("component.invalid", f"Component {name} failed validation; it will be reinstalled", details=outcome)
        if not allow_install:
            raise PipelineError(
                ExitCode.ENVIRONMENT,
                "E_COMPONENT_MISSING",
                f"Component {name} is missing or invalid and environment.install_missing is false",
                hint="Set environment.install_missing: true or provision it manually (docs/deployment-*.md).",
            )
        installer = {
            "heretic": self._install_heretic,
            "fp8": self._install_package_env,
            "vllm": self._install_package_env,
            "llama_cpp": self._install_llama_cpp,
        }[name]
        self.logger.info("component.install", f"Installing component {name} (fingerprint {fingerprint[:12]})")
        record = installer(name, spec, fingerprint)
        record.installed_utc = utc_now()
        outcome = self.validate(record, gpu_smoke)
        record.validation = outcome
        if not outcome["ok"]:
            raise PipelineError(
                ExitCode.ENVIRONMENT,
                "E_COMPONENT_VALIDATION_FAILED",
                f"Component {name} was installed but failed validation: {outcome.get('failed')}",
                hint="See logs/<stage>/validate-*.log; record the fix in docs/implementation-log.md.",
                details=outcome,
            )
        record.validated_utc = utc_now()
        records = self.load()
        records[name] = record
        self._save(records)
        return record, "installed"

    # --------------------------------------------------------------- installers
    def _uv(self) -> str:
        if self.uv:
            return self.uv
        raise PipelineError(
            ExitCode.ENVIRONMENT,
            "E_UV_MISSING",
            "uv is required to provision tool environments but was not found",
            hint="Bootstrap the control environment (./main.sh without --dry-run) which ships a pinned uv, or install uv.",
        )

    def _run(self, argv: list[str], name: str, cwd: Path | None = None, timeout: float = 7200) -> None:
        result = self.runner.run(argv, cwd=cwd, env=self._env(), log_dir=self.log_dir, name=name, timeout_seconds=timeout)
        if not result.ok:
            raise PipelineError(
                ExitCode.ENVIRONMENT,
                "E_INSTALL_STEP_FAILED",
                f"Install step '{name}' failed (exit {result.returncode}): {' '.join(result.argv)}",
                hint=f"See {result.stderr_log}. Last lines:\n    " + "\n    ".join(result.tail[-8:]),
            )

    def _capture(self, argv: list[str], cwd: Path | None = None) -> str:
        try:
            return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL).stdout.strip()
        except (subprocess.SubprocessError, OSError) as exc:
            raise PipelineError(ExitCode.ENVIRONMENT, "E_INSTALL_STEP_FAILED", f"Command failed: {' '.join(argv)}: {exc}") from exc

    def _git_checkout(self, repo: str, revision: str, dest: Path, label: str) -> str:
        if dest.exists():
            try:
                head = self._capture(["git", "-C", str(dest), "rev-parse", "HEAD"])
                wanted = self._capture(["git", "-C", str(dest), "rev-parse", f"{revision}^{{commit}}"])
                status = self._capture(["git", "-C", str(dest), "status", "--porcelain"])
                if head == wanted and not status:
                    return head
            except PipelineError:
                pass
            remove_tree(dest)
        partial = dest.with_name(dest.name + ".partial")
        remove_tree(partial)
        self._run(["git", "clone", "--filter=blob:none", "--no-checkout", repo, str(partial)], f"{label}-git-clone")
        self._run(["git", "-C", str(partial), "checkout", "--detach", revision], f"{label}-git-checkout")
        os.replace(partial, dest)
        return self._capture(["git", "-C", str(dest), "rev-parse", "HEAD"])

    def _make_venv(self, venv: Path, python_version: str, system_site: bool, label: str) -> Path:
        remove_tree(venv)
        argv = [self._uv(), "venv", "--python", python_version, str(venv)]
        if system_site:
            argv.append("--system-site-packages")
        self._run(argv, f"{label}-venv")
        return python_in_venv(venv)

    def _torch_args(self, torch_plan: dict[str, Any]) -> list[str]:
        backend = torch_plan.get("backend")
        return ["--torch-backend", backend] if torch_plan.get("source") == "pip" and backend else []

    def _python_versions(self, python: Path, modules: list[str]) -> dict[str, Any]:
        code = (
            "import importlib.metadata as m, json, sys\n"
            f"mods = {modules!r}\n"
            "out = {'python': sys.version.split()[0]}\n"
            "for name in mods:\n"
            "    try:\n        out[name] = m.version(name)\n"
            "    except Exception:\n        out[name] = None\n"
            "print(json.dumps(out))\n"
        )
        try:
            return _last_json_line(self._capture([str(python), "-c", code]).splitlines()) or {}
        except PipelineError:
            return {}

    def _install_heretic(self, name: str, spec: dict[str, Any], fingerprint: str) -> ComponentRecord:
        src = self.tools_dir / "heretic" / f"src-{spec['revision'][:12]}"
        commit = self._git_checkout(spec["repo"], spec["revision"], src, "heretic")
        constraints = self.tools_dir / "heretic" / f"constraints-{commit[:12]}.txt"
        raw = self.tools_dir / "heretic" / f"constraints-{commit[:12]}.raw.txt"
        self._run(
            [self._uv(), "export", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes", "--project", str(src), "-o", str(raw)],
            "heretic-uv-export",
        )
        kept, dropped = filter_constraints(raw.read_text(encoding="utf-8"))
        constraints.write_text(kept, encoding="utf-8")
        venv = self.tools_dir / "envs" / f"heretic-{fingerprint[:12]}"
        torch_plan = spec["torch"]
        python = self._make_venv(venv, spec["python_version"], torch_plan.get("source") == "system", "heretic")
        torch_pins = locked_torch_pins(dropped)
        if torch_plan.get("source") == "pip":
            # Same torch/torchvision versions as Heretic's uv.lock, but the CUDA variant chosen by --torch-backend.
            self._run([self._uv(), "pip", "install", "--python", str(python), *self._torch_args(torch_plan), *torch_pins], "heretic-torch")
        self._run(
            [self._uv(), "pip", "install", "--python", str(python), *self._torch_args(torch_plan), "-c", str(constraints), str(src)],
            "heretic-install",
        )
        versions = self._python_versions(python, ["heretic-llm", "torch", "transformers", "optuna", "peft", "datasets", "accelerate"])
        return ComponentRecord(
            name=name,
            fingerprint=fingerprint,
            spec=spec,
            commands={"python": [str(python)], "heretic": [str(bin_in_venv(venv, "heretic"))]},
            paths={"venv": str(venv), "src": str(src), "constraints": str(constraints), "scripts_dir": str(self.scripts_dir)},
            versions={**versions, "heretic_commit": commit, "dropped_constraints": dropped, "torch_pins": torch_pins},
            validation_commands=[
                {"argv": [str(python), str(self.scripts_dir / "env_check.py"), "--component", "heretic"], "json": True, "gpu_flag": "--gpu-smoke"},
                {"argv": [str(bin_in_venv(venv, "heretic")), "--help"], "expect": "--model"},
            ],
        )

    def _install_package_env(self, name: str, spec: dict[str, Any], fingerprint: str) -> ComponentRecord:
        venv = self.tools_dir / "envs" / f"{name}-{fingerprint[:12]}"
        python = self._make_venv(venv, spec["python_version"], False, name)
        self._run([self._uv(), "pip", "install", "--python", str(python), *self._torch_args(spec["torch"]), *spec["packages"]], f"{name}-install")
        modules = {"fp8": ["llmcompressor", "compressed-tensors", "torch", "transformers"], "vllm": ["vllm", "torch", "transformers"]}[name]
        return ComponentRecord(
            name=name,
            fingerprint=fingerprint,
            spec=spec,
            commands={"python": [str(python)]},
            paths={"venv": str(venv), "scripts_dir": str(self.scripts_dir)},
            versions=self._python_versions(python, modules),
            validation_commands=[{"argv": [str(python), str(self.scripts_dir / "env_check.py"), "--component", name], "json": True, "gpu_flag": "--gpu-smoke"}],
        )

    def _install_llama_cpp(self, name: str, spec: dict[str, Any], fingerprint: str) -> ComponentRecord:
        base = self.tools_dir / "llama.cpp"
        src = base / f"src-{_safe(spec['revision'])}"
        commit = self._git_checkout(spec["repo"], spec["revision"], src, "llama-cpp")
        if spec["revision"] == configspec.LLAMA_CPP_PINNED_REVISION and commit != configspec.LLAMA_CPP_PINNED_COMMIT:
            raise PipelineError(
                ExitCode.ENVIRONMENT,
                "E_PIN_MISMATCH",
                f"Tag {spec['revision']} resolved to {commit}, expected {configspec.LLAMA_CPP_PINNED_COMMIT}",
                hint="The upstream tag moved; investigate before trusting the build.",
            )
        nvcc = self.probe.which("nvcc")
        build_cuda = spec["build_cuda"] == "on" or (spec["build_cuda"] == "auto" and nvcc is not None)
        if spec["build_cuda"] == "on" and nvcc is None:
            raise PipelineError(ExitCode.ENVIRONMENT, "E_NVCC_MISSING", "quantization.gguf.build_cuda=on but nvcc was not found")
        build = base / f"build-{fingerprint[:12]}"
        remove_tree(build)
        flags = [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_SERVER=OFF",
            "-DLLAMA_OPENSSL=OFF",
        ]
        if build_cuda:
            flags.append("-DGGML_CUDA=ON")
            if spec["cuda_architectures"] not in (None, "auto"):
                flags.append(f"-DCMAKE_CUDA_ARCHITECTURES={spec['cuda_architectures']}")
        self._run(["cmake", "-S", str(src), "-B", str(build), *flags], "llama-cpp-configure")
        jobs = str(max(1, (os.cpu_count() or 2) - 1))
        self._run(["cmake", "--build", str(build), "--config", "Release", "-j", jobs, "--target", *LLAMA_TARGETS], "llama-cpp-build", timeout=14400)
        bin_dir = build / "bin"
        conv_venv = self.tools_dir / "envs" / f"llama-convert-{fingerprint[:12]}"
        conv_python = self._make_venv(conv_venv, spec["python_version"], False, "llama-convert")
        self._run(
            [
                self._uv(),
                "pip",
                "install",
                "--python",
                str(conv_python),
                "--index-strategy",
                "unsafe-best-match",
                "-r",
                str(src / "requirements" / "requirements-convert_hf_to_gguf.txt"),
            ],
            "llama-convert-reqs",
        )
        self._run([self._uv(), "pip", "install", "--python", str(conv_python), str(src / "gguf-py")], "llama-convert-gguf-py")
        if spec.get("converter_transformers"):
            self._run(
                [self._uv(), "pip", "install", "--python", str(conv_python), f"transformers=={spec['converter_transformers']}"],
                "llama-convert-transformers",
            )
        compiler = ""
        try:
            compiler = self._capture(["c++", "--version"]).splitlines()[0]
        except (PipelineError, IndexError):
            pass
        commands = {tool: [str(bin_dir / tool)] for tool in LLAMA_TARGETS}
        commands["convert"] = [str(conv_python), str(src / "convert_hf_to_gguf.py")]
        commands["python"] = [str(conv_python)]
        return ComponentRecord(
            name=name,
            fingerprint=fingerprint,
            spec=spec,
            commands=commands,
            paths={"src": str(src), "build": str(build), "converter_venv": str(conv_venv), "scripts_dir": str(self.scripts_dir)},
            versions={"commit": commit, **self._python_versions(conv_python, ["gguf", "transformers", "torch"])},
            build={"cmake_flags": flags, "cuda": build_cuda, "nvcc": self.probe.nvcc_version() if build_cuda else None, "compiler": compiler, "jobs": jobs},
            validation_commands=[
                {"argv": [*commands["llama-completion"], "--version"], "returncodes": [0]},
                {"argv": [*commands["llama-quantize"], "--help"], "returncodes": [0, 1], "expect": "usage"},
                {"argv": [*commands["convert"], "--print-supported-models"], "returncodes": [0]},
            ],
        )


def filter_constraints(text: str) -> tuple[str, list[str]]:
    """Drop torch-family pins (installed separately for the right CUDA) and option lines."""
    kept, dropped = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(("-", ".", "/")) or TORCH_FAMILY_RE.match(stripped):
            dropped.append(stripped)
            continue
        kept.append(stripped)
    return "\n".join(kept) + "\n", dropped


def locked_torch_pins(dropped: list[str]) -> list[str]:
    """['torch==2.14.1', 'torchvision==0.29.1'] from the engine lock's non-macOS lines (local +cpu tag removed)."""
    pins: dict[str, str] = {}
    for line in dropped:
        match = re.match(r"^(torch|torchvision)==([0-9][^ ;+]*)(?:\+[^ ;]+)?\s*(?:;(.*))?$", line)
        if not match:
            continue
        marker = (match.group(3) or "").replace(" ", "")
        if "sys_platform=='darwin'" in marker:
            continue
        pins[match.group(1)] = match.group(2)
    return [f"{name}=={pins[name]}" if name in pins else name for name in ("torch", "torchvision")]


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", text)


def _last_json_line(lines: list[str]) -> dict[str, Any] | None:
    import json

    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def find_uv(configured: str | None) -> str | None:
    if configured:
        return configured
    candidate = Path(sys.prefix) / ("Scripts/uv.exe" if sys.platform == "win32" else "bin/uv")
    if candidate.exists():
        return str(candidate)
    return shutil.which("uv")
