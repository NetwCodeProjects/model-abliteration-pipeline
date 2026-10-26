"""Subprocess runner (budgets, interrupts, redaction), environment detection/profiles, components, planning."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest
from conftest import FakeProbe, fake_component_records

from model_pipeline import configspec
from model_pipeline.config import build_config
from model_pipeline.environment import get_profile
from model_pipeline.environment.components import ComponentManager
from model_pipeline.environment.detect import HostProbe, classify_storage, cuda_arch_from_capability, detect_profile
from model_pipeline.errors import ExitCode, PipelineError
from model_pipeline.fsutil import atomic_write_json
from model_pipeline.orchestrator import RunOptions, affected_stages, selected_stage_ids
from model_pipeline.planning import components_for_config, estimate_disk
from model_pipeline.stages import STAGES, stage_by_ref
from model_pipeline.subprocesses import CommandRunner
from model_pipeline.telemetry import EventLogger, Redactor

CHILD = r"""
import signal, sys, time
def stop(*_):
    print("stopping gracefully", flush=True)
    sys.exit(0)
signal.signal(signal.SIGINT, stop)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, stop)
print("token=hf_abcdefghijklmnopqrstuvwxyz123456", flush=True)
for _ in range(600):
    time.sleep(0.05)
"""


def runner(event=None):
    return CommandRunner(Redactor(), event or threading.Event())


def test_runner_redacts_and_reports_success(tmp_path):
    result = runner().run([sys.executable, "-c", "print('hello hf_abcdefghijklmnopqrstuvwxyz0123')"], cwd=None, env=None, log_dir=tmp_path, name="ok")
    assert result.ok and result.returncode == 0
    text = Path(result.stdout_log).read_text()
    assert "hf_***REDACTED***" in text and "abcdefghijkl" not in text


def test_runner_budget_sends_graceful_interrupt(tmp_path):
    started = time.monotonic()
    result = runner().run([sys.executable, "-c", CHILD], cwd=None, env=None, log_dir=tmp_path, name="budget", budget_seconds=1.0, grace_seconds=20)
    assert result.budget_stopped and not result.killed
    assert "stopping gracefully" in Path(result.stdout_log).read_text()
    assert time.monotonic() - started < 15


def test_runner_hard_timeout(tmp_path):
    result = runner().run(
        [sys.executable, "-c", "import time; time.sleep(60)"], cwd=None, env=None, log_dir=tmp_path, name="slow", timeout_seconds=1.0, kill_after_seconds=2
    )
    assert result.timed_out and not result.ok


def test_runner_forwards_interrupt_event(tmp_path):
    event = threading.Event()
    threading.Timer(1.0, event.set).start()
    result = runner(event).run([sys.executable, "-c", CHILD], cwd=None, env=None, log_dir=tmp_path, name="int")
    assert result.interrupted and not result.ok


def test_missing_executable_is_an_environment_error(tmp_path):
    with pytest.raises(PipelineError) as err:
        runner().run(["definitely-not-a-real-binary-xyz"], cwd=None, env=None, log_dir=tmp_path, name="missing")
    assert err.value.code == ExitCode.ENVIRONMENT


def test_profile_detection_never_equates_cuda_with_lambda_stack():
    assert detect_profile(FakeProbe(lambda_stack=True), "auto")[0] == "lambda-stack"
    assert detect_profile(FakeProbe(lambda_stack=False), "auto")[0] == "ubuntu"  # CUDA present, no Lambda markers
    assert detect_profile(FakeProbe(), "lambda-stack")[0] == "lambda-stack"
    with pytest.raises(PipelineError) as err:
        detect_profile(FakeProbe(platform="win32"), "auto")
    assert err.value.error_id == "E_ENV_UNSUPPORTED_PLATFORM"

    class Fedora(FakeProbe):
        def os_release(self):
            return {"ID": "fedora", "PRETTY_NAME": "Fedora 41"}

    with pytest.raises(PipelineError) as err:
        detect_profile(Fedora(), "auto")
    assert err.value.error_id == "E_ENV_AMBIGUOUS"


def test_probe_reads_dpkg_status_without_executing(tmp_path):
    (tmp_path / "var/lib/dpkg").mkdir(parents=True)
    (tmp_path / "var/lib/dpkg/status").write_text(
        "Package: lambda-stack-cuda\nStatus: install ok installed\n\nPackage: lambda-other\nStatus: deinstall ok config-files\n"
    )
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc/os-release").write_text('ID=ubuntu\nPRETTY_NAME="Ubuntu 22.04"\n')
    probe = HostProbe(allow_exec=False, root=str(tmp_path))
    assert probe.dpkg_packages(("lambda-stack",)) == ["lambda-stack-cuda"]
    assert probe.os_release()["ID"] == "ubuntu"
    assert probe.gpus() is None  # no execution in static mode


def test_storage_classification_and_cuda_arch():
    assert classify_storage({"fstype": "nfs4", "mountpoint": "/lambda/nfs/x"})["kind"] == "network-or-persistent-candidate"
    assert classify_storage({"fstype": "ext4", "mountpoint": "/"})["kind"] == "instance-local"
    assert cuda_arch_from_capability("10.0") == "100" and cuda_arch_from_capability("9.0") == "90" and cuda_arch_from_capability(None) is None


def cfg(tmp_path, **changes):
    import yaml

    data = {"model": {"url": "a/b"}, **changes}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(data))
    return build_config(path)


def test_ubuntu_profile_guards_host_changes(tmp_path):
    probe = FakeProbe(tools={"git"}, gpus=False)
    probe.tools.discard("nvidia-smi")
    config = cfg(tmp_path)
    assessment = get_profile("ubuntu", probe).assess({"heretic", "llama_cpp"}, config.data)
    assert any("nvidia-smi" in b or "NVIDIA driver" in b for b in assessment.blockers)
    action = assessment.actions[0]
    assert action.flag == "environment.allow_system_packages" and action.argv[:2] == ["sudo", "-n"]
    assert {"cmake", "build-essential"} <= set(action.argv)
    allowed = cfg(tmp_path, environment={"allow_driver_changes": True})
    assessment = get_profile("ubuntu", probe).assess({"heretic"}, allowed.data)
    assert not assessment.blockers
    assert any(a.reboot_required and "ubuntu-drivers" in a.argv for a in assessment.actions)


def test_lambda_profile_never_installs_drivers(tmp_path):
    probe = FakeProbe(lambda_stack=True, tools={"git", "cmake", "g++", "make"})
    assessment = get_profile("lambda-stack", probe).assess({"heretic"}, cfg(tmp_path, environment={"allow_driver_changes": True}).data)
    assert assessment.blockers and not any("ubuntu-drivers" in a.argv for a in assessment.actions)


def test_components_are_reused_only_after_successful_validation(tmp_path):
    config = cfg(tmp_path)
    probe = FakeProbe()
    records = fake_component_records(config, probe)
    tools = tmp_path / "tools"
    atomic_write_json(tools / "components.json", {"components": {n: r.to_dict() for n, r in records.items()}})
    logger = EventLogger(None, Redactor(), "warning")
    manager = ComponentManager(tools, runner(), logger, tmp_path / "logs", None, Path("."), tmp_path / "cache", probe)
    record, action = manager.ensure("fp8", records["fp8"].spec, allow_install=False, gpu_smoke=True)
    assert action == "reused" and record.validated_utc
    broken = records["vllm"]
    broken.validation_commands = [{"argv": [sys.executable, "-c", "import sys; sys.exit(3)"]}]
    atomic_write_json(tools / "components.json", {"components": {n: r.to_dict() for n, r in {**records, "vllm": broken}.items()}})
    with pytest.raises(PipelineError) as err:
        manager.ensure("vllm", broken.spec, allow_install=False, gpu_smoke=False)
    assert err.value.error_id == "E_COMPONENT_MISSING"
    with pytest.raises(PipelineError) as err:
        manager.ensure("fp8", {**records["fp8"].spec, "packages": ["llmcompressor==0.0.1"]}, allow_install=True, gpu_smoke=False)
    assert err.value.error_id == "E_UV_MISSING", "fingerprint change triggers (re)installation"


def test_stage_selection_and_references():
    assert selected_stage_ids(RunOptions(setup_only=True)) == ["00", "01"]
    assert selected_stage_ids(RunOptions(through_stage="2")) == ["00", "01", "02"]
    assert selected_stage_ids(RunOptions(from_stage="5", through_stage="quantize")) == ["05", "06", "07", "08"]
    assert selected_stage_ids(RunOptions(only_stage="abliterate")) == ["05"]
    with pytest.raises(PipelineError):
        selected_stage_ids(RunOptions(from_stage="8", through_stage="2"))
    assert [s.id for s in STAGES] == [f"{i:02d}" for i in range(11)]
    assert stage_by_ref(10).name == "report_export" and stage_by_ref("verify_exports").id == "09"


def test_dependency_graph_by_mode(tmp_path):
    full = cfg(tmp_path)
    qonly = cfg(tmp_path, abliteration={"enabled": False})
    assert stage_by_ref("08").dependencies(full) == ("01", "07")
    assert stage_by_ref("08").dependencies(qonly) == ("01", "02")
    assert stage_by_ref("05").applicable(qonly)[0] is False
    assert components_for_config(full) == {"heretic", "fp8", "vllm", "llama_cpp"}
    assert components_for_config(qonly) == {"fp8", "vllm", "llama_cpp"}
    assert affected_stages(["quantization.formats"]) == ["08", "09", "10"]
    assert affected_stages(["model.url"])[0] == "02"


def test_disk_estimate_is_unknown_without_metadata(tmp_path):
    config = cfg(tmp_path)
    assert estimate_disk(config, None)["total_bytes"] is None
    est = estimate_disk(config, 55 * 1024**3)
    labels = [k for k, _ in est["items"]]
    assert any("iq3_s" in k for k in labels) and est["total_bytes"] > 2 * 55 * 1024**3


def test_pins_are_consistent():
    assert len(configspec.HERETIC_PINNED_REVISION) == 40 and len(configspec.LLAMA_CPP_PINNED_COMMIT) == 40
