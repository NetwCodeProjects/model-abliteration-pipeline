"""GPU smoke tests for the B200 host. Never run in CI; select explicitly:

    GPU_WS=$PWD/workspace pytest -m gpu tests/gpu -x -s

They use the components provisioned by stage 01 (``./main.sh --setup-only --yes``) and the source
snapshot from stage 02 (``./main.sh --through 2 --yes``). Each test is bounded; the Heretic test runs a
*single small trial* and requires GPU_ALLOW_HERETIC_TRIAL=1 (explicit authorization of GPU time).
Nothing here is claimed as passing until it has actually been run on the B200.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu
WS = Path(os.environ.get("GPU_WS", "workspace"))
ROOT = Path(__file__).resolve().parents[2]


def components() -> dict:
    path = WS / "tools" / "components.json"
    if not path.exists():
        pytest.skip("run ./main.sh --setup-only --yes first")
    return json.loads(path.read_text())["components"]


def source_snapshot() -> Path:
    records = sorted((WS / "models" / "source").glob("*/*.json"))
    if not records:
        pytest.skip("run ./main.sh --through 2 --yes first")
    return Path(json.loads(records[-1].read_text())["snapshot_path"])


def run(argv, timeout=1800, **kw):
    print("$", " ".join(map(str, argv)))
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True, timeout=timeout, **kw)


def test_nvidia_smi_reports_b200():
    out = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"], timeout=60)
    assert out.returncode == 0, out.stderr
    print(out.stdout)
    assert "B200" in out.stdout or os.environ.get("GPU_ALLOW_OTHER"), "set GPU_ALLOW_OTHER=1 to run on other GPUs"


@pytest.mark.parametrize("name", ["heretic", "fp8", "vllm"])
def test_component_cuda_matmul(name):
    comp = components().get(name)
    if comp is None:
        pytest.skip(f"{name} not provisioned")
    out = run([*comp["commands"]["python"], ROOT / "scripts" / "runtime" / "env_check.py", "--component", name, "--gpu-smoke"], timeout=600)
    print(out.stdout[-2000:])
    payload = json.loads(out.stdout.strip().splitlines()[-1])
    assert payload["ok"], payload.get("error")
    assert payload.get("capability") == [10, 0] or os.environ.get("GPU_ALLOW_OTHER")


def test_source_model_loads_and_generates(tmp_path):
    comp = components()["heretic"]
    spec = {
        "model_path": str(source_snapshot()),
        "benign": [{"id": "s", "prompt": "What is 17 * 23? Reply with the number.", "expect_any": ["391"]}],
        "heldout_refusal_specs": [],
        "heldout_limit": 0,
        "extended_kl": False,
        "max_new_tokens": 32,
        "enable_thinking": False,
        "seed": 0,
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    run(
        [*comp["commands"]["python"], ROOT / "scripts" / "runtime" / "qualify.py", "--spec", tmp_path / "spec.json", "--out", tmp_path / "o.json"], timeout=3600
    )
    result = json.loads((tmp_path / "o.json").read_text())
    print(json.dumps({k: result.get(k) for k in ("load_ok", "load_seconds", "dtype", "peak_gpu_memory_bytes")}), result.get("benign"))
    assert result["load_ok"], result.get("error")
    assert result["benign"][0]["response"].strip()


def test_heretic_single_trial(tmp_path):
    if not os.environ.get("GPU_ALLOW_HERETIC_TRIAL"):
        pytest.skip("set GPU_ALLOW_HERETIC_TRIAL=1 to authorize one bounded Heretic trial on the GPU")
    config = tmp_path / "one-trial.yaml"
    config.write_text(
        "schema_version: '1.0'\n"
        "model: {url: Qwen/Qwen3.8-27B, revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}\n"
        "abliteration: {max_trials: 1, max_runtime_minutes: 120}\n"
        "evaluation: {qualification: fast}\n"
        "selection: {max_optimizer_refusal_rate: null, max_optimizer_kl: null, max_heldout_refusal_rate: null, min_benign_pass_rate: null}\n"
        "quantization: {enabled: false}\n"
        f"storage: {{root: '{WS}'}}\n"
    )
    out = run([ROOT / "main.sh", "--config", config, "--through", "7", "--yes", "--new-run"], timeout=4 * 3600, cwd=ROOT)
    print(out.stdout[-4000:], out.stderr[-4000:])
    assert out.returncode == 0


def test_fp8_and_gguf_exports_load(tmp_path):
    """Requires a finished run with fp8 and GGUF exports (./main.sh --yes)."""
    manifests = [json.loads(p.read_text()) for p in (WS / "manifests").glob("art-*.json")]
    fp8 = [m for m in manifests if m["precision"] == "fp8"]
    gguf = [m for m in manifests if m["artifact"]["format"] == "gguf"]
    if not fp8 and not gguf:
        pytest.skip("no exports yet")
    for m in fp8 + gguf:
        print(m["artifact_id"], m["precision"], m["validation"], m["inference"])
        assert m["validation"]["integrity_passed"]
        assert m["validation"]["inference_passed"] is True, f"{m['artifact_id']} loader test did not pass"


def test_llama_cpp_cuda_build():
    comp = components().get("llama_cpp")
    if comp is None:
        pytest.skip("llama_cpp not provisioned")
    assert comp["build"]["cuda"], "llama.cpp was built without CUDA (nvcc missing or build_cuda=off)"
    assert any("CMAKE_CUDA_ARCHITECTURES=100" in f for f in comp["build"]["cmake_flags"]) or os.environ.get("GPU_ALLOW_OTHER")
