"""Opt-in verification with the REAL pinned tools on CPU (no GPU needed, network needed).

Runs stages 00-10 against tiny-random/qwen3.8 (Qwen3_5ForConditionalGeneration, 28 MB) with real
Heretic (ARA), the real Optuna study helper, real transformers-based qualification and (optionally)
real llm-compressor FP8. Only the *host probe* is simulated (platform/GPU facts), because this test
is meant to run on development machines; it never claims GPU results.

Prepare once (uses the platform's own installer):
    REALTOOLS_WS=/path/ws python tests/realtools/provision_cpu.py
Run:
    REALTOOLS_WS=/path/ws python -m pytest tests/realtools -m realtools
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest
import yaml
from conftest import ROOT, FakeProbe

from model_pipeline.adapters.hf_hub import HubClient
from model_pipeline.cli import main
from model_pipeline.orchestrator import Services

pytestmark = pytest.mark.realtools
WS = os.environ.get("REALTOOLS_WS")
TINY = "tiny-random/qwen3.8"
TINY_SHA = "346bb76e05402f7bfd4186c2cf6de3d3956325be"


def tiny_template(path: Path) -> Path:
    text = (ROOT / "configs" / "heretic" / "optimize.toml").read_text(encoding="utf-8")
    text = text.replace("batch_size = 0", "batch_size = 2").replace("max_response_length = 100", "max_response_length = 10")
    for split in ("train[:256]", "train[:100]", "test[:100]", "train[:400]"):
        text = text.replace(f'split = "{split}"', f'split = "{split.split("[")[0]}[:5]"')
    text = text.replace("lora_rank = 50", "lora_rank = 8").replace("n_optimization_steps = 5", "n_optimization_steps = 2")
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.skipif(not WS, reason="set REALTOOLS_WS to a workspace provisioned by tests/realtools/provision_cpu.py")
def test_real_heretic_cpu_end_to_end(tmp_path):
    workspace = Path(WS)
    components = json.loads((workspace / "tools" / "components.json").read_text())["components"]
    formats = ["fp8"] if "fp8" in components else []
    config = {
        "schema_version": "1.0",
        "model": {"url": TINY, "revision": TINY_SHA},
        "environment": {"profile": "ubuntu", "torch_backend": "cpu", "min_free_disk_gb": 1, "disk_buffer_gb": 0, "install_missing": False},
        "abliteration": {"experiment": "optimize", "max_trials": 3, "max_runtime_minutes": 60, "heretic_config": str(tiny_template(tmp_path / "tiny.toml"))},
        "evaluation": {"published_baselines": None, "qualification": "fast", "independent_smoke_prompts": 12, "max_new_tokens": 16},
        "selection": {
            "max_optimizer_refusal_rate": None,
            "max_optimizer_kl": None,
            "max_heldout_refusal_rate": None,
            "min_benign_pass_rate": None,
            "require_integrity": False,
            "graft_missing_source_tensors": ["mtp."],
        },
        "quantization": {"enabled": bool(formats), "formats": formats, "fp8": {"device": "cpu", "verify_backend": "none"}},
        "storage": {"root": str(workspace)},
    }
    cfg_path = tmp_path / "real.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    services = Services(
        probe_factory=lambda allow_exec: FakeProbe(),  # simulated Linux/B200 facts; tools themselves are real
        hub_factory=lambda cache, token: HubClient(cache, token),
        gpu_query=lambda: None,
        confirm=lambda message: True,
    )
    out = io.StringIO()
    code = main(["--config", str(cfg_path), "--yes", "--new-run"], services=services, stream=out)
    assert code == 0, out.getvalue()
    run = sorted((workspace / "runs").iterdir(), key=lambda p: p.stat().st_mtime)[-1]
    study = json.loads((run / "metrics" / "heretic_study.json").read_text())
    assert len(study["trials"]) == 3 and study["pareto"]
    candidate = json.loads((run / "metrics" / "candidate.json").read_text())
    assert candidate["summary"]["optimizer_refusal_rate"] is not None and candidate["summary"]["optimizer_kl"] is not None
    qual = json.loads((run / "metrics" / "qualification.json").read_text())
    assert qual["independent"]["reload_ok"] is True
    arts = json.loads((run / "artifacts.json").read_text())["artifacts"]
    master = json.loads(Path(arts["master"]["manifest"]).read_text())
    source = json.loads(Path(arts["source"]["manifest"]).read_text())
    assert master["parent_artifact_id"] == source["artifact_id"] and master["source_revision"] == TINY_SHA
    print(
        json.dumps(
            {
                "precision": master["precision"],
                "tensor_diff": master["extensions"]["tensor_diff_vs_source"]["missing_prefixes"],
                "optimizer": candidate["summary"]["optimizer_refusal_rate"],
                "kl": candidate["summary"]["optimizer_kl"],
                "formats": formats,
                "run": run.name,
            },
            indent=1,
        )
    )
