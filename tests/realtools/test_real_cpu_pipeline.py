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

from llm_surgeon.adapters.hf_hub import HubClient
from llm_surgeon.cli import main
from llm_surgeon.orchestrator import Services

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
        "profiles": [str(ROOT / "configs" / "profiles" / "qwen3_5.yaml")],  # tiny-random/qwen3.8 is a Qwen3.5 architecture (Step 4)
        "model": {"url": TINY, "revision": TINY_SHA},
        "environment": {"profile": "ubuntu", "torch_backend": "cpu", "min_free_disk_gb": 1, "disk_buffer_gb": 0, "install_missing": False},
        "abliteration": {
            "experiment": "optimize",
            "max_trials": 5,
            "seed": 1234,  # fixed, so the trial sequence is reproducible
            "max_runtime_minutes": 60,
            "heretic_config": str(tiny_template(tmp_path / "tiny.toml")),
        },
        "evaluation": {"published_baselines": None, "qualification": "fast", "independent_smoke_prompts": 12, "max_new_tokens": 16},
        "selection": {
            "max_optimizer_refusal_rate": None,
            "max_optimizer_kl": None,
            "max_heldout_refusal_rate": None,
            "min_benign_pass_rate": None,
            "require_integrity": False,
            # Step 1: a null threshold disables a gate only when the gate is optional.
            "optional_gates": ["integrity", "optimizer_kl", "benign_pass_rate", "heldout_refusal_rate", "optimizer_refusal_rate"],
            "graft_missing_source_tensors": ["mtp."],
        },
        "quantization": {
            "enabled": bool(formats),
            "formats": formats,
            "fp8": {"device": "cpu", "verify_backend": "none", "scheme": "fp8_dynamic"},  # FP8_BLOCK: implementation-log finding 18
        },
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
    run = sorted((workspace / "runs").iterdir(), key=lambda p: p.stat().st_mtime)[-1]
    qual = json.loads((run / "metrics" / "qualification.json").read_text())
    if code == 20:
        # Random weights produce word salad: the integrity check (and so stage 09's master smoke test, which reuses
        # stage 06) correctly fails while the reload itself succeeds. Nothing else may be partial.
        verification = json.loads(next((run / "receipts").glob("09-*.json")).read_text())["diagnostics"]
        assert (verification["inference_failures"], verification["integrity_failures"]) == (["master"], []), out.getvalue()
        assert qual["independent"]["reload_ok"] is True and qual["independent"]["integrity_passed"] is False
    else:
        assert code == 0, out.getvalue()
    study = json.loads((run / "metrics" / "heretic_study.json").read_text())
    assert len(study["trials"]) == 5 and study["pareto"]
    # Steps 2-3 with the real engine: search-only, then export-only of the selected trial (zero new trials).
    receipt = json.loads(next((run / "receipts").glob("05-*.json")).read_text())
    hist = history(receipt)
    first = hist[0]["action"]
    assert first in ("new", "replay"), "replay: an earlier run in this workspace had the same inputs (fixed seed)"
    assert [(e["action"], e["phase"]) for e in hist] == [
        (first, "intent"),
        (first, "completion"),
        ("select", "completion"),
        ("export", "intent"),
        ("export", "completion"),
    ]
    chosen = hist[2]["selection"]["chosen"]
    assert hist[1]["trials_added"] == 5 and hist[-1]["trials_added"] == 0 and hist[-1]["export"]["restored_trial"] == chosen["index"]
    assert hist[3]["before"]["trial_set_sha256"] == hist[-1]["after"]["trial_set_sha256"]
    candidate = json.loads((run / "metrics" / "candidate.json").read_text())
    assert candidate["summary"]["optimizer_refusal_rate"] is not None and candidate["summary"]["optimizer_kl"] is not None
    assert qual["independent"]["reload_ok"] is True
    arts = json.loads((run / "artifacts.json").read_text())["artifacts"]
    master = json.loads(Path(arts["master"]["manifest"]).read_text())
    source = json.loads(Path(arts["source"]["manifest"]).read_text())
    assert master["parent_artifact_id"] == source["artifact_id"] and master["source_revision"] == TINY_SHA
    # Step 5 with the real transformers runtime: the master is the comparison baseline, rendered by its own tokenizer.
    assert [r["kind"] for r in master["assessments"]][:1] == ["promotion"] and master["derived_status"]["evaluation"]
    comparison = json.loads((run / "metrics" / "comparison.json").read_text()) if formats else None
    if comparison:
        assert comparison["baseline"]["role"] == "master" and comparison["baseline"]["measured"] is True
        prompts = comparison["contract"]["prompts"]
        assert len(prompts) == 12 and all(p["rendered_chars"] > p["prompt_chars"] for p in prompts), "real chat-template rendering"
        assert comparison["contract"]["chat_template"]["template_sha256"] and comparison["contract"]["context"]["max_prompt_tokens"]
        fp8 = comparison["exports"]["export:fp8"]
        assert fp8["outcome"] == "unmeasured" and "verify_backend=none" in fp8["unmeasured_reason"]
    # A fork with a larger budget continues its copy of the study with the real engine (set-settings for a
    # search-only continue, 3 more trials, then export-only); the parent's study is never written.
    parent_files = {f: f.read_bytes() for f in Path(receipt["diagnostics"]["study_dir"]).rglob("*.json*")}
    fork = fork_run(cfg_path, services, run.name, "--max-trials", "8")
    fork_receipt = json.loads(next((fork / "receipts").glob("05-*.json")).read_text())
    fork_hist = history(fork_receipt)
    assert [(e["action"], e["phase"]) for e in fork_hist] == [
        ("fork-continue", "intent"),
        ("fork-continue", "completion"),
        ("continue", "intent"),
        ("continue", "completion"),
        ("select", "completion"),
        ("export", "intent"),
        ("export", "completion"),
    ]
    assert fork_hist[3]["trials_added"] == 3 and fork_hist[-1]["trials_added"] == 0 and fork_hist[-1]["after"]["trials"] == 8
    assert fork_receipt["diagnostics"]["fingerprints"]["study_identity"] == receipt["diagnostics"]["fingerprints"]["study_identity"]
    assert {f: f.read_bytes() for f in parent_files} == parent_files, "the parent's study is untouched"
    fork_study = json.loads((fork / "metrics" / "heretic_study.json").read_text())
    fork_result = {"run": fork.name, "trials": len(fork_study["trials"]), "pareto": [t["index"] for t in fork_study["pareto"]]}
    exported = fork_hist[-1]["export"]["index"]
    other = next((t["index"] for t in fork_study["pareto"] if t["index"] != exported), None)
    if other is not None:
        # Another Pareto trial by --trial-id: real export-only at a different position, zero new trials.
        second = fork_run(cfg_path, services, fork.name, "--trial-id", str(other))
        second_hist = history(json.loads(next((second / "receipts").glob("05-*.json")).read_text()))
        assert [e["action"] for e in second_hist] == ["fork-continue", "fork-continue", "select", "export", "export"]
        assert second_hist[-1]["export"]["restored_trial"] == other and second_hist[-1]["after"]["trials"] == 8
        fork_result["trial_id_fork"] = {"run": second.name, "trial": other, "pareto_position": second_hist[-1]["export"]["pareto_position"]}
    print(
        json.dumps(
            {
                "selected_trial": chosen,
                "first_action": first,
                "fork": fork_result,
                "precision": master["precision"],
                "tensor_diff": master["extensions"]["tensor_diff_vs_source"]["missing_prefixes"],
                "comparison_baseline": comparison and {k: comparison["baseline"].get(k) for k in ("benign_correct", "heldout_refused", "integrity_issues")},
                "optimizer": candidate["summary"]["optimizer_refusal_rate"],
                "kl": candidate["summary"]["optimizer_kl"],
                "formats": formats,
                "run": run.name,
            },
            indent=1,
        )
    )


def history(receipt: dict) -> list[dict]:
    return [json.loads(line) for line in Path(receipt["diagnostics"]["history"]).read_text().splitlines()]


A100_JOURNAL = os.environ.get("A100_JOURNAL")


@pytest.mark.skipif(not (WS and A100_JOURNAL), reason="set REALTOOLS_WS and A100_JOURNAL (the A100 run's checkpoints/model.jsonl)")
def test_real_study_tool_on_the_a100_journal_selects_trial_3(tmp_path):
    """Step 2 acceptance with the real Optuna journal of the A100 run (read from a copy; the evidence is never modified)."""
    import shutil
    import subprocess

    from llm_surgeon.evaluation.selection import select_trial

    components = json.loads((Path(WS) / "tools" / "components.json").read_text())["components"]
    heretic = components["heretic"]
    journal = tmp_path / "model.jsonl"
    shutil.copy2(A100_JOURNAL, journal)
    out = tmp_path / "export.json"
    tool = Path(heretic["paths"]["scripts_dir"]) / "heretic_study_tool.py"
    subprocess.run([*heretic["commands"]["python"], str(tool), "export", "--journal", str(journal), "--out", str(out)], check=True)
    export = json.loads(out.read_text())
    assert [t["index"] for t in export["pareto"]] == [8, 9, 3, 5]
    doc = select_trial(export, study_id="a100", policy="min_refusals_within_kl", max_refusal=0.2, max_kl=0.1, refusal_name="Refusals", kl_name="KL divergence")
    assert (doc["chosen"]["index"], doc["chosen"]["number"], doc["chosen"]["pareto_position"]) == (3, 2, 2)
    assert doc["chosen"]["refusal_rate"] == 0.03 and doc["chosen"]["kl"] == 0.0916491448879242
    assert journal.read_bytes() == Path(A100_JOURNAL).read_bytes(), "reading adds nothing to the journal"


def fork_run(cfg_path: Path, services, parent: str, *extra: str) -> Path:
    out = io.StringIO()
    code = main(["--config", str(cfg_path), "--yes", "--resume", parent, "--fork", "--through", "5", *extra], services=services, stream=out)
    assert code == 0, out.getvalue()
    return sorted((Path(WS) / "runs").iterdir(), key=lambda p: p.stat().st_mtime)[-1]
