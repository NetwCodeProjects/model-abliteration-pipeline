"""Full 00-10 pipeline runs with fake Heretic / llama.cpp / FP8 / vLLM tools (CPU only)."""

from __future__ import annotations

import json
from pathlib import Path

from model_pipeline.fsutil import quick_fingerprint

STAGES = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]


def test_full_pipeline_produces_manifests_with_lineage(env):
    env.provision()
    code, out = env.run()
    assert code == 0, out
    run = env.latest_run()
    for stage in STAGES:
        assert env.receipt(run, stage)["outcome"] == "succeeded", stage
    arts = env.artifacts(run)
    assert set(arts) == {"source", "master", "export:fp8", "export:gguf:iq4_xs", "export:gguf:iq3_s"}
    manifests = {role: json.loads(Path(a["manifest"]).read_text()) for role, a in arts.items()}
    source, master = manifests["source"], manifests["master"]
    assert source["transformation"]["kind"] == "original" and source["source_revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert master["parent_artifact_id"] == source["artifact_id"]
    assert master["precision"] == "bf16"
    for role in ("export:fp8", "export:gguf:iq4_xs", "export:gguf:iq3_s"):
        assert manifests[role]["parent_artifact_id"] == master["artifact_id"], role
        assert manifests[role]["source_revision"] == source["source_revision"]
        assert manifests[role]["validation"]["integrity_passed"] is True
        assert manifests[role]["validation"]["inference_passed"] is True
        assert all(len(f["sha256"]) == 64 for f in manifests[role]["artifact"]["files"])
    assert manifests["export:fp8"]["precision"] == "fp8"
    assert manifests["export:gguf:iq4_xs"]["validation"]["modalities_tested"] == ["text", "image"]
    # Heretic's merged export dropped mtp.* tensors: recorded, not hidden.
    assert master["extensions"]["tensor_diff_vs_source"]["missing_prefixes"] == {"mtp": 2}
    assert (run / "reports" / "report.md").exists()
    summary = json.loads((run / "reports" / "run-summary.json").read_text())
    assert summary["complete"] is True


def test_published_baselines_do_no_inference_or_reference_download(env):
    env.provision()
    code, out = env.run()
    assert code == 0, out
    run = env.latest_run()
    receipt = env.receipt(run, "03")
    assert receipt["diagnostics"]["local_inference_runs"] == 0
    assert ("snapshot", "coder3101/Qwen3.8-27B-heretic-ara") not in env.hub.calls
    # Only our own artifacts ran: the candidate (stage 06) and the master as comparison baseline (stage 09).
    assert env.count("qualify") == 2
    specs = sorted((run / "logs").glob("*/*-spec.json"))
    assert {json.loads(s.read_text())["model_path"] for s in specs} <= {str(p) for p in (run.parents[1] / "models" / "candidates").glob("*/*")} | {
        env.artifacts(run)["master"]["path"]
    }


def test_qualification_separates_optimizer_metrics_from_independent_ones(env):
    env.provision()
    assert env.run()[0] == 0
    qual = json.loads((env.latest_run() / "metrics" / "qualification.json").read_text())
    assert "heretic_optimizer" in qual["imported_optimizer_metrics"]["provenance"]
    assert qual["independent"]["provenance"].startswith("independent_qualification")
    assert qual["independent"]["heldout_refusal_prompts"] == 20  # 30 total - 10 benign
    assert qual["independent"]["benign_pass_rate"] == 1.0


def test_source_and_master_are_never_modified(env):
    env.provision()
    assert env.run()[0] == 0
    run = env.latest_run()
    arts = env.artifacts(run)
    for role in ("source", "master"):
        manifest = json.loads(Path(arts[role]["manifest"]).read_text())
        before = {f["relative_path"]: f["size_bytes"] for f in manifest["artifact"]["files"]}
        now = {k: v[0] for k, v in quick_fingerprint(manifest["artifact"]["uri"]).items()}
        assert before == now


def test_resume_reuses_every_valid_receipt(env):
    env.provision()
    assert env.run()[0] == 0
    run_id = env.latest_run().name
    heretic_calls, fp8_calls = env.count("heretic"), env.count("fp8")
    code, out = env.run("--resume", run_id)
    assert code == 0, out
    assert env.count("heretic") == heretic_calls and env.count("fp8") == fp8_calls
    status = json.loads((env.workspace / "runs" / run_id / "status.json").read_text())
    assert status["stages"]["05"]["status"] == "reused"
    assert status["stages"]["00"]["status"] == "succeeded"  # host checks always re-run
