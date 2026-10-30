"""Mocked integration scenarios: quantize-only, partial failures, crash/resume, forks, gates, reproduction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_pipeline.errors import ExitCode


def manifests(env, run_dir):
    return {role: json.loads(Path(a["manifest"]).read_text()) for role, a in env.artifacts(run_dir).items()}


def test_quantize_only_skips_heretic(env):
    env.provision()
    code, out = env.run("--quantize-only")
    assert code == 0, out
    run = env.latest_run()
    assert env.count("heretic") == 0 and env.count("qualify") == 0
    for stage in ("03", "04", "05", "06", "07"):
        receipt = env.receipt(run, stage)
        assert receipt["outcome"] == "skipped" and receipt["diagnostics"]["not_applicable"], stage
    m = manifests(env, run)
    assert "master" not in m
    for role in ("export:fp8", "export:gguf:iq4_xs", "export:gguf:iq3_s"):
        assert m[role]["parent_artifact_id"] == m["source"]["artifact_id"]


def test_quantize_only_from_local_path_records_supplied_provenance(env):
    env.provision()
    code, out = env.run("--quantize-only", "--model-path", str(env.model), "--quantizations", "iq4_xs")
    assert code == 0, out
    m = manifests(env, env.latest_run())
    assert m["source"]["source_model"] is None and m["source"]["source_revision"] is None
    assert m["source"]["extensions"]["provenance"] == "supplied"
    assert m["export:gguf:iq4_xs"]["parent_artifact_id"] == m["source"]["artifact_id"]
    assert not list(env.model.parent.glob("*.manifest.json")), "no files written next to user-supplied models"


def test_failed_format_keeps_other_exports(env, monkeypatch):
    env.provision()
    monkeypatch.setenv("FAKE_QUANT_FAIL", "IQ3_S")
    code, out = env.run()
    assert code == ExitCode.PARTIAL, out
    run = env.latest_run()
    m = manifests(env, run)
    assert {"export:fp8", "export:gguf:iq4_xs"} <= set(m) and "export:gguf:iq3_s" not in m
    sub = json.loads((run / "receipts" / "08-quantize" / "iq3_s.json").read_text())
    assert sub["outcome"] == "failed" and sub["error"]["error_id"] == "E_QUANT_TOOL_FAILED"
    assert not list((env.workspace / "exports" / "gguf").glob("*iq3_s*")), "no partial or fake-complete iq3_s package"
    assert env.receipt(run, "08")["partial"] is True


def test_unsupported_format_is_rejected_before_any_work(env):
    env.provision()
    code, _ = env.run("--quantizations", "fp8,q9_magic")
    assert code == ExitCode.UNSUPPORTED
    assert not (env.workspace / "runs").exists()


def test_resume_after_stage05_crash_continues_study(env, monkeypatch):
    env.provision()
    monkeypatch.setenv("FAKE_HERETIC_MODE", "crash-mid")
    code, out = env.run()
    assert code == ExitCode.ENGINE, out
    run = env.latest_run()
    assert env.receipt(run, "05")["error"]["error_id"] == "E_HERETIC_NO_OUTPUT"
    assert (run / "reports" / "report.md").read_text().startswith("# Run report")
    monkeypatch.delenv("FAKE_HERETIC_MODE")
    code, out = env.run("--resume", run.name)
    assert code == 0, out
    trials = (env.counters / "heretic.log").read_text().split()
    started = [int(t) for t in trials if t.isdigit()]
    assert started[:3] == [1, 2, 3] and started.count(1) == 1, "study continued instead of restarting"
    receipt = env.receipt(run, "05")
    assert receipt["outcome"] == "succeeded" and receipt["diagnostics"]["candidate_path"]


def test_resume_after_stage08_crash_reuses_finished_formats(env, monkeypatch):
    env.config_path = env.write_config({"execution": {"allow_partial_exports": False}})
    env.base_args[1] = str(env.config_path)
    env.provision()
    monkeypatch.setenv("FAKE_QUANT_FAIL", "IQ3_S")
    code, out = env.run()
    assert code == ExitCode.QUANTIZE, out
    run = env.latest_run()
    fp8_runs, quant_runs = env.count("fp8"), env.count("llama-quantize")
    monkeypatch.delenv("FAKE_QUANT_FAIL")
    code, out = env.run("--resume", run.name)
    assert code == 0, out
    assert env.count("fp8") == fp8_runs, "finished fp8 export must be reused"
    assert env.count("llama-quantize") == quant_runs + 1, "only iq3_s is re-run"


def test_resume_rejects_config_changes_and_fork_reuses_valid_work(env):
    env.provision()
    assert env.run()[0] == 0
    run = env.latest_run()
    code, _ = env.run("--resume", run.name, "--model-url", "other-org/other-model")
    assert code == ExitCode.DEPENDENCY
    heretic_calls = env.count("heretic")
    code, out = env.run("--resume", run.name, "--fork", "--quantizations", "fp8")
    assert code == 0, out
    forked = [p for p in (env.workspace / "runs").iterdir() if p.name != run.name]
    assert len(forked) == 1
    status = json.loads((forked[0] / "status.json").read_text())
    assert status["parent_run_id"] == run.name
    assert env.count("heretic") == heretic_calls, "abliteration receipts reused by the fork"
    assert status["stages"]["05"]["status"] == "reused"


def test_failed_gates_do_not_promote(env, monkeypatch):
    env.provision()
    monkeypatch.setenv("FAKE_QUALIFY_REFUSE_ALL", "1")
    code, out = env.run()
    assert code == ExitCode.GATE, out
    run = env.latest_run()
    assert "master" not in env.artifacts(run)
    selection = json.loads((run / "metrics" / "selection.json").read_text())
    assert selection["promoted"] is False and "heldout_refusal_rate" in selection["failed"]
    assert "Incomplete run" in (run / "reports" / "report.md").read_text()


def test_optimizer_kl_gate_blocks_aggressive_trial(env, monkeypatch):
    env.provision()
    monkeypatch.setenv("FAKE_KL_STEP", "0.01")
    code, _ = env.run()
    assert code == ExitCode.GATE
    selection = json.loads((env.latest_run() / "metrics" / "selection.json").read_text())
    assert selection["failed"] == ["optimizer_kl"]


def test_exact_reproduction_is_blocked_without_reproduce_json(env):
    env.provision()
    code, out = env.run("--experiment", "reproduce")
    assert code == ExitCode.UNSUPPORTED, out
    run = env.latest_run()
    assert env.receipt(run, "04")["error"]["error_id"] == "E_EXACT_REPRODUCTION_UNAVAILABLE"
    assert env.count("heretic") == 0


def use_config(env, changes):
    env.config_path = env.write_config(changes)
    env.base_args[1] = str(env.config_path)


def test_reproduction_is_not_qualified_without_optimizer_evidence(env):
    env.provision()
    code, out = env.run("--experiment", "reproduce", "--allow-approximate-reproduction")
    assert code == ExitCode.GATE, out
    run = env.latest_run()
    assert env.receipt(run, "07")["error"]["error_id"] == "E_GATES_UNMEASURED"
    selection = json.loads((run / "metrics" / "selection.json").read_text())
    assert selection["outcome"] == "not_promoted" and selection["triggered_gates"] == ["optimizer_kl", "optimizer_refusal_rate"]
    assert "master" not in env.artifacts(run)


def test_approximate_reproduction_is_labelled(env):
    use_config(env, {"selection": {"profile": "research"}})
    env.provision()
    code, out = env.run("--experiment", "reproduce", "--allow-approximate-reproduction")
    assert code == 0, out
    master = manifests(env, env.latest_run())["master"]
    assert master["extensions"]["promotion"]["outcome"] == "promoted:research"
    assert master["extensions"]["promotion"]["run_mode"] == "approximate_reproduction"
    assert master["extensions"]["label"] == "approximate_reproduction"
    assert "approximate_reproduction" in master["transformation"]["method"]
    assert any("not published" in item or "unknown" in item for item in master["extensions"]["limitations"])


def test_exact_reproduction_attempt_patches_only_control_flow(env):
    reproduce = {
        "version": "4",
        "environment": {"heretic": {"version": "2.0.0", "metadata": {}}, "pytorch_version": "2.11", "requirements": {}},
        "settings": {"model": "Qwen/Qwen3.8-27B", "model_action": "upload", "upload_repo_id": "x/y", "seed": 7},
        "parameters": {"start_layer_index": 1},
        "scores": [],
        "hashes": {},
    }
    (env.reference / "reproduce").mkdir()
    (env.reference / "reproduce" / "reproduce.json").write_text(json.dumps(reproduce))
    use_config(env, {"selection": {"optional_gates": ["optimizer_kl", "optimizer_refusal_rate"]}})
    env.provision()
    code, out = env.run("--experiment", "reproduce")
    assert code == 0, out
    run = env.latest_run()
    plan = json.loads((run / "checkpoints" / "heretic" / "engine-plan.json").read_text())
    assert plan["label"] == "exact_reproduction_attempt"
    patched = json.loads(Path(plan["reproduce_json"]).read_text())
    assert patched["settings"]["model_action"] == "save" and patched["settings"]["upload_repo_id"] is None
    assert patched["settings"]["seed"] == 7 and patched["parameters"] == {"start_layer_index": 1}
    promotion = manifests(env, run)["master"]["extensions"]["promotion"]
    assert promotion["outcome"] == "promoted:qualified", "independent evidence only, recorded via optional_gates"


def test_qualification_off_blocks_qualified_promotion_and_research_labels_it(env):
    env.provision()
    code, out = env.run("--qualification", "off")
    assert code == ExitCode.GATE, out
    run = env.latest_run()
    assert env.receipt(run, "07")["error"]["error_id"] == "E_GATES_UNMEASURED"
    use_config(env, {"selection": {"profile": "research"}})
    code, out = env.run("--qualification", "off", "--new-run")
    assert code == 0, out
    run = env.latest_run()
    selection = json.loads((run / "metrics" / "selection.json").read_text())
    assert selection["outcome"] == "promoted:research" and selection["deciding_gate"] == "reload"
    report = (run / "reports" / "report.md").read_text()
    assert "**master**: promoted:research" in report and "Project B" in report
    assert "export of" in report and "not independently qualified" in report


def test_per_gate_override_promotes_ungated(env, monkeypatch):
    use_config(env, {"selection": {"override_gates": ["optimizer_kl"]}})
    env.provision()
    monkeypatch.setenv("FAKE_KL_STEP", "0.01")
    code, out = env.run("--quantizations", "fp8")
    assert code == 0, out
    master = manifests(env, env.latest_run())["master"]
    assert master["extensions"]["ungated"] is True
    kl = next(g for g in master["extensions"]["gates"] if g["gate"] == "optimizer_kl")
    assert kl["status"] == "bypassed_by_override" and kl["override_key"] == "selection.override_gates"


def test_setup_only_runs_stage_00_and_01(env):
    env.provision()
    code, out = env.run("--setup-only")
    assert code == 0, out
    run = env.latest_run()
    assert sorted(p.name[:2] for p in (run / "receipts").glob("*.json")) == ["00", "01"]
    assert env.hub.calls == []


def test_through_then_resume(env):
    env.provision()
    code, _ = env.run("--through", "2")
    assert code == 0
    run = env.latest_run()
    assert sorted(p.name[:2] for p in (run / "receipts").glob("*.json")) == ["00", "01", "02"]
    snapshot_calls = env.hub.calls.count(("snapshot", "Qwen/Qwen3.8-27B"))
    code, out = env.run("--resume", run.name)
    assert code == 0, out
    assert env.hub.calls.count(("snapshot", "Qwen/Qwen3.8-27B")) == snapshot_calls, "verified source is not downloaded again"
    stored = json.loads((run / "configs" / "resolved-config.json").read_text())
    assert any(p["path"].endswith("qwen3_5.yaml") and len(p["sha256"]) == 64 for p in stored["profiles"])
    assert "- Profile: `" in (run / "reports" / "report.md").read_text(), "profile hashes survive resume into the report"


def test_from_stage_without_dependency_receipts_fails_fast(env):
    env.provision()
    code, out = env.run("--from", "8", "--new-run")
    assert code == ExitCode.DEPENDENCY, out
    assert env.count("heretic") == 0 and env.count("fp8") == 0


def test_graft_restores_missing_source_tensors(env):
    use_config(env, {"selection": {"graft_missing_source_tensors": ["mtp."]}})
    env.provision()
    code, out = env.run("--quantizations", "fp8")
    assert code == 0, out
    m = manifests(env, env.latest_run())
    assert m["master"]["extensions"]["tensor_diff_vs_source"]["missing_count"] == 0
    assert len(m["master"]["extensions"]["grafted_tensors"]["grafted"]) == 2
    candidate = Path(json.loads((env.latest_run() / "metrics" / "candidate.json").read_text())["path"])
    index = json.loads((candidate / "model.safetensors.index.json").read_text())
    assert not any(k.startswith("mtp.") for k in index["weight_map"]), "candidate untouched by grafting"


def test_concurrent_run_is_locked(env):
    env.provision()
    assert env.run("--through", "0")[0] == 0
    run = env.latest_run()
    lock = run / ".lock"
    lock.write_text(json.dumps({"host": __import__("socket").gethostname(), "pid": __import__("os").getpid(), "since": 0, "purpose": "test"}))
    code, _ = env.run("--resume", run.name)
    assert code == ExitCode.LOCKED


@pytest.mark.parametrize("mode,error", [("config-error", "E_HERETIC_CONFIG"), ("no-output", "E_HERETIC_NO_OUTPUT")])
def test_heretic_exit_zero_is_not_trusted(env, monkeypatch, mode, error):
    env.provision()
    monkeypatch.setenv("FAKE_HERETIC_MODE", mode)
    code, _ = env.run()
    assert code == ExitCode.ENGINE
    assert env.receipt(env.latest_run(), "05")["error"]["error_id"] == error


def test_fork_with_other_pareto_index_continues_study_without_new_trials(env, monkeypatch):
    env.provision()
    monkeypatch.setenv("FAKE_KL_STEP", "0.01")
    code, _ = env.run()
    assert code == ExitCode.GATE
    parent = env.latest_run()
    trials_before = (env.counters / "heretic.log").read_text().count("trial")
    code, out = env.run("--resume", parent.name, "--fork", "--pareto-index", "45")
    assert code == 0, out
    fork = env.latest_run()
    assert fork.name != parent.name
    assert (env.counters / "heretic.log").read_text().count("trial") == trials_before, "no new Heretic trials"
    receipt = env.receipt(fork, "05")
    assert receipt["diagnostics"]["study_copied_from_run"] == parent.name
    candidate = json.loads((fork / "metrics" / "candidate.json").read_text())
    assert candidate["summary"]["exported_trial_index"] == 5 and candidate["summary"]["optimizer_kl"] <= 0.1
    assert env.receipt(parent, "04")["diagnostics"]["seed"] == env.receipt(fork, "04")["diagnostics"]["seed"]


@pytest.mark.parametrize("fails", [False, True])
def test_fp8_loader_test_on_a100_is_not_representative(env, monkeypatch, fails):
    env.probe.gpu_name, env.probe.compute_capability = "NVIDIA A100-SXM4-40GB", "8.0"
    if fails:
        monkeypatch.setenv("FAKE_VLLM_FAIL", "1")
    env.provision()
    code, out = env.run("--quantizations", "fp8")
    assert code == 0, out  # a fallback failure neither fails nor passes the export
    run = env.latest_run()
    fp8 = manifests(env, run)["export:fp8"]
    assert fp8["validation"]["inference_passed"] is None
    smoke = fp8["extensions"]["verification"]["smoke"]
    assert smoke["ran"] and smoke["representative"] is False and smoke["passed"] is (not fails)
    preflight = json.loads((run / "metrics" / "preflight.json").read_text())
    assert [f["rule"] for f in preflight["capabilities"]] == ["fp8_vllm_weight_only"]
