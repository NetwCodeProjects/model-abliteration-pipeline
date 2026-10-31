"""Step 5 acceptance with the fake runtimes: assessment records, master vs export comparison, redaction."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from conftest import ROOT

from model_pipeline.errors import ExitCode


def manifests(env, run):
    return {role: json.loads(Path(a["manifest"]).read_text()) for role, a in env.artifacts(run).items()}


def records(env, artifact_id):
    directory = env.workspace / "assessments" / artifact_id
    return sorted((json.loads(p.read_text()) for p in directory.glob("asm-*.json")), key=lambda r: r["created_utc"])


def comparison(run):
    return json.loads((run / "metrics" / "comparison.json").read_text())


def test_per_export_deltas_with_denominators(env, monkeypatch):
    monkeypatch.setenv("FAKE_DEGRADE", "iq3_s")
    env.provision()
    code, out = env.run()
    assert code == 0, out  # export gates report only by default
    run = env.latest_run()
    doc = comparison(run)
    assert doc["baseline"]["role"] == "master" and doc["baseline"]["benign_correct"] == {"count": 8, "of": 8}
    assert doc["baseline"]["heldout_refused"] == {"count": 1, "of": 4}
    good, bad = doc["exports"]["export:gguf:iq4_xs"], doc["exports"]["export:gguf:iq3_s"]
    assert good["outcome"] == "passed" and good["deltas"]["benign_correct"] == {"baseline": "8/8", "export": "8/8", "delta": 0, "rate_delta": 0.0}
    assert bad["outcome"] == "failed"
    assert bad["deltas"]["benign_correct"] == {"baseline": "8/8", "export": "0/8", "delta": -8, "rate_delta": -1.0}
    assert bad["deltas"]["heldout_refused"] == {"baseline": "1/4", "export": "4/4", "delta": 3, "rate_delta": 0.75}
    assert "exported runtime (llama.cpp, iq3_s) vs baseline runtime (transformers, master)" == bad["description"]
    assert [g["field"] for g in bad["contract_gaps"]] == ["tokenization", "stop"]
    assert doc["exports"]["export:fp8"]["outcome"] == "passed" and doc["exports"]["export:fp8"]["contract_gaps"] == []
    m = manifests(env, run)
    master = m["master"]
    kinds = [r["kind"] for r in master["assessments"]]
    assert kinds == ["promotion", "comparison", "verification"], "promotion (07), baseline comparison and verification (09)"
    promotion = next(r for r in master["assessments"] if r["kind"] == "promotion")
    assert master["derived_status"]["evaluation"] == promotion["assessment_id"] and master["evaluation"]["metrics_provenance"] == "fast-qualification"
    record = next(r for r in records(env, m["export:gguf:iq3_s"]["artifact_id"]) if r["kind"] == "comparison")
    assert record["results"]["baseline"]["assessment_id"] == doc["baseline"]["assessment_id"]
    assert record["generation_contract"]["contract_id"] == doc["contract"]["contract_id"] and record["redaction"]["exact_text_replay"] == "available"
    report = (run / "reports" / "report.md").read_text(encoding="utf-8")
    assert "Master vs export comparison" in report and "| export:gguf:iq3_s | llama.cpp | 0/8 (-8) |" in report


def test_enforced_export_gates_fail_the_degraded_export(env, monkeypatch):
    monkeypatch.setenv("FAKE_DEGRADE", "iq3_s")
    env.provision()
    code, out = env.run("--set", "selection.export_gates.mode=enforce")
    assert code == ExitCode.PARTIAL, out
    receipt = env.receipt(env.latest_run(), "09")
    assert receipt["diagnostics"]["comparison_failures"] == ["export:gguf:iq3_s"] and receipt["partial"]


def test_quantize_only_compares_with_the_source(env):
    env.provision()
    code, out = env.run("--quantize-only")
    assert code == 0, out
    run = env.latest_run()
    doc = comparison(run)
    source = manifests(env, run)["source"]
    assert doc["baseline"]["role"] == "source" and doc["baseline"]["artifact_id"] == source["artifact_id"]
    assert doc["baseline"]["benign_correct"] == {"count": 8, "of": 8}
    assert all(e["outcome"] == "passed" and e["deltas"]["benign_correct"]["delta"] == 0 for e in doc["exports"].values())
    assert [r["kind"] for r in source["assessments"]] == ["comparison"] and records(env, source["artifact_id"])[0]["results"]["baseline_role"] == "source"


def test_missing_baseline_makes_every_comparison_unmeasured(env, monkeypatch):
    monkeypatch.setenv("FAKE_BASELINE_FAIL", "1")
    env.provision()
    code, out = env.run()
    assert code == 0, out
    doc = comparison(env.latest_run())
    assert doc["baseline"]["measured"] is False and doc["baseline"]["reason"] == "did not load"
    for export in doc["exports"].values():
        assert export["outcome"] == "unmeasured" and export["deltas"] is None and export["benign_correct"] is None
        assert {c["result"] for c in export["checks"][1:]} == {"unmeasured"} and "no baseline" in export["unmeasured_reason"]


def test_two_runs_assessing_the_same_bytes_keep_both_records(env):
    env.provision()
    assert env.run()[0] == 0
    first = env.latest_run()
    fp8 = manifests(env, first)["export:fp8"]
    before = {p: p.read_bytes() for p in (env.workspace / "assessments" / fp8["artifact_id"]).glob("asm-*.json")}
    assert len(before) == 2  # verification + comparison
    # A fork that changes only an export-gate threshold: stages 00-08 are reused, stage 09 assesses the same bytes again.
    code, out = env.run("--resume", first.name, "--fork", "--set", "selection.export_gates.max_benign_correct_drop=0.3")
    assert code == 0, out
    (second,) = [r for r in (env.workspace / "runs").iterdir() if r.name != first.name]
    assert env.receipt(second, "07")["copied_from_run"] == first.name, "the master was not re-promoted"
    after = records(env, fp8["artifact_id"])
    assert len(after) == 4 and {r["run_id"] for r in after} == {first.name, second.name}
    assert {p: p.read_bytes() for p in before} == before, "the first run's records are untouched"
    current = json.loads((env.workspace / "manifests" / f"{fp8['artifact_id']}.json").read_text())
    newest = [r for r in after if r["kind"] == "verification" and r["run_id"] == second.name][0]
    assert current["derived_status"]["validation"] == newest["assessment_id"] and len(current["assessments"]) == 4


def test_redaction_leaves_no_prompt_or_response_text_in_the_workspace(env):
    env.provision()
    code, out = env.run("--set", "logging.redact_prompts=true")
    assert code == 0, out
    run = env.latest_run()
    benign = [p["prompt"] for p in yaml.safe_load((ROOT / "configs" / "qualification" / "benign_prompts.yaml").read_text(encoding="utf-8"))["prompts"]]
    smoke = yaml.safe_load((ROOT / "configs" / "pipeline.yaml").read_text(encoding="utf-8")).get("quantization", {}).get("smoke_prompts") or [
        "What is the capital of France? Answer in one word."
    ]
    responses = ["The answer is", "I'm sorry, but I can't", "high-level overview", "Paris.", "def square", "17 * 23 = 391", "I do not know"]
    texts = benign + [f"held-out prompt {i}" for i in range(4)] + ["<|im_start|>"] + responses  # fake prompt and rendering text
    leaks = []
    for path in env.workspace.rglob("*"):
        if not path.is_file():
            continue
        content = path.read_bytes().decode("utf-8", errors="ignore")
        # The operator's own smoke prompts are configuration (resolved-config) by design (docs/assessments.md#redaction).
        allowed = smoke if path.parent.name == "configs" and path.name.startswith("resolved-config") else []
        leaks += [(str(path.relative_to(env.workspace)), t) for t in texts + smoke if t in content and t not in allowed]
    assert leaks == [], "\n".join(map(str, leaks))
    record = next(r for r in records(env, manifests(env, run)["export:fp8"]["artifact_id"]) if r["kind"] == "comparison")
    assert record["redaction"] == {"redacted": True, "exact_text_replay": "unavailable (redacted)"}
    assert all("response" not in p and p["response_sha256"] for p in record["results"]["per_prompt"])
    assert comparison(run)["exports"]["export:fp8"]["outcome"] == "passed", "comparisons work on hashes and scores"
    assert env.receipt(run, "09")["diagnostics"]["generation_files_discarded"]
