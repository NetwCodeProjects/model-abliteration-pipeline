"""Step 5: append-only assessment records, derived manifest status, comparison scoring and contract."""

from __future__ import annotations

import json

import pytest
from conftest import ROOT

from model_pipeline.artifacts import assessments as A
from model_pipeline.artifacts.manifest import ArtifactRegistry, build_manifest
from model_pipeline.config import build_config
from model_pipeline.errors import PipelineError
from model_pipeline.evaluation import comparison as C

FILES = [{"relative_path": "model.safetensors", "size_bytes": 10, "sha256": "c" * 64}]
LOADER = {"inference_passed": True, "modalities_tested": ["text"], "backend": "llama.cpp", "backend_revision": "llama.cpp abc", "tested_context": 4096}


def manifest() -> dict:
    return build_manifest(
        run_id="r1",
        model_family="tiny",
        source_model="org/tiny",
        source_revision="a" * 40,
        parent_artifact_id=None,
        kind="quantization",
        method="test",
        config_hash="b" * 64,
        tool_revision=None,
        precision="iq4_xs",
        fmt="gguf",
        uri="/x",
        files=FILES,
    )


def record(artifact_id: str, run_id: str = "r1", kind: str = "verification", created: str | None = None, **derives) -> dict:
    rec = A.new_record(
        artifact_id=artifact_id,
        role="export:gguf:iq4_xs",
        run_id=run_id,
        stage_id="09",
        kind=kind,
        results={"n": 1},
        checks=[A.check("integrity", "passed")],
        derives=derives,
    )
    if created:  # deterministic order for derivation tests; the id is recomputed like new_record does
        rec = {k: v for k, v in rec.items() if k != "assessment_id"}
        rec["created_utc"] = created
        rec["assessment_id"] = "asm-" + A.sha256_obj(rec)[:24]
    return rec


def test_outcome_weakest_claim_wins():
    assert A.outcome_of([A.check("a", "passed"), A.check("b", "unmeasured"), A.check("c", "failed")]) == "failed"
    assert A.outcome_of([A.check("a", "passed"), A.check("b", "unmeasured")]) == "unmeasured"
    assert A.outcome_of([A.check("a", "passed"), A.check("b", "disabled")]) == "passed"
    assert A.outcome_of([A.check("a", "disabled")]) == "unmeasured", "nothing measured is not a pass"


def test_records_are_append_only_and_two_runs_give_two_records(tmp_path):
    art = manifest()["artifact_id"]
    first, second = record(art, "run-1"), record(art, "run-2")
    assert first["assessment_id"] != second["assessment_id"], "same bytes, same results, different run: two records"
    path = A.write_record(tmp_path / "ws", tmp_path / "run-1", first)
    before = path.read_bytes()
    with pytest.raises(PipelineError) as err:
        A.write_record(tmp_path / "ws", tmp_path / "run-1", first)
    assert err.value.error_id == "E_ASSESSMENT_EXISTS" and path.read_bytes() == before
    A.write_record(tmp_path / "ws", tmp_path / "run-2", second)
    assert {r["assessment_id"] for r in A.records_for(tmp_path / "ws", art)} == {first["assessment_id"], second["assessment_id"]}
    assert (tmp_path / "run-2" / "assessments" / f"{second['assessment_id']}.json").exists(), "run copy"
    with pytest.raises(PipelineError) as err:
        A.write_record(tmp_path / "ws", tmp_path / "run-1", {**first, "kind": "rumour"})
    assert err.value.error_id == "E_ASSESSMENT_INVALID"


def test_derive_latest_record_per_part_and_idempotent():
    m = manifest()
    art = m["artifact_id"]
    tested = record(art, "run-1", created="2026-10-01T00:00:00Z", validation={"integrity_passed": True}, inference=LOADER)
    untested = record(art, "run-2", created="2026-10-02T00:00:00Z", validation={"integrity_passed": False})
    derived = A.derive(m, [untested, tested][::-1])
    assert derived["validation"] == {"integrity_passed": False, "inference_passed": True, "modalities_tested": ["text"]}
    assert derived["derived_status"] == {"validation": untested["assessment_id"], "inference": tested["assessment_id"], "evaluation": None}
    assert derived["inference"]["backend"] == "llama.cpp", "a later run without a loader test keeps the earlier loader result and names it"
    assert [r["run_id"] for r in derived["assessments"]] == ["run-1", "run-2"]
    assert A.derive(derived, [tested, untested]) == derived


def test_schema_1_0_values_are_kept_until_a_record_measures_them():
    legacy = {**manifest(), "schema_version": "1.0"}
    legacy["validation"]["inference_passed"] = True
    legacy["evaluation"]["metrics_provenance"] = "integrity-only"
    rec = record(legacy["artifact_id"], validation={"integrity_passed": True})
    derived = A.derive(legacy, [rec])
    assert derived["validation"]["inference_passed"] is True and derived["evaluation"]["metrics_provenance"] == "integrity-only"
    assert derived["derived_status"] == {"validation": rec["assessment_id"], "inference": None, "evaluation": None}


def test_registry_never_loses_records_of_another_run(tmp_path):
    ws = tmp_path / "ws"
    m = manifest()
    target = ws / "exports" / "gguf" / "x"
    first = ArtifactRegistry(tmp_path / "run-1", ws)
    first.register(m, "export:gguf:iq4_xs", target, "08")
    r1 = record(m["artifact_id"], "run-1", validation={"integrity_passed": True}, inference=LOADER)
    first.attach("export:gguf:iq4_xs", r1)
    second = ArtifactRegistry(tmp_path / "run-2", ws)
    second.register(json.loads(target.with_name("x.manifest.json").read_text()), "export:gguf:iq4_xs", target, "08")  # stage 08 reuse
    r2 = record(m["artifact_id"], "run-2", validation={"integrity_passed": True})
    final = second.attach("export:gguf:iq4_xs", r2)
    assert [r["assessment_id"] for r in final["assessments"]] == [r1["assessment_id"], r2["assessment_id"]]
    assert final["validation"]["inference_passed"] is True and final["derived_status"]["inference"] == r1["assessment_id"]
    assert json.loads((ws / "manifests" / f"{m['artifact_id']}.json").read_text()) == final


def test_assessment_and_manifest_schemas_agree_with_jsonschema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    schemas = {n: json.loads((ROOT / "schemas" / n).read_text()) for n in ("assessment.schema.json", "artifact.schema.json")}
    for schema in schemas.values():
        jsonschema.Draft202012Validator.check_schema(schema)
    m = manifest()
    rec = record(m["artifact_id"], validation={"integrity_passed": True}, inference=LOADER)
    jsonschema.validate(rec, schemas["assessment.schema.json"])
    jsonschema.validate(A.derive(m, [rec]), schemas["artifact.schema.json"])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**rec, "outcome": ""}, schemas["assessment.schema.json"])


# ---------------------------------------------------------------------------- comparison
SPEC = {
    "benign": [
        {"id": "b1", "prompt": "What is 2+2?", "expect_any": ["4"], "category": "fast"},
        {"id": "b2", "prompt": "Capital of France?", "expect_any": ["paris"], "category": "fast"},
    ],
    "heldout_refusal_specs": [],
    "heldout_limit": 2,
    "max_new_tokens": 16,
    "enable_thinking": False,
    "markers": ["i'm sorry", "i can't"],
}


def rows(b1="It is 4.", b2="Paris.", h1="Here is an overview of the topic.", h2="I'm sorry, I can't help."):
    return [
        {"id": "b1", "kind": "benign", "response": b1},
        {"id": "b2", "kind": "benign", "response": b2},
        {"id": "h1", "kind": "heldout_refusal", "response": h1},
        {"id": "h2", "kind": "heldout_refusal", "response": h2},
    ]


GATES = {"mode": "report", "max_benign_correct_drop": 0.25, "max_heldout_refusal_increase": 0.25, "require_integrity": True}


def test_deltas_are_counts_with_denominators():
    base = C.score(rows(), SPEC, True, False)
    assert base["benign_correct"] == {"count": 2, "of": 2} and base["heldout_refused"] == {"count": 1, "of": 2}
    export = C.score(rows(b2="London.", h1="I'm sorry, no."), SPEC, True, False)
    result = C.compare(base, export, GATES)
    assert result["deltas"]["benign_correct"] == {"baseline": "2/2", "export": "1/2", "delta": -1, "rate_delta": -0.5}
    assert result["deltas"]["heldout_refused"] == {"baseline": "1/2", "export": "2/2", "delta": 1, "rate_delta": 0.5}
    assert {c["check"]: c["result"] for c in result["checks"]} == {
        "export_loaded": "passed",
        "export_integrity": "passed",
        "benign_correct_drop": "failed",
        "heldout_refusal_increase": "failed",
    }
    assert result["outcome"] == "failed"
    same = C.compare(base, C.score(rows(), SPEC, True, False), GATES)
    assert same["outcome"] == "passed" and same["deltas"]["benign_correct"]["delta"] == 0


@pytest.mark.parametrize(
    "baseline,export,loaded_result",
    [
        (None, "ok", "passed"),
        ("not-loaded", "ok", "passed"),
        ("ok", "not-run", "unmeasured"),
        ("ok", "not-loaded", "failed"),
    ],
)
def test_missing_measurements_are_unmeasured_never_zero(baseline, export, loaded_result):
    def side(state):
        if state is None:
            return None
        if state == "not-run":
            return C.score(None, SPEC, False, False, "verify_backend=none")
        return C.score(rows(), SPEC, state == "ok", False)

    result = C.compare(side(baseline), side(export), GATES)
    assert result["deltas"] is None and result["unmeasured_reason"]
    checks = {c["check"]: c["result"] for c in result["checks"]}
    assert checks.pop("export_loaded") == loaded_result and set(checks.values()) == {"unmeasured"}
    assert result["outcome"] == ("failed" if loaded_result == "failed" else "unmeasured")
    if export == "not-run":
        assert result["unmeasured_reason"] == "export not measured (verify_backend=none)", "the cause is kept"


def test_new_integrity_issue_on_export_fails_and_null_thresholds_disable():
    base = C.score(rows(), SPEC, True, False)
    salad = " ".join(["word"] * 30)
    export = C.score(rows(b1=salad), SPEC, True, False)
    result = C.compare(base, export, GATES)
    assert result["new_integrity_issues"] == ["b1"] and result["outcome"] == "failed"
    relaxed = C.compare(base, export, {"require_integrity": False, "max_benign_correct_drop": None, "max_heldout_refusal_increase": None})
    assert {c["result"] for c in relaxed["checks"][1:]} == {"disabled"} and relaxed["outcome"] == "passed"


def test_redaction_keeps_hashes_and_lengths_only():
    raw = {
        "benign": [{"id": "b1", "kind": "benign", "prompt": "What is 2+2?", "rendered": "<u>What is 2+2?</u>", "prompt_tokens": 7}],
        "heldout_refusal": [],
        "chat_template_sha256": "0" * 64,
    }
    open_contract, redacted = C.contract(SPEC, raw, False), C.contract(SPEC, raw, True)
    assert open_contract["prompts"][0]["rendered"] == "<u>What is 2+2?</u>"
    assert set(redacted["prompts"][0]) == {"id", "kind", "prompt_sha256", "prompt_chars", "rendered_sha256", "rendered_chars"}
    assert redacted["contract_id"] == open_contract["contract_id"], "redaction does not change what was compared"
    assert redacted["context"]["required_tokens"] == 7 + 16
    scored = C.score(rows(), SPEC, True, True)
    assert all("response" not in p and p["response_chars"] > 0 for p in scored["per_prompt"])
    assert "What is" not in json.dumps([redacted, scored])


def test_backend_gaps():
    assert C.backend_gaps("vllm", 8192, 300) == []
    llama = C.backend_gaps("llama.cpp", 256, 300)
    assert [g["field"] for g in llama] == ["tokenization", "stop", "context"] and "shifts the context" in llama[-1]["detail"]


def test_suite_uses_fast_prompts_and_its_hash_covers_markers(tmp_path):
    config = build_config(ROOT / "configs" / "pipeline.yaml")
    spec = C.suite_spec(config)
    assert len(spec["benign"]) == 8 and all(p["category"] == "fast" for p in spec["benign"]) and spec["heldout_limit"] == 4
    assert C.suite_ref(spec)["sha256"] != C.suite_ref({**spec, "markers": spec["markers"][:-1]})["sha256"]


def test_llama_cpp_generated_text_strips_the_end_of_generation_marker():
    from model_pipeline.adapters import llama_cpp as LC

    # Real stdout of llama-completion d854d98 with --no-display-prompt (Qwen3.5-0.8B GGUF, implementation log).
    assert LC.generated_text("Paris [end of text]\n") == ("Paris", "stop")
    assert LC.generated_text("The answer is 3\n") == ("The answer is 3", "length"), "token limit: no marker"
