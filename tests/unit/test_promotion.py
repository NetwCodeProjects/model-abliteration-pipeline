"""Promotion decision table (docs/evaluation.md): every single-gate row, then mixed-gate cases."""

from __future__ import annotations

import itertools

import pytest

from model_pipeline import configspec
from model_pipeline.evaluation.promotion import decide, evidence_line
from model_pipeline.evaluation.qualification import evaluate_gates

RUN_MODES = ("optimize", "exact_reproduction_attempt", "approximate_reproduction")
ORIGINS = ({"kind": "policy"}, {"kind": "manual", "trial_id": 3}, {"kind": "pareto_position", "position": 0})
OVERRIDES = ("none", "per_gate", "global")

# The table from docs/evaluation.md written out by hand, one gate under test (all others passed):
# (kind, result, profile, override) -> (outcome, status of the gate). Override "any" = all three.
REQUIRED = {
    ("passed", "qualified", "any"): ("promoted:qualified", "passed"),
    ("passed", "research", "any"): ("promoted:qualified", "passed"),
    ("failed", "qualified", "none"): ("not_promoted", "failed"),
    ("failed", "research", "none"): ("not_promoted", "failed"),
    ("failed", "qualified", "set"): ("promoted:ungated", "bypassed_by_override"),
    ("failed", "research", "set"): ("promoted:ungated", "bypassed_by_override"),
    ("unmeasured", "qualified", "none"): ("not_promoted", "unmeasured"),
    ("unmeasured", "research", "none"): ("promoted:research", "unmeasured"),
    ("unmeasured", "qualified", "set"): ("promoted:ungated", "bypassed_by_override"),
    ("unmeasured", "research", "set"): ("promoted:research", "unmeasured"),
}
OPTIONAL = {  # optional gates never change the outcome and are never bypassed
    "passed": ("promoted:qualified", "passed", False),
    "failed": ("promoted:qualified", "failed", True),
    "unmeasured": ("promoted:qualified", "unmeasured", True),
    "disabled": ("promoted:qualified", "disabled", False),
}


def gates(**results: str) -> list[dict]:
    return [
        {"gate": g, "value": None, "threshold": None, "op": "<=", "result": results.get(g, "passed"), "passed": None, "provenance": "test"}
        for g in configspec.GATE_ORDER
    ]


def policy(profile: str = "qualified", optional=(), override: str = "none", gate: str | None = None) -> dict:
    return {
        "profile": profile,
        "optional_gates": list(optional),
        "override_gates": [gate] if override == "per_gate" else [],
        "allow_ungated_promotion": override == "global",
    }


def status_of(decision: dict, gate: str) -> dict:
    return next(g for g in decision["gates"] if g["gate"] == gate)


@pytest.mark.parametrize("gate", configspec.GATE_ORDER)
@pytest.mark.parametrize("result", ("passed", "failed", "unmeasured", "disabled"))
@pytest.mark.parametrize("profile", ("qualified", "research"))
@pytest.mark.parametrize("override", OVERRIDES)
def test_single_gate_rows_required(gate, result, profile, override):
    if result == "disabled":
        with pytest.raises(ValueError):
            decide(gates(**{gate: result}), policy(profile, override=override, gate=gate), run_mode="optimize", origin=ORIGINS[0])
        return
    expected = REQUIRED.get((result, profile, "any")) or REQUIRED[(result, profile, "none" if override == "none" else "set")]
    for run_mode, origin in itertools.product(RUN_MODES, ORIGINS):  # neither changes the outcome
        d = decide(gates(**{gate: result}), policy(profile, override=override, gate=gate), run_mode=run_mode, origin=origin)
        assert (d["outcome"], status_of(d, gate)["status"]) == expected
        assert d["promoted"] is (expected[0] != "not_promoted")
        assert d["run_mode"] == run_mode and d["candidate_origin"] == origin
        if expected[1] == "bypassed_by_override":
            key = "selection.allow_ungated_promotion" if override == "global" else "selection.override_gates"
            assert status_of(d, gate)["override_key"] == key
        if expected[0] == "promoted:qualified":
            assert d["reason"] == "all_required_gates_passed" and d["deciding_gate"] is None
        else:
            assert d["deciding_gate"] == gate and d["triggered_gates"] == [gate]


@pytest.mark.parametrize("gate", configspec.GATE_ORDER)
@pytest.mark.parametrize("result", ("passed", "failed", "unmeasured", "disabled"))
@pytest.mark.parametrize("profile", ("qualified", "research"))
@pytest.mark.parametrize("override", OVERRIDES)
def test_single_gate_rows_optional(gate, result, profile, override):
    outcome, status, warns = OPTIONAL[result]
    d = decide(gates(**{gate: result}), policy(profile, optional=[gate], override=override, gate=gate), run_mode="optimize", origin=ORIGINS[0])
    assert d["outcome"] == outcome and status_of(d, gate)["status"] == status and not status_of(d, gate)["required"]
    assert bool(d["warnings"]) is warns and d["triggered_gates"] == []


def test_mixed_passed_unmeasured_and_overridden_failure_is_ungated():
    d = decide(
        gates(benign_pass_rate="unmeasured", optimizer_kl="failed"),
        policy("research", override="per_gate", gate="optimizer_kl"),
        run_mode="optimize",
        origin=ORIGINS[0],
    )
    assert d["outcome"] == "promoted:ungated" and d["reason"] == "required_gate_bypassed_by_override"
    assert d["deciding_gate"] == "optimizer_kl" and d["triggered_gates"] == ["optimizer_kl", "benign_pass_rate"]


def test_mixed_unoverridden_failure_beats_research_acceptance():
    d = decide(gates(reload="unmeasured", heldout_refusal_rate="failed"), policy("research"), run_mode="optimize", origin=ORIGINS[0])
    assert d["outcome"] == "not_promoted" and d["reason"] == "required_gate_failed" and d["deciding_gate"] == "heldout_refusal_rate"
    assert d["triggered_gates"] == ["reload", "heldout_refusal_rate"], "every triggering gate is listed, in canonical order"


def test_mixed_two_failures_use_canonical_order():
    d = decide(gates(optimizer_refusal_rate="failed", optimizer_kl="failed"), policy(), run_mode="optimize", origin=ORIGINS[0])
    assert d["deciding_gate"] == "optimizer_kl" and d["triggered_gates"] == ["optimizer_kl", "optimizer_refusal_rate"]


def test_mixed_optional_failure_with_required_passed_is_qualified_with_warning():
    d = decide(gates(benign_pass_rate="failed"), policy(optional=["benign_pass_rate"]), run_mode="optimize", origin=ORIGINS[0])
    assert d["outcome"] == "promoted:qualified" and d["warnings"] == ["optional gate benign_pass_rate failed"]
    assert "optional gates: benign_pass_rate" in evidence_line(d)


def test_mixed_override_does_not_hide_other_failures():
    d = decide(gates(optimizer_kl="failed", integrity="failed"), policy(override="per_gate", gate="optimizer_kl"), run_mode="optimize", origin=ORIGINS[0])
    assert d["outcome"] == "not_promoted" and d["deciding_gate"] == "integrity"
    assert status_of(d, "optimizer_kl")["status"] == "bypassed_by_override"


def test_no_path_qualifies_with_a_required_gate_not_passed():
    for results in itertools.product(("passed", "failed", "unmeasured"), repeat=len(configspec.GATE_ORDER)):
        for profile, override in itertools.product(("qualified", "research"), ("none", "global")):
            d = decide(
                gates(**dict(zip(configspec.GATE_ORDER, results, strict=True))), policy(profile, override=override), run_mode="optimize", origin=ORIGINS[0]
            )
            if d["outcome"] == "promoted:qualified":
                assert all(g["status"] == "passed" for g in d["gates"])


def test_evaluate_gates_reports_every_gate_even_without_qualification():
    selection = configspec.defaults()["selection"]
    result = evaluate_gates(selection, {"optimizer_refusal_rate": 0.1, "optimizer_kl": 0.5}, None)
    assert [g["gate"] for g in result] == list(configspec.GATE_ORDER)
    by = {g["gate"]: g["result"] for g in result}
    assert by == {
        "reload": "unmeasured",
        "integrity": "unmeasured",
        "optimizer_kl": "failed",
        "benign_pass_rate": "unmeasured",
        "heldout_refusal_rate": "unmeasured",
        "optimizer_refusal_rate": "passed",
    }
    assert evaluate_gates({**selection, "max_optimizer_kl": None}, {"optimizer_kl": 0.5}, None)[2]["result"] == "disabled"
