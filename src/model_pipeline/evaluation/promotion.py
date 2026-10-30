"""Promotion decision table (docs/evaluation.md, "Gates and promotion").

``decide`` is a pure function: per-gate results (``qualification.evaluate_gates``) plus the
selection policy in, one outcome + one machine-readable reason + per-gate statuses out. Required
gates are combined with a fixed precedence where the weakest claim wins; ties are broken by the
canonical gate order. Candidate origin and run mode are recorded but never change the outcome.
"""

from __future__ import annotations

from typing import Any

from ..configspec import GATE_ORDER

OUTCOMES = ("promoted:qualified", "promoted:research", "promoted:ungated", "not_promoted")
STATUSES = ("passed", "failed", "unmeasured", "disabled", "bypassed_by_override")

EVIDENCE_CAVEAT = "fast qualification is a smoke test; deep evaluation is Project B"


def decide(gates: list[dict[str, Any]], selection: dict[str, Any], *, run_mode: str, origin: dict[str, Any]) -> dict[str, Any]:
    profile = selection.get("profile") or "qualified"
    optional = set(selection.get("optional_gates") or [])
    if selection.get("allow_ungated_promotion"):
        overridable, override_key = set(GATE_ORDER), "selection.allow_ungated_promotion"
    else:
        overridable, override_key = set(selection.get("override_gates") or []), "selection.override_gates"
    rank = {name: i for i, name in enumerate(GATE_ORDER)}
    statuses, warnings = [], []
    for gate in sorted(gates, key=lambda g: rank[g["gate"]]):
        name, result = gate["gate"], gate["result"]
        required = name not in optional
        status, allowed_by = result, None
        if required:
            if result == "disabled":
                # Config validation rejects this (E_GATE_REQUIREMENT_MISSING); never treat it as passed.
                raise ValueError(f"required gate {name!r} is disabled")
            if name in overridable and (result == "failed" or (result == "unmeasured" and profile == "qualified")):
                status, allowed_by = "bypassed_by_override", override_key
        elif result in ("failed", "unmeasured"):
            warnings.append(f"optional gate {name} {result}")
        statuses.append({**gate, "required": required, "status": status, "override_key": allowed_by})

    def required_with(status: str) -> list[str]:
        return [s["gate"] for s in statuses if s["required"] and s["status"] == status]

    failed, bypassed, unmeasured = required_with("failed"), required_with("bypassed_by_override"), required_with("unmeasured")
    if failed:
        outcome, reason, deciding = "not_promoted", "required_gate_failed", failed
    elif bypassed:
        outcome, reason, deciding = "promoted:ungated", "required_gate_bypassed_by_override", bypassed
    elif unmeasured:
        outcome = "promoted:research" if profile == "research" else "not_promoted"
        reason, deciding = "required_gate_unmeasured", unmeasured
    else:
        outcome, reason, deciding = "promoted:qualified", "all_required_gates_passed", []
    triggered = sorted(set(failed) | set(bypassed) | set(unmeasured), key=rank.__getitem__)
    return {
        "outcome": outcome,
        "promoted": outcome != "not_promoted",
        "reason": reason,
        "deciding_gate": deciding[0] if deciding else None,
        "triggered_gates": triggered,
        "profile": profile,
        "run_mode": run_mode,
        "candidate_origin": origin,
        "gates": statuses,
        "warnings": warnings,
    }


def evidence_line(decision: dict[str, Any]) -> str:
    """One-line evidence status for reports."""
    outcome = decision["outcome"]
    if decision["reason"] == "all_required_gates_passed":
        detail = "all required gates passed"
    else:
        detail = f"{decision['reason'].replace('_', ' ')}: {', '.join(decision['triggered_gates'])}"
    optional = [s["gate"] for s in decision["gates"] if not s["required"]]
    if optional:
        detail += f"; optional gates: {', '.join(optional)}"
    return f"{outcome} ({decision['profile']} profile, {detail}); {EVIDENCE_CAVEAT}"
