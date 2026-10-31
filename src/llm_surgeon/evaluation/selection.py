"""Constraint-based choice of the study trial to export (docs/study-lifecycle.md, phase 3).

Optimizer metrics only: the held-out prompts of stage 06 never influence the choice. A Pareto
position is resolved to a trial identifier here, once; only the identifier is used afterwards.
"""

from __future__ import annotations

from typing import Any

from ..errors import ExitCode
from ..fsutil import sha256_obj

SCHEMA_VERSION = "1.0"
# Rank keys per policy; the second key breaks ties on the other objective, which keeps the winner
# on the Pareto front (only Pareto trials can be exported headlessly, step0-feasibility.md C3).
POLICIES = {"min_refusals_within_kl": ("refusal_rate", "kl"), "min_kl_within_refusals": ("kl", "refusal_rate")}
NOTE = (
    "Selection uses Heretic's optimizer metrics only; held-out qualification prompts are never used here. "
    "The Pareto position is informational: the trial is identified by its number and index."
)


def score(trial: dict[str, Any], name: str) -> float | None:
    for record in trial.get("scores") or []:
        if record.get("name") == name:
            return (record.get("score") or {}).get("value")
    return None


def trial_set_sha256(trials: list[dict[str, Any]]) -> str:
    """Digest of the recorded trial set: numbers, states and objective values."""
    return sha256_obj([[t.get("number"), t.get("state"), t.get("values")] for t in trials])


def _excess(row: dict[str, Any], limits: dict[str, float | None]) -> float:
    total = 0.0
    for key, limit in limits.items():
        if limit is not None and row[key] > limit:
            total += (row[key] - limit) / limit if limit > 0 else row[key]
    return total


def select_trial(
    export: dict[str, Any],
    *,
    study_id: str,
    policy: str,
    max_refusal: float | None,
    max_kl: float | None,
    refusal_name: str,
    kl_name: str,
    trial_id: int | None = None,
    pareto_position: int | None = None,
    allow_ineligible: bool = False,
) -> dict[str, Any]:
    """Return the selection document. ``error`` is set (and ``chosen`` is None) when nothing can be exported."""
    pareto = export.get("pareto") or []
    positions = {t["number"]: i for i, t in enumerate(pareto)}
    limits = {"refusal_rate": max_refusal, "kl": max_kl}
    rows = []
    for trial in export.get("trials") or []:
        refusal, kl = score(trial, refusal_name), score(trial, kl_name)
        reasons = []
        if trial.get("state") != "COMPLETE":
            reasons.append(f"state {trial.get('state')}")
        missing = [n for n, v in ((refusal_name, refusal), (kl_name, kl)) if v is None]
        if missing:
            reasons.append(f"score missing: {', '.join(missing)}")
        else:
            if max_refusal is not None and refusal > max_refusal:
                reasons.append(f"refusal rate {refusal:g} > {max_refusal:g}")
            if max_kl is not None and kl > max_kl:
                reasons.append(f"KL {kl:.4g} > {max_kl:g}")
        rows.append(
            {
                "number": trial.get("number"),
                "index": trial.get("index"),
                "state": trial.get("state"),
                "pareto_position": positions.get(trial.get("number")),
                "refusal_rate": refusal,
                "kl": kl,
                "eligible": not reasons,
                "reasons": reasons,
                "rank": None,
            }
        )
    first, second = POLICIES[policy]
    eligible = sorted((r for r in rows if r["eligible"]), key=lambda r: (r[first], r[second], r["number"]))
    for rank, row in enumerate(eligible, 1):
        row["rank"] = rank
    scored = [r for r in rows if r["state"] == "COMPLETE" and r["refusal_rate"] is not None and r["kl"] is not None]
    closest = [] if eligible else [{**r, "excess": round(_excess(r, limits), 6)} for r in sorted(scored, key=lambda r: (_excess(r, limits), r["number"]))[:3]]
    manual = None
    if trial_id is not None:
        manual = {"by": "trial_id", "value": trial_id}
    elif pareto_position is not None:
        manual = {"by": "pareto_position", "value": pareto_position}
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "selection_fingerprint": sha256_obj(
            {
                "study_id": study_id,
                "trial_set": trial_set_sha256(export.get("trials") or []),
                "policy": policy,
                "constraints": limits,
                "score_names": [refusal_name, kl_name],
                "manual": manual,
                "allow_ineligible": allow_ineligible,
            }
        ),
        "policy": policy,
        "constraints": {"max_optimizer_refusal_rate": max_refusal, "max_optimizer_kl": max_kl},
        "score_names": {"refusal": refusal_name, "kl": kl_name},
        "manual": manual,
        "allow_ineligible_choice": allow_ineligible,
        "chosen": None,
        "origin": None,
        "error": None,
        "complete_trials": sum(1 for r in rows if r["state"] == "COMPLETE"),
        "eligible_trials": len(eligible),
        "trials": rows,
        "closest": closest,
        "note": NOTE,
    }

    def fail(code: ExitCode, error_id: str, message: str, hint: str) -> dict[str, Any]:
        doc["error"] = {"exit_code": int(code), "error_id": error_id, "message": message, "hint": hint}
        return doc

    if manual is None:
        if not eligible:
            near = "; ".join(f"trial {r['index']} (refusal {r['refusal_rate']:g}, KL {r['kl']:.4g})" for r in closest) or "none scored"
            return fail(
                ExitCode.GATE,
                "E_NO_ELIGIBLE_TRIAL",
                f"No completed trial meets the optimizer constraints (refusal <= {max_refusal}, KL <= {max_kl}); closest: {near}",
                "Continue the search with a larger budget, relax the constraints, or choose a trial with --trial-id (see metrics/selection-candidates.json).",
            )
        row = eligible[0]
    else:
        if manual["by"] == "trial_id":
            row = next((r for r in rows if r["index"] == trial_id), None)
            label = f"Trial {trial_id}"
        else:
            row = next((r for r in rows if r["number"] == pareto[pareto_position]["number"]), None) if pareto_position < len(pareto) else None
            label = f"Pareto position {pareto_position} (front has {len(pareto)} trials)"
        if row is None or row["state"] != "COMPLETE":
            what = "does not exist in this study" if row is None else f"is {row['state']}; only completed trials can be exported"
            return fail(ExitCode.CONFIG, "E_TRIAL_NOT_FOUND", f"{label} {what}", "List the trials in metrics/selection-candidates.json.")
        if not row["eligible"] and not allow_ineligible:
            return fail(
                ExitCode.GATE,
                "E_MANUAL_TRIAL_INELIGIBLE",
                f"Trial {row['index']} fails the optimizer constraints: {'; '.join(row['reasons'])}",
                "Choose an eligible trial, or set selection.allow_ineligible_choice: true to export it anyway (recorded; gates still apply).",
            )
    if row["pareto_position"] is None:
        return fail(
            ExitCode.CONFIG,
            "E_TRIAL_NOT_EXPORTABLE",
            f"Trial {row['index']} is not on the Pareto front; Heretic can export only Pareto-optimal trials headlessly",
            f"Pareto trials: {', '.join(str(t.get('index')) for t in pareto)}.",
        )
    by = manual["by"] if manual else "policy"
    doc["chosen"] = {
        **{k: row[k] for k in ("number", "index", "pareto_position", "refusal_rate", "kl", "eligible")},
        "by": by,
        "ineligible_override": not row["eligible"],
    }
    doc["origin"] = (
        {"kind": "policy", "policy": policy}
        if manual is None
        else {"kind": "manual", "by": by, "value": manual["value"], "ineligible_override": not row["eligible"]}
    )
    return doc
