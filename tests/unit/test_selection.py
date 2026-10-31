"""Step 2-3: constraint-based selection (real A100 study scores), study identity and history."""

from __future__ import annotations

import json
import sys

import pytest
from conftest import FAKE_BIN, FIXTURES

from llm_surgeon.adapters import heretic_study as HS
from llm_surgeon.evaluation.selection import select_trial

A100 = json.loads((FIXTURES / "a100_study" / "study.json").read_text())


def a100_export() -> dict:
    by_index = {t["index"]: t for t in A100["trials"]}
    return {"trials": A100["trials"], "pareto": [by_index[i] for i in A100["pareto_indices"]], "n_trials_setting": 12, "finished": True}


def choose(export=None, **kw):
    args = {
        "study_id": "s",
        "policy": "min_refusals_within_kl",
        "max_refusal": 0.2,
        "max_kl": 0.1,
        "refusal_name": "Refusals",
        "kl_name": "KL divergence",
        **kw,
    }
    return select_trial(export or a100_export(), **args)


def test_a100_study_selects_trial_3_not_pareto_position_0():
    doc = choose()
    chosen = doc["chosen"]
    assert (chosen["index"], chosen["number"], chosen["pareto_position"]) == (3, 2, 2)
    assert chosen["refusal_rate"] == 0.03 and chosen["kl"] == 0.0916491448879242
    assert doc["eligible_trials"] == 2 and [r["index"] for r in doc["trials"] if r["eligible"]] == [3, 5]
    position0 = next(r for r in doc["trials"] if r["pareto_position"] == 0)
    assert position0["index"] == 8 and position0["reasons"] == ["KL 11.81 > 0.1"], "the trial that destroyed the A100 model"
    assert doc["origin"] == {"kind": "policy", "policy": "min_refusals_within_kl"} and doc["error"] is None


def test_policy_and_thresholds_change_the_choice():
    assert choose(policy="min_kl_within_refusals")["chosen"]["index"] == 5
    assert choose(max_kl=0.08)["chosen"]["index"] == 5
    assert choose(max_kl=None)["chosen"]["index"] == 8, "an unconstrained KL reproduces the old position-0 choice"


def test_no_eligible_trial_lists_the_closest():
    doc = choose(max_refusal=0.02)
    assert doc["chosen"] is None and doc["error"]["error_id"] == "E_NO_ELIGIBLE_TRIAL"
    # relative excess over the limits: trial 3 = 0.5, trial 6 = 3 + 0.39, trial 5 = 4
    assert [r["index"] for r in doc["closest"]] == [3, 6, 5]
    assert "trial 3" in doc["error"]["message"]


@pytest.mark.parametrize(
    "kw,error",
    [
        ({"trial_id": 99}, "E_TRIAL_NOT_FOUND"),
        ({"pareto_position": 4}, "E_TRIAL_NOT_FOUND"),
        ({"trial_id": 6}, "E_MANUAL_TRIAL_INELIGIBLE"),
        ({"trial_id": 6, "allow_ineligible": True}, "E_TRIAL_NOT_EXPORTABLE"),
        ({"trial_id": 8}, "E_MANUAL_TRIAL_INELIGIBLE"),
    ],
)
def test_manual_choice_errors(kw, error):
    assert choose(**kw)["error"]["error_id"] == error


def test_manual_choice_resolves_position_to_trial_once():
    doc = choose(pareto_position=3)
    assert doc["chosen"]["index"] == 5 and doc["manual"] == {"by": "pareto_position", "value": 3}
    forced = choose(trial_id=8, allow_ineligible=True)
    assert forced["chosen"]["index"] == 8 and forced["chosen"]["ineligible_override"] is True
    assert forced["origin"] == {"kind": "manual", "by": "trial_id", "value": 8, "ineligible_override": True}


def test_only_completed_trials_are_candidates():
    export = a100_export()
    export["trials"] = [*export["trials"], {"number": 12, "index": 13, "state": "RUNNING", "values": None, "scores": None}]
    doc = choose(export)
    running = doc["trials"][-1]
    assert running["eligible"] is False and running["reasons"][0] == "state RUNNING"
    assert choose(export, trial_id=13)["error"]["error_id"] == "E_TRIAL_NOT_FOUND"
    assert doc["selection_fingerprint"] != choose()["selection_fingerprint"], "the trial set is part of the selection fingerprint"


def test_fake_engine_pareto_matches_the_real_front():
    sys.path.insert(0, str(FAKE_BIN))
    from _fakeutil import pareto_front

    assert [t["index"] for t in pareto_front(A100["trials"])] == A100["pareto_indices"]


def test_study_identity_ignores_budget_but_not_environment():
    native = {"model": "/m", "seed": 1, "n_trials": 50, "trial_index": 0, "model_action": "save", "save_directory": "/x"}
    env = {"torch": "2.10.0+cu128", "transformers": "5.18.0"}
    base = HS.study_identity(native, "c", env)
    assert HS.study_identity({**native, "n_trials": 80, "model_action": "", "trial_index": 3}, "c", env) == base
    other = HS.study_identity(native, "c", {**env, "torch": "2.10.0+cu130"})
    assert other["inputs_id"] == base["inputs_id"] and other["study_id"] != base["study_id"]
    assert HS.study_identity({**native, "seed": 2}, "c", env)["inputs_id"] != base["inputs_id"]
    startup = {**native, "n_startup_trials": 16}
    assert HS.study_identity(startup, "c", env) != HS.study_identity({**startup, "n_startup_trials": 26}, "c", env), "explicit value"
    assert HS.study_identity(startup, "c", env, ("n_startup_trials",)) == HS.study_identity(
        {**startup, "n_startup_trials": 26}, "c", env, ("n_startup_trials",)
    )


def test_history_pairs_intents_with_end_entries(tmp_path):
    history = HS.History(tmp_path / "study-history.jsonl")
    a = history.intent("new", "r1", search_fingerprint="f1", before={"trials": 0})
    history.append("completion", "new", a, "r1", search_fingerprint="f1", budget_stopped=True)
    b = history.intent("export", "r1", before={"trials": 4})
    assert [e["invocation_id"] for e in history.open_intents()] == [b]
    assert history.finished_searches() == {"f1"}
    history.append("reconciled", "export", b, "r2")
    assert history.open_intents() == [] and [e["seq"] for e in history.entries()] == [1, 2, 3, 4]
    fork = HS.History(tmp_path / "fork.jsonl")
    c = fork.intent("fork-continue", "r3")
    fork.append("completion", "fork-continue", c, "r3", parent_finished_searches=sorted(history.finished_searches()))
    assert fork.finished_searches() == {"f1"}, "a fork inherits its parent's finished searches"
