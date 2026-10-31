"""Steps 2-3 (docs/study-lifecycle.md) with the fake engine: selection and export reuse, study
history, kill recovery, study locking and environment identity."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest
from conftest import FIXTURES

from model_pipeline.errors import ExitCode

A100 = FIXTURES / "a100_study" / "study.json"


def receipt05(env, run):
    return env.receipt(run, "05")


def history(env, run) -> list[dict]:
    path = Path(receipt05(env, run)["diagnostics"]["history"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def steps(entries) -> list[tuple[str, str]]:
    return [(e["action"], e["phase"]) for e in entries]


def decisions(entries) -> list[tuple[str, int | None]]:
    """The decision sequence, reconstructed from the history alone."""
    out = []
    for e in entries:
        if e["phase"] != "completion":
            continue
        if e["action"] == "select":
            out.append(("select", e["selection"]["chosen"]["index"]))
        elif e["action"] == "export":
            out.append(("export", e["export"]["index"]))
        elif e["action"] == "fork-continue":
            out.append(("fork-continue", e["trials_at_copy"]))
        else:
            out.append((e["action"], e["trials_added"]))
    return out


def started_trials(env) -> list[int]:
    log = env.counters / "heretic.log"
    return [int(t) for t in log.read_text().split() if t.isdigit()] if log.exists() else []


def study_files(env, run) -> dict[Path, bytes]:
    root = Path(receipt05(env, run)["diagnostics"]["study_dir"])
    return {p: p.read_bytes() for p in root.rglob("*") if p.is_file() and p.suffix in (".jsonl", ".json")}


def a100_run(env, monkeypatch, *extra):
    monkeypatch.setenv("FAKE_TRIALS_FILE", str(A100))
    env.provision()
    return env.run("--max-trials", "12", "--through", "5", *extra)


def with_overrides(monkeypatch, extra: dict):
    """Config overrides for a fork that the CLI has no flag for (thresholds, policy)."""
    import model_pipeline.cli as cli

    original = cli.to_overrides
    monkeypatch.setattr(cli, "to_overrides", lambda args: {**original(args), **extra})


def test_a100_study_replay_selects_trial_3_with_zero_new_trials(env, monkeypatch):
    code, out = a100_run(env, monkeypatch)
    assert code == 0, out
    run = env.latest_run()
    candidate = json.loads((run / "metrics" / "candidate.json").read_text())
    summary = candidate["summary"]
    assert summary["exported_trial_index"] == 3 and summary["optimizer_refusal_rate"] == 0.03 and summary["optimizer_kl"] == 0.0916491448879242
    assert candidate["selection"]["chosen"]["pareto_position"] == 2, "position 2 is only where trial 3 sits in this study"
    assert started_trials(env) == list(range(1, 13)), "the export added no trials"
    entries = history(env, run)
    assert steps(entries) == [("new", "intent"), ("new", "completion"), ("select", "completion"), ("export", "intent"), ("export", "completion")]
    assert decisions(entries) == [("new", 12), ("select", 3), ("export", 3)]
    export = entries[-1]
    assert export["trials_added"] == 0 and export["export"]["restored_trial"] == 3
    assert entries[3]["before"]["stored_n_trials"] == 12, "the stored budget is recorded before export-only overwrites it"
    fingerprints = receipt05(env, run)["diagnostics"]["fingerprints"]
    assert all(fingerprints.values()) and len(set(fingerprints.values())) == 4


@pytest.mark.parametrize(
    "overrides,cli,expected",
    [
        ({"selection.max_optimizer_kl": 0.08}, [], 5),
        ({"selection.max_optimizer_refusal_rate": 0.02}, [], "E_NO_ELIGIBLE_TRIAL"),
        ({"selection.policy": "min_kl_within_refusals"}, [], 5),
        ({}, ["--trial-id", "5"], 5),
    ],
    ids=["kl-threshold", "refusal-threshold", "policy", "manual-trial"],
)
def test_selection_change_in_a_fork_reselects_and_exports_without_new_trials(env, monkeypatch, overrides, cli, expected):
    assert a100_run(env, monkeypatch)[0] == 0
    parent = env.latest_run()
    parent_files = study_files(env, parent)
    invocations = env.count("heretic")
    with_overrides(monkeypatch, overrides)
    code, out = env.run("--resume", parent.name, "--fork", *cli)
    fork = env.latest_run()
    assert fork.name != parent.name
    assert started_trials(env) == list(range(1, 13)), "zero new trials"
    assert {p: p.read_bytes() for p in parent_files} == parent_files, "the parent's journal and history are untouched"
    ours, theirs = receipt05(env, fork)["diagnostics"]["fingerprints"], receipt05(env, parent)["diagnostics"]["fingerprints"]
    assert ours["study_identity"] == theirs["study_identity"] and ours["search"] == theirs["search"]
    assert ours["selection"] != theirs["selection"]
    entries = history(env, fork)
    assert steps(entries)[:3] == [("fork-continue", "intent"), ("fork-continue", "completion"), ("select", "completion" if expected == 5 else "failure")]
    assert not any(e["action"] in ("new", "continue", "replay") for e in entries), "no search ran"
    if expected == 5:
        assert code == 0, out
        assert json.loads((fork / "metrics" / "candidate.json").read_text())["summary"]["exported_trial_index"] == 5
        assert env.count("heretic") == invocations + 1, "export only"
    else:
        assert code == ExitCode.GATE and receipt05(env, fork)["error"]["error_id"] == expected
        closest = json.loads((fork / "metrics" / "selection-candidates.json").read_text())["closest"]
        assert [r["index"] for r in closest] == [3, 6, 5]
        assert env.count("heretic") == invocations, "no export"


def test_budget_increase_continues_the_same_study(env, monkeypatch):
    assert a100_run(env, monkeypatch)[0] == 0
    parent = env.latest_run()
    parent_files = study_files(env, parent)
    code, out = env.run("--resume", parent.name, "--fork", "--max-trials", "15")
    assert code == 0, out
    fork = env.latest_run()
    diag, parent_diag = receipt05(env, fork)["diagnostics"], receipt05(env, parent)["diagnostics"]
    assert diag["fingerprints"]["study_identity"] == parent_diag["fingerprints"]["study_identity"]
    assert Path(diag["study_dir"]).name == Path(parent_diag["study_dir"]).name
    assert diag["fingerprints"]["search"] != parent_diag["fingerprints"]["search"]
    assert started_trials(env) == [*range(1, 13), 13, 14, 15], "continued at trial 13, nothing repeated"
    journal = json.loads(next(Path(diag["study_dir"], "checkpoints").glob("*.jsonl")).read_text())
    original = json.loads(next(p for p in parent_files if p.suffix == ".jsonl" and p.parent.name == "checkpoints").read_text())
    assert journal["trials"][:12] == original["trials"] and len(journal["trials"]) == 15
    assert {p: p.read_bytes() for p in parent_files} == parent_files
    assert decisions(history(env, fork)) == [("fork-continue", 12), ("continue", 3), ("select", 3), ("export", 3)]


def test_rerun_reuses_the_search_and_the_export(env, monkeypatch):
    assert a100_run(env, monkeypatch)[0] == 0
    run = env.latest_run()
    invocations = env.count("heretic")
    code, out = env.run("--resume", run.name, "--only", "5", "--rerun")
    assert code == 0, out
    assert env.count("heretic") == invocations, "neither search nor export ran again"
    phases = receipt05(env, run)["diagnostics"]["engine"]["phases"]
    assert phases["search"]["ran"] is False and phases["export"]["ran"] is False
    assert steps(history(env, run))[-1] == ("select", "completion")


@pytest.mark.parametrize(
    "point,open_action,added",
    [
        ("export-settings-after", "export", 0),  # killed right after the export settings update
        ("trial:2", "new", 2),  # killed after a trial completed
        ("export-settings-before", "export", 0),  # killed after selection, before the study was touched
        ("export-saved", "export", 0),  # killed after Heretic saved, before verification
    ],
    ids=["settings-update", "trial-completion", "selection", "export"],
)
def test_kill_then_resume_reconciles_the_history(env, point, open_action, added):
    env.provision()
    code = env.run_killable("--max-trials", "5", "--through", "5", FAKE_KILL_PARENT_AT=point)
    assert code != 0
    run = env.latest_run()
    assert not list((run / "receipts").glob("05-*.json")), "the killed process wrote no stage 05 receipt"
    entries_at_kill = [json.loads(line) for line in next((run / "checkpoints" / "heretic").glob("study-*/study-history.jsonl")).read_text().splitlines()]
    assert entries_at_kill[-1]["phase"] == "intent" and entries_at_kill[-1]["action"] == open_action
    if point == "export-settings-before":
        assert steps(entries_at_kill)[-2] == ("select", "completion")
    code, out = env.run("--resume", run.name)
    assert code == 0, out
    entries = history(env, run)
    reconciled = [e for e in entries if e["phase"] == "reconciled"]
    assert [(e["action"], e["trials_added"], e["invocation_id"]) for e in reconciled] == [(open_action, added, entries_at_kill[-1]["invocation_id"])]
    assert started_trials(env) == [1, 2, 3, 4, 5], "every trial ran exactly once"
    assert steps(entries)[-2:] == [("export", "intent"), ("export", "completion")] and entries[-1]["export"]["restored_trial"] == 5
    assert not list(Path(receipt05(env, run)["diagnostics"]["candidate_path"]).parent.glob("*.partial"))


def lock_file(path: Path) -> None:
    path.write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "since": 0, "purpose": "test"}))


def test_concurrent_invocation_on_a_study_is_refused(env, monkeypatch):
    env.provision()
    monkeypatch.setenv("FAKE_HERETIC_MODE", "crash-mid")
    assert env.run("--through", "5")[0] == ExitCode.ENGINE
    run = env.latest_run()
    study = Path(receipt05(env, run)["diagnostics"]["study_dir"])
    entries = history(env, run)
    lock_file(study / ".lock")
    monkeypatch.delenv("FAKE_HERETIC_MODE")
    code, _ = env.run("--resume", run.name)
    assert code == ExitCode.LOCKED and receipt05(env, run)["error"]["error_id"] == "E_STUDY_LOCKED"
    assert history(env, run) == entries, "nothing was written without the lock"


def test_fork_needs_the_parent_study_lock_to_copy(env, monkeypatch):
    assert a100_run(env, monkeypatch)[0] == 0
    parent = env.latest_run()
    lock_file(Path(receipt05(env, parent)["diagnostics"]["study_dir"]) / ".lock")
    code, _ = env.run("--resume", parent.name, "--fork", "--trial-id", "5")
    fork = env.latest_run()
    assert code == ExitCode.LOCKED and receipt05(env, fork)["error"]["error_id"] == "E_STUDY_LOCKED"
    assert history(env, fork) == [] and not list(Path(receipt05(env, fork)["diagnostics"]["study_dir"]).glob("checkpoints/*.jsonl"))


def test_environment_change_is_refused_and_a_new_run_is_recorded_as_replay(env, monkeypatch):
    env.config_path = env.write_config({"abliteration": {"seed": 1234}})
    env.base_args[1] = str(env.config_path)
    env.provision()
    monkeypatch.setenv("FAKE_HERETIC_MODE", "crash-mid")
    assert env.run("--max-trials", "5", "--through", "5")[0] == ExitCode.ENGINE
    monkeypatch.delenv("FAKE_HERETIC_MODE")
    run = env.latest_run()
    old_study = Path(receipt05(env, run)["diagnostics"]["study_dir"])
    old_files = study_files(env, run)
    components_path = env.workspace / "tools" / "components.json"
    components = json.loads(components_path.read_text())
    components["components"]["heretic"]["versions"]["transformers"] = "5.19.0"
    components_path.write_text(json.dumps(components))
    code, _ = env.run("--resume", run.name)
    error = receipt05(env, run)["error"]
    assert code == ExitCode.DEPENDENCY and error["error_id"] == "E_STUDY_ENVIRONMENT_CHANGED"
    assert error["details"]["environment_changes"] == {"transformers": ["5.18.0", "5.19.0"]}
    assert {p: p.read_bytes() for p in old_files} == old_files, "the old study is untouched"
    code, out = env.run("--max-trials", "5", "--through", "5", "--new-run")
    assert code == 0, out
    first = history(env, env.latest_run())[0]
    assert first["action"] == "replay" and first["phase"] == "intent"
    record = json.loads((Path(receipt05(env, env.latest_run())["diagnostics"]["study_dir"]) / "study.json").read_text())
    assert record["replay_of"] == [str(old_study)] and record["environment"]["transformers"] == "5.19.0"
