"""Read/update a Heretic Optuna study journal (runs inside the heretic environment).

export          - all trials plus the Pareto front sorted exactly like Heretic's trial menu
                  (ascending objective values in configured scorer order).
update-settings - set n_trials / trial_index in the study's stored settings before Heretic is run
                  with checkpoint_action=continue (Heretic restores settings from the study, so this
                  is the only way to change them for a continued study). Uses Optuna's public API,
                  mirroring what Heretic itself does when running additional trials.
"""

from __future__ import annotations

import argparse
import json
import sys

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState


def load(journal: str) -> optuna.Study:
    storage = JournalStorage(JournalFileBackend(journal, lock_obj=JournalFileOpenLock(journal)))
    return optuna.load_study(study_name="heretic", storage=storage)


def trial_dict(trial) -> dict:
    return {
        "number": trial.number,
        "index": trial.user_attrs.get("index"),
        "state": trial.state.name,
        "values": trial.values,
        "parameters": trial.user_attrs.get("parameters"),
        "scores": trial.user_attrs.get("scores"),
        "datetime_start": str(trial.datetime_start) if trial.datetime_start else None,
        "datetime_complete": str(trial.datetime_complete) if trial.datetime_complete else None,
    }


def export(study: optuna.Study, requested: int) -> dict:
    settings = json.loads(study.user_attrs["settings"])
    positions = [i for i, s in enumerate(settings.get("scorers", [])) if s.get("optimization") != "none"]
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pareto = study.best_trials if complete else []

    def key(trial):
        records = trial.user_attrs.get("scores") or []
        return tuple(records[i]["score"]["value"] for i in positions if i < len(records))

    return {
        "ok": True,
        "finished": bool(study.user_attrs.get("finished")),
        "n_trials_setting": settings.get("n_trials"),
        "trial_index_setting": settings.get("trial_index"),
        "requested_trial_index": requested,
        "seed": settings.get("seed"),
        "objective_scorers": [settings["scorers"][i] for i in positions],
        "trials": [trial_dict(t) for t in study.trials],
        "pareto": [trial_dict(t) for t in sorted(pareto, key=key)],
    }


def update(study: optuna.Study, n_trials: int, trial_index: int) -> dict:
    settings = json.loads(study.user_attrs["settings"])
    recorded = len(study.trials)
    settings["n_trials"] = max(n_trials, recorded)
    settings["trial_index"] = trial_index
    study.set_user_attr("settings", json.dumps(settings))
    finished = recorded >= settings["n_trials"]
    study.set_user_attr("finished", finished)
    return {"ok": True, "trials": recorded, "n_trials": settings["n_trials"], "trial_index": trial_index, "finished": finished}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["export", "update-settings"])
    parser.add_argument("--journal", required=True)
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    study = load(args.journal)
    if args.command == "export":
        result = export(study, args.trial_index)
    else:
        result = update(study, args.n_trials or len(study.trials), args.trial_index)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
