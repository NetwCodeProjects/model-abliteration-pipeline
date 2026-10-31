"""Read/update a Heretic Optuna study journal (runs inside the heretic environment).

export       - all trials plus the Pareto front sorted exactly like Heretic's trial menu
               (ascending objective values in configured scorer order).
set-settings - set n_trials / trial_index / model_action in the study's stored settings before
               Heretic is run with checkpoint_action=continue (a continued study restores its stored
               settings and ignores config.toml for them). Sets exactly the given values: export-only
               passes n_trials = len(study.trials) so Heretic adds zero trials (docs/step0-feasibility.md).
               Uses Optuna's public API, as Heretic itself does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.WARNING)
CONTROLLED = ("n_trials", "trial_index", "model_action")


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
    raw = study.user_attrs["settings"]
    settings = json.loads(raw)
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
        "model_action_setting": settings.get("model_action"),
        "settings_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "requested_trial_index": requested,
        "seed": settings.get("seed"),
        "objective_scorers": [settings["scorers"][i] for i in positions],
        "trials": [trial_dict(t) for t in study.trials],
        "pareto": [trial_dict(t) for t in sorted(pareto, key=key)],
    }


def set_settings(study: optuna.Study, n_trials: int, trial_index: int, model_action: str) -> dict:
    settings = json.loads(study.user_attrs["settings"])
    before = {k: settings.get(k) for k in CONTROLLED}
    settings.update(n_trials=n_trials, trial_index=trial_index, model_action=model_action)
    study.set_user_attr("settings", json.dumps(settings))
    return {"ok": True, "trials": len(study.trials), "before": before, "after": {k: settings[k] for k in CONTROLLED}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["export", "set-settings"])
    parser.add_argument("--journal", required=True)
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--model-action", choices=["save", "none"], help="none = '' (search only: finish without saving)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    study = load(args.journal)
    if args.command == "export":
        result = export(study, args.trial_index)
    else:
        if args.n_trials is None or args.model_action is None:
            parser.error("set-settings needs --n-trials and --model-action")
        result = set_settings(study, args.n_trials, args.trial_index, "" if args.model_action == "none" else args.model_action)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
