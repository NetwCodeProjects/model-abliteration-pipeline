"""Step 0 helper (runs in the heretic venv): inspect or patch a Heretic study journal.

inspect JOURNAL           -> JSON: study attrs, every trial, Pareto front sorted like Heretic's menu
patch JOURNAL k=v ...     -> set keys in the stored settings (json values), print before/after
check-restore JOURNAL LOG... -> per engine log: the trial Heretic restored and whether the parameters it
                             printed equal that trial's recorded parameters
"""

import hashlib
import json
import re
import sys

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load(path):
    storage = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
    return optuna.load_study(study_name="heretic", storage=storage)


def params_hash(p):
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:12]


def inspect(study):
    settings = json.loads(study.user_attrs["settings"])
    positions = [i for i, s in enumerate(settings["scorers"]) if s.get("optimization") != "none"]
    first = next((t for t in study.trials if t.user_attrs.get("scores")), None)
    names = [first.user_attrs["scores"][i]["name"] for i in positions] if first else []
    trials = [
        {
            "number": t.number,
            "index": t.user_attrs.get("index"),
            "state": t.state.name,
            "values": t.values,
            "params_hash": params_hash(t.user_attrs.get("parameters")),
        }
        for t in study.trials
    ]

    def key(t):
        return tuple(next((s["score"]["value"] for s in t.user_attrs["scores"] if s["name"] == n), None) for n in names)

    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pareto = sorted(study.best_trials, key=key) if complete else []
    return {
        "finished": study.user_attrs.get("finished"),
        "settings": {k: settings.get(k) for k in ("n_trials", "trial_index", "model_action", "checkpoint_action")},
        "objective_names": names,
        "n_trials_recorded": len(study.trials),
        "states": {s: sum(1 for t in trials if t["state"] == s) for s in {t["state"] for t in trials}},
        "trials": trials,
        "pareto": [{"position": i, "number": t.number, "index": t.user_attrs.get("index"), "key": key(t)} for i, t in enumerate(pareto)],
    }


def main():
    cmd, path = sys.argv[1], sys.argv[2]
    study = load(path)
    if cmd == "inspect":
        print(json.dumps(inspect(study), indent=1))
    elif cmd == "check-restore":
        recorded = {t.user_attrs.get("index"): t.user_attrs.get("parameters") for t in study.trials}
        for log in sys.argv[3:]:
            text = open(log, encoding="utf-8").read()
            found = re.search(r"Restoring model from trial (\d+)", text)
            if not found:
                print(log, "no restore line")
                continue
            index = int(found.group(1))
            block = text[found.end() :].split("* Resetting")[0]
            printed = dict(re.findall(r"\* (\w+) = (\S+)", block))
            params = recorded[index]
            match = set(printed) == set(params) and all(
                abs(float(printed[k]) - v) < 1e-4 if isinstance(v, float) else str(v) == printed[k] for k, v in params.items()
            )
            print(log, "restored trial index", index, "| printed parameters == recorded:", match)
    elif cmd == "patch":
        settings = json.loads(study.user_attrs["settings"])
        before = {}
        for pair in sys.argv[3:]:
            k, v = pair.split("=", 1)
            before[k] = settings.get(k)
            settings[k] = json.loads(v)
        study.set_user_attr("settings", json.dumps(settings))
        print(json.dumps({"before": before, "after": {k: settings[k] for k in before}}))


if __name__ == "__main__":
    main()
