"""Fake study tool operating on fake_heretic's JSON journal with the real tool's CLI and output shape.

FAKE_KILL_PARENT_AT = export-settings-before | export-settings-after kills the pipeline around the
export-only settings update (see _fakeutil.kill_parent_at).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "fake_bin"))
from _fakeutil import kill_parent_at, pareto_front  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("command", choices=["export", "set-settings"])
p.add_argument("--journal", required=True)
p.add_argument("--trial-index", type=int, default=0)
p.add_argument("--n-trials", type=int)
p.add_argument("--model-action", choices=["save", "none"])
p.add_argument("--out", required=True)
a = p.parse_args()
study = json.loads(Path(a.journal).read_text())
settings = study["settings"]
if a.command == "export":
    result = {
        "ok": True,
        "finished": study["finished"],
        "n_trials_setting": settings["n_trials"],
        "trial_index_setting": settings.get("trial_index"),
        "model_action_setting": settings.get("model_action"),
        "settings_sha256": hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
        "requested_trial_index": a.trial_index,
        "trials": study["trials"],
        "pareto": pareto_front(study["trials"]),
    }
else:
    export = a.model_action == "save"
    if export:
        kill_parent_at("export-settings-before")
    before = {k: settings.get(k) for k in ("n_trials", "trial_index", "model_action")}
    settings.update(n_trials=a.n_trials, trial_index=a.trial_index, model_action="" if a.model_action == "none" else a.model_action)
    Path(a.journal).write_text(json.dumps(study))
    if export:
        kill_parent_at("export-settings-after")
    result = {"ok": True, "trials": len(study["trials"]), "before": before, "after": {k: settings[k] for k in before}}
Path(a.out).write_text(json.dumps(result))
