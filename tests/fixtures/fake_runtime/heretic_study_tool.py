"""Fake study tool operating on fake_heretic's JSON journal with the real tool's CLI and output shape."""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("command")
p.add_argument("--journal", required=True)
p.add_argument("--trial-index", type=int, default=0)
p.add_argument("--n-trials", type=int)
p.add_argument("--out", required=True)
a = p.parse_args()
study = json.loads(Path(a.journal).read_text())
if a.command == "export":
    pareto = sorted(study["trials"], key=lambda t: tuple(s["score"]["value"] for s in t["scores"]))
    result = {
        "ok": True,
        "finished": study["finished"],
        "n_trials_setting": study["settings"]["n_trials"],
        "trial_index_setting": study["settings"].get("trial_index"),
        "requested_trial_index": a.trial_index,
        "trials": study["trials"],
        "pareto": pareto,
    }
else:
    study["settings"]["n_trials"] = max(a.n_trials or 0, len(study["trials"]))
    study["settings"]["trial_index"] = a.trial_index
    study["finished"] = len(study["trials"]) >= study["settings"]["n_trials"]
    Path(a.journal).write_text(json.dumps(study))
    result = {"ok": True, "trials": len(study["trials"]), "n_trials": study["settings"]["n_trials"], "finished": study["finished"]}
Path(a.out).write_text(json.dumps(result))
