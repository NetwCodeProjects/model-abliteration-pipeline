"""Fake Heretic for CPU tests. Reads config.toml from the working directory like the real tool.

Behaviour switches (env): FAKE_HERETIC_MODE = ok | config-error | crash-mid | no-output;
FAKE_TRIAL_SECONDS (sleep per trial). Writes a JSON 'journal' (the fake study tool reads it), prints
the same progress lines the adapter parses, drops 'mtp.*' tensors like the real merged export.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _fakeutil import copy_model, count, log  # noqa: E402

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


class Stop(Exception):
    pass


def handler(_signum, _frame):
    raise KeyboardInterrupt


def main() -> int:
    if "--help" in sys.argv:
        print("usage: heretic [-h] [--model str] [--model-commit str] [--n-trials int] ... (fake)")
        return 0
    count("heretic")
    signal.signal(signal.SIGINT, handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, handler)
    mode = os.environ.get("FAKE_HERETIC_MODE", "ok")
    cfg = tomllib.loads(Path("config.toml").read_text(encoding="utf-8"))
    if mode == "config-error":
        print("Configuration contains 1 errors:")
        print("n_trials: Input should be greater than 0")
        return 0
    save = Path(cfg["save_directory"])
    if "--reproduce" in sys.argv:
        doc = json.loads(Path(sys.argv[sys.argv.index("--reproduce") + 1]).read_text(encoding="utf-8"))
        assert doc["settings"]["model_action"] == "save" and doc["settings"]["trial_index"] == 0
        print("Restoring model from reproduction information...")
        copy_model(Path(doc["settings"]["model"]), save, drop_prefix="mtp.", flip=True)
        print(f"Model saved to {save}.")
        return 0
    journal_dir = Path(cfg.get("study_checkpoint_dir", "checkpoints"))
    journal_dir.mkdir(exist_ok=True)
    journal = journal_dir / "model.jsonl"
    settings = {k: cfg[k] for k in ("n_trials", "trial_index", "seed", "scorers") if k in cfg}
    settings.setdefault("scorers", [{"plugin": "KeywordRate", "optimization": "minimize"}, {"plugin": "KLDivergence", "optimization": "minimize"}])
    study = {"settings": settings, "trials": [], "finished": False}
    if journal.exists() and cfg.get("checkpoint_action") == "continue":
        study = json.loads(journal.read_text(encoding="utf-8"))
        settings = study["settings"]  # like Heretic: continued studies restore their stored settings
        print("Resuming existing study.")
    n_trials = int(settings["n_trials"])
    trial_seconds = float(os.environ.get("FAKE_TRIAL_SECONDS", "0"))
    try:
        for index in range(len(study["trials"]) + 1, n_trials + 1):
            print(f"Running trial {index} of {n_trials}...", flush=True)
            log("heretic", f"trial {index}")
            time.sleep(trial_seconds)
            refusals = round(0.5 / index, 4)
            kl = round(float(os.environ.get("FAKE_KL_STEP", "0.001")) * index, 4)
            study["trials"].append(
                {
                    "number": index - 1,
                    "index": index,
                    "state": "COMPLETE",
                    "parameters": {"start_layer_index": index},
                    "scores": [{"name": "Refusals", "score": {"value": refusals}}, {"name": "KL divergence", "score": {"value": kl}}],
                }
            )
            journal.write_text(json.dumps(study), encoding="utf-8")
            if mode == "crash-mid" and index == 2:
                print("RuntimeError: CUDA error: an illegal memory access was encountered", file=sys.stderr)
                return 1
    except KeyboardInterrupt:
        print("Stopping study (interrupt)")
    if len(study["trials"]) >= n_trials:
        study["finished"] = True
    journal.write_text(json.dumps(study), encoding="utf-8")
    if not study["trials"]:
        print("Shutting down...")
        return 0
    if mode == "no-output":
        return 0
    pareto = sorted(study["trials"], key=lambda t: tuple(s["score"]["value"] for s in t["scores"]))
    chosen = pareto[min(int(settings.get("trial_index", 0)), len(pareto) - 1)]
    print(f"Restoring model from trial {chosen['index']}...")
    copy_model(Path(cfg["model"]), save, drop_prefix="mtp.", flip=True)
    print(f"Model saved to {save}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
