# Resume, rerun and fork recipes

```bash
# Start, stop after acquisition, continue later
./main.sh --through 2 --yes
./main.sh --list-runs
./main.sh --resume 20261026T120000Z-a1b2c3              # 00 re-checks the host; 01/02 reused; 03+ run

# Interrupted during Heretic (Ctrl+C, SSH drop, OOM): the Optuna journal is kept
./main.sh --resume 20261026T120000Z-a1b2c3              # stage 05 continues the same study

# Give Heretic more time on resume (operational keys may change)
./main.sh --resume 20261026T120000Z-a1b2c3 --max-runtime-minutes 240

# The budget stopped a finished stage 05 early: continue that search with more time (same study)
./main.sh --resume 20261026T120000Z-a1b2c3 --from 5 --rerun --max-runtime-minutes 240

# Re-run qualification only (05 must have a valid receipt)
./main.sh --resume 20261026T120000Z-a1b2c3 --only 6 --rerun

# Re-run 5..6 explicitly (search only if the budget is not reached; the export is reused if valid)
./main.sh --resume 20261026T120000Z-a1b2c3 --from 5 --through 6 --rerun

# Gate failed: export another trial (Heretic trial 7) from a copy of the study, zero new trials
./main.sh --resume 20261026T120000Z-a1b2c3 --fork --trial-id 7

# No eligible trial: continue the same study with a larger budget (history action 'continue')
./main.sh --resume 20261026T120000Z-a1b2c3 --fork --max-trials 80

# Add a format later: only stages that depend on it run
./main.sh --resume 20261026T120000Z-a1b2c3 --fork --quantizations fp8,iq4_xs,iq3_s,q8_0

# One format failed (exit 20): fix the cause, resume; finished formats are reused
./main.sh --resume 20261026T120000Z-a1b2c3

# Rejected: changing the model of an existing run (exit 7, E_RESUME_CONFIG_CONFLICT)
./main.sh --resume 20261026T120000Z-a1b2c3 --model-url other/model
```

Rules: a stage is reused only if its receipt succeeded, its key (config subset + upstream keys + inputs)
is unchanged and its outputs still validate. Forks copy the parent's receipts and artifact registry,
so only invalidated stages execute. With `execution.resume: true` (default), a plain `./main.sh --yes`
automatically resumes the most recent incomplete run that has the *identical* config hash.
