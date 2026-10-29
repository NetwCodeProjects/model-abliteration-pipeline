# Step 0 feasibility: search-only and export-only Heretic operation

Result: **both behaviors are achievable with the pinned engine, without modifying Heretic.**
Steps 2–3 of [PROJECT_A_NEXT_CHANGES_PROMPT.md](PROJECT_A_NEXT_CHANGES_PROMPT.md) may proceed,
subject to the constraints listed below.

Date: 2026-10-29. The pipeline code was not changed; this document records engine behavior only.

## Environment

| Item | Value |
|---|---|
| Host | Windows 11, WSL2 Ubuntu 24.04 (kernel 6.18.54.1), CPU only |
| Heretic | `171bb98b8b9a41c081631b5e678f7133128973e0` (pipeline-provisioned env `heretic-d2213fd61ce7`) |
| optuna / torch / transformers | 5.0.0 / 2.14.1+cpu / 5.18.0 |
| Model | `tiny-random/qwen3.8@346bb76e` (`Qwen3_5ForConditionalGeneration`), ARA modifier |
| Engine config | `tests/realtools/step0/base.toml` (the config Project A generated in run `20261026T091348Z-4bb489`), seed 1234 |

## Reproduce

```bash
REALTOOLS_WS=/path/to/provisioned/workspace bash tests/realtools/step0/run_all.sh   # about 5 minutes on 8 CPU cores
```

`study.py` (runs in the Heretic venv) inspects the Optuna journal, patches the stored settings
through Optuna's public API, and compares restored trial parameters with the recorded ones.
The full output of the recorded run and per-experiment logs and journal dumps are kept outside the
repository in `../step0-evidence/`. The checks were run twice (an exploratory run, then `run_all.sh`), and
both runs gave the same trial identifiers and weight hashes.

## How it works in the engine (`src/heretic/main.py`)

- `checkpoint_action = "continue"` replaces the current settings with the study's **stored**
  settings (`Settings.model_validate_init_only`). Only fields excluded from serialization, such as
  `save_directory`, come from the current `config.toml`. `n_trials`, `trial_index` and
  `model_action` for a continued study must therefore be set in the stored settings.
- After continuing, Heretic runs `study.optimize(n_trials = settings.n_trials - len(study.trials))`.
  `len(study.trials)` counts every recorded trial, including `PRUNED` and `RUNNING` ones.
- `trial_index` selects a position in the Pareto front (`study.best_trials` sorted by objective
  values). Heretic prints `Restoring model from trial N`, where N is the trial's `index` user attribute.
- With `model_action` set, the action loop runs once. `model_action = ""` takes the
  "return to trial selection" branch. Because `trial_index` is set, the trial loop also ends, so
  Heretic exits without saving.

## Results

| # | Check | Evidence | Result |
|---|---|---|---|
| A | Search-only, normal finish: 5 trials, `model_action = ""` | 5 × `Running trial`, exit 0, no save directory created | **pass** |
| A | Export-only, Pareto position 2, stored `n_trials` = 5 = recorded | `Resuming existing study.`, no `Running trial`, `Restoring model from trial 5`, `Model saved` | **pass**; trials 5 → 5, identical trial set |
| A | The same export repeated, then position 0 | Trial 5 again, then trial 3. The parameters each log printed equal that trial's recorded parameters | **pass** |
| A | `config.toml` said `trial_index = 0`, `model_action = ""` | It was ignored: the stored settings won | confirms the precedence above |
| D | Export-only bytes vs a direct export of the same weight-changing trial (KL 0.0051) | Weight hash: direct `634b7d27…`, export-only `634b7d27…`, source `3de72190…` | **identical**, and different from the source |
| C | Budget stop: SIGINT once trial 4 of 10 started, `model_action = ""` | Exit 0. States: 3 `COMPLETE`, 1 `PRUNED`; `finished = false`; no save | **pass** (search-only after a budget stop) |
| C | Export-only after the budget stop, stored `n_trials` = 4 = `len(study.trials)` | No `Running trial`, `Restoring model from trial 3`, saved | **pass**; trials 4 → 4, identical trial set |
| C-short | Stored `n_trials` = 3 (COMPLETE count only, below recorded) | No new trials (Optuna is passed a negative count), saved; `finished` stays `false` | works, but depends on the negative-count behavior. Not the chosen rule |
| C-hazard | **Current pipeline behavior**: `update-settings --n-trials 10`, so `n_trials = max(10, 4)` | `Running trial 5 … 10 of 10`, then export | **6 trials added**: the known hazard is confirmed |
| K | Hard kill: SIGKILL while trial 3 of 10 was running | Journal: 2 `COMPLETE`, 1 `RUNNING` (orphaned; Optuna never reclaims it); `finished = false` | recorded |
| K | Export-only after the hard kill, stored `n_trials` = 3 = `len(study.trials)` | `Restoring model from trial 2`, saved | **pass**; trials 3 → 3, identical trial set |

## Constraints the implementation must respect

1. **Export-only contract.** Set the stored settings to `n_trials = len(study.trials)` (all
   states), `trial_index = <Pareto position of the chosen trial>` and `model_action = "save"`, then
   run with `checkpoint_action = "continue"`. Verify afterwards that the trial set is unchanged.
   Verify that `Restoring model from trial N` names the expected trial `index`, and fail the stage
   otherwise. The existing `heretic_study_tool.py update-settings` cannot set `model_action`, and
   stage 05 passes the configured budget ([stage05_abliterate.py:112](../src/model_pipeline/stages/stage05_abliterate.py#L112)),
   which is the C-hazard path. Both must change.
2. **Search-only contract.** Use `model_action = ""` with `trial_index` set. Heretic still
   restores one trial in memory after the search. That is cheap on the tiny model, but costs one
   ARA modification on a large model. This cost is unmeasured on GPU.
3. **Only Pareto-front trials are exportable.** `trial_index` indexes `study.best_trials`, so a
   dominated trial cannot be exported headlessly. The Step 2 policies always pick a
   Pareto-optimal trial if ties are broken on the other objective (the minimum on one objective
   within a threshold on the other). A manual `--trial-id` naming a dominated trial must be
   refused with a clear error rather than mapped to some position.
4. **Ties.** On this model the refusals score is always 0/5, and three trials tie at KL 0.0000.
   Heretic's position-to-trial mapping matched the sort in `study.py` in every case. That mapping
   still depends on Optuna's `best_trials` order for ties, so always map by trial identifier and
   check the restore line. Never trust a position on its own.
5. **Exporting erases interruption state.** Exporting from a budget-stopped or killed study sets
   `finished = true` and overwrites the stored `n_trials` (10 → 4). The original budget and the
   interruption must be recorded in `study-history.jsonl` *before* the export (Step 3).
6. **Orphaned `RUNNING` trials.** A hard kill leaves a `RUNNING` trial that counts towards
   `len(study.trials)`, and so towards the budget. Step 3's resume reconciliation must report it.
   Selection must use `COMPLETE` trials only. This was already the plan.
7. **The weight-level evidence is limited** by the tiny model. Its KL-0 trials leave every tensor
   byte-identical to the source, so check D (a weight-changing trial) is the only check at the
   weight level. Positions above 0 are confirmed by the restore line and printed parameters, not
   by weights.

## Not verified

- Behavior on GPU and on a full-size model (time and memory of the restore during search-only).
- SIGINT during the export itself.
- Studies with more than two objectives, or with `optimization = "none"` scorers.
