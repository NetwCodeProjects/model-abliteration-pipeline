# Study lifecycle: search, selection, export and history (stage 05)

Steps 2–3 of [PROJECT_A_NEXT_CHANGES_PROMPT.md](PROJECT_A_NEXT_CHANGES_PROMPT.md). The engine
mechanics this design relies on were verified with the real pinned Heretic in
[step0-feasibility.md](step0-feasibility.md); the constraints listed there are referenced as C1–C7.

Applies to `abliteration.experiment: optimize`. Reproduction runs (`exact_reproduction_attempt`,
`approximate_reproduction`) have no study and keep their single `--reproduce` invocation.

## Phases

Stage 05 keeps its public ID and runs four internal phases, all under the study lock:

| Phase | Engine call | Skipped when |
|---|---|---|
| 1. locate | none (copies the parent journal for a fork) | — |
| 2. search | Heretic with `model_action = ""` (search-only, C2) | the study already has `n_trials` recorded trials, or the history records a finished search with the same search fingerprint (e.g. a budget stop) |
| 3. select | none (reads the journal with the study tool) | never: cheap and deterministic |
| 4. export | Heretic export-only (C1) | a validated candidate for the same export fingerprint exists |

### Search-only and export-only

- **Search.** A new study is started with `checkpoint_action = "restart"`, `model_action = ""` and
  `trial_index = 0`. A continued study first gets its *stored* settings set to
  `n_trials = <budget>`, `trial_index = 0`, `model_action = ""` (Heretic ignores `config.toml` for
  these when continuing). A wall-clock budget stop (SIGINT) ends the search without saving.
- **Export.** Stored settings are set to `n_trials = len(study.trials)` (every state),
  `trial_index = <Pareto position of the chosen trial>` and `model_action = "save"`, then Heretic
  runs with `checkpoint_action = "continue"`. Afterwards the stage verifies that the trial set is
  unchanged (`E_EXPORT_ADDED_TRIALS`) and that Heretic's `Restoring model from trial N` names the
  chosen trial (`E_EXPORT_TRIAL_MISMATCH`). The configured budget is never passed on this path (C1).
- `scripts/runtime/heretic_study_tool.py set-settings` sets exactly the given values. The former
  `update-settings` command (`n_trials = max(requested, recorded)`, the C-hazard) is removed.

## Fingerprints

All four are recorded in the stage 05 receipt (`diagnostics.fingerprints`) and in the history.

| Fingerprint | Inputs | Decides |
|---|---|---|
| **study identity** (`study_id`) | native Heretic settings without control/budget keys (`n_trials`, `trial_index`, `model_action`, `save_directory`, `checkpoint_action`): model source, scorers, modifiers, datasets, seed, …; engine commit; **tool environment** | which physical study (`checkpoints/heretic/study-<study_id>/`) |
| **search** | study identity + `abliteration.max_trials` + `abliteration.max_runtime_minutes` | whether phase 2 runs |
| **selection** | study identity + recorded trial set (numbers, states, values) + `selection.policy`, `selection.max_optimizer_refusal_rate`, `selection.max_optimizer_kl`, score names, manual choice, `selection.allow_ineligible_choice` | phase 3 result |
| **export** | study identity + chosen trial (`number`, `index`) + engine commit + export strategy + source artifact | the candidate directory `c-<export fp>-t<index>` |

`inputs_id` is the study identity without the environment; it is used only to detect an
environment change (below).

`n_startup_trials` is excluded from the identity when it is derived from the budget
(`abliteration.n_startup_trials: null` → `min(template value, max_trials // 3)`): a continued
study keeps the value stored at its creation, so a budget increase must not create a new study.
An explicitly configured value is part of the identity.

Tool environment = from the Heretic component record: `torch` (its version string carries the
CUDA build, e.g. `2.10.0+cu128`), `transformers`, `optuna`, `python`, and the torch source/backend
of the component spec.

### Reuse rules

| Change | Result |
|---|---|
| selection inputs only (thresholds, policy, manual choice) | selection and export only; **zero** new trials |
| export inputs only | export only |
| budget increase (`max_trials`, or a longer time limit after a budget stop) | search **continues** the same study (`continue`); same study identity and journal |
| budget decrease below the recorded trials | no search; selection sees every recorded trial |
| study identity | a different study |

The stage key of stage 05 includes the selection inputs (optimize only), so a fork that changes
them re-executes stage 05, and the phases above decide what actually runs.

## Selection (phase 3)

Selection uses **optimizer metrics only** (Heretic's in-loop scores). It never uses the held-out
qualification prompts of stage 06, so those stay held out. Multi-candidate qualification is not
implemented; if it is added later it must use a separate final holdout.

1. Candidates are the `COMPLETE` trials (C6). `PRUNED`, `FAIL` and orphaned `RUNNING` trials are
   listed but never eligible.
2. A trial is **eligible** when its refusal score ≤ `selection.max_optimizer_refusal_rate` and its
   KL score ≤ `selection.max_optimizer_kl`. A `null` threshold does not constrain. A trial without
   the configured score names is ineligible.
3. Eligible trials are ranked by `selection.policy`:
   - `min_refusals_within_kl` (default): refusal, then KL, then trial number;
   - `min_kl_within_refusals`: KL, then refusal, then trial number.

   Breaking ties on the other objective makes the winner Pareto-optimal, so it is exportable (C3).
4. No eligible trial → `E_NO_ELIGIBLE_TRIAL` (exit 9), no export; the error and
   `selection-candidates.json` list the three closest trials (smallest relative threshold excess).

### Manual choice

| Key | CLI | Meaning |
|---|---|---|
| `abliteration.trial_id` | `--trial-id N` | Heretic trial `index` N (the N in `Restoring model from trial N`) |
| `abliteration.pareto_trial_index` | `--pareto-index N` | position N of the Pareto front sorted like Heretic's menu; kept for compatibility |

- Setting both is a config error (`E_TRIAL_CHOICE_CONFLICT`, exit 3).
- A Pareto position is resolved to a trial identifier **once**, at selection time; only the
  identifier (`number` and `index`) is used afterwards.
- Unknown trial, a trial that is not `COMPLETE`, or a position past the front → `E_TRIAL_NOT_FOUND` (exit 3).
- A trial that is not on the Pareto front cannot be exported headlessly (C3) → `E_TRIAL_NOT_EXPORTABLE` (exit 3).
- A manual choice **selects**; it does not bypass promotion (Step 1). An ineligible manual choice
  is exported only with `selection.allow_ineligible_choice: true`, recorded in the selection, the
  history and the candidate. Stage 07 then still applies every gate.

### `metrics/selection-candidates.json`

```json
{
  "schema_version": "1.0",
  "study_id": "…", "selection_fingerprint": "…",
  "policy": "min_refusals_within_kl",
  "constraints": {"max_optimizer_refusal_rate": 0.2, "max_optimizer_kl": 0.1},
  "score_names": {"refusal": "Refusals", "kl": "KL divergence"},
  "manual": null, "allow_ineligible_choice": false,
  "chosen": {"number": 2, "index": 3, "pareto_position": 2, "refusal_rate": 0.03, "kl": 0.0916, "eligible": true, "by": "policy", "ineligible_override": false},
  "origin": {"kind": "policy", "policy": "min_refusals_within_kl"},
  "error": null,
  "complete_trials": 12, "eligible_trials": 2,
  "trials": [{"number": 2, "index": 3, "state": "COMPLETE", "pareto_position": 2, "refusal_rate": 0.03, "kl": 0.0916,
              "eligible": true, "reasons": [], "rank": 1}],
  "closest": [],
  "note": "optimizer metrics only; the Pareto position is informational"
}
```

## Study history (`<study dir>/study-history.jsonl`)

Append-only JSON lines, written only under the study lock. Every line has `schema_version`,
`seq`, `utc`, `run_id`, `invocation_id`, `phase` and `action`.

| Phase | Written | Content |
|---|---|---|
| `intent` | **before** the study is modified | `planned` (settings/budget to set, fingerprint), `before` state |
| `completion` | after success | `after` state, `trials_added`, plus `budget_stopped` (search), `selection` (select), `export` (export) |
| `interruption` / `failure` | when the invocation stops early | `after` state, `trials_added`, error id |
| `reconciled` | on the next invocation, for an intent without an end entry | the journal's observed state, `trials_added` since the intent, orphaned `RUNNING` trials |

State = `trials` (all states), `states` (count per state), `trial_set_sha256`, stored `n_trials`,
`finished`, `settings_sha256`.

| Action | Adds trials | Notes |
|---|---|---|
| `new` | yes | first search of a new study |
| `continue` | yes | search on an existing study (resume, budget increase) |
| `replay` | yes | new study with the same inputs as a study of another run (`replay_of`) |
| `select` | no | single `completion` line: chosen trial, how, selection fingerprint |
| `export` | no (verified) | records the stored `n_trials` and `finished` **before** export-only overwrites them (C5) |
| `fork-continue` | no | first line(s) of a fork's study: parent run, parent study, trials at copy time, the parent's finished search fingerprints |

The history alone reconstructs the decision sequence: which searches ran with which budget, how
many trials each added, which trial was selected by what rule, and which trial was exported where.

## Locking

- **Study lock** `<study dir>/.lock`, same stale-owner semantics as the run lock (a lock left by a
  dead process on the same host is reclaimed). Held for the whole of stage 05's optimize path.
  Order: run lock first, then study lock. A second holder fails fast with `E_STUDY_LOCKED` (exit 12).
- **Forks own independent copies.** A fork copies the parent journal into its own study directory
  (atomic replace) while holding the *parent's* study lock, and never writes to the parent's study
  or history. Its history begins with `fork-continue`.

## Environment identity

The tool environment is part of the study identity. If no study exists for the current identity
but one with the same `inputs_id` exists in this run, or in the parent of a fork, under another
environment, stage 05 refuses with `E_STUDY_ENVIRONMENT_CHANGED` (exit 7). The way forward is
either to restore the recorded environment (shown in the error and in `study.json`) or to start a
new study with `--new-run`.

A new study whose inputs (`inputs_id`) match a study of another run in the workspace is recorded
as `replay` with `replay_of`: same resolved inputs, new study. This needs a fixed
`abliteration.seed`, because the derived seed differs per run.

Why refuse: the trials of one study form one Pareto front only if their scores were measured with
the same numerics. A different torch/CUDA build or transformers version can change refusal counts
and KL values, and an export-only restore under a different environment may not reproduce the
weights that were scored. Silently continuing would mix both.

Design note: a config key to start a new study in place was considered and dropped. A resume
cannot change the configuration and a fork can change it only through CLI flags, so such a key
could not be reached when it is needed.

## Resume and kill recovery

On entry (under the study lock) every intent without an end entry is reconciled against the
journal and reported as a warning. Then the phases run as usual:

| Killed after | History on resume | Resumed outcome |
|---|---|---|
| settings update (export) | `export` intent → `reconciled` (no trials added) | export redone, zero trials |
| a completed trial (search) | `new` intent → `reconciled` (`trials_added` = completed trials) | search continues to the budget (`continue`) |
| selection | `select` completion, `export` intent → `reconciled` | export redone |
| export (model saved, not yet verified) | `export` intent → `reconciled` | partial export removed, export redone |

## Error ids

| Error id | Exit | When |
|---|---|---|
| `E_NO_ELIGIBLE_TRIAL` | 9 | no completed trial meets the optimizer constraints |
| `E_MANUAL_TRIAL_INELIGIBLE` | 9 | manual choice fails the constraints and `allow_ineligible_choice` is false |
| `E_TRIAL_NOT_FOUND` | 3 | manual trial unknown, not `COMPLETE`, or Pareto position out of range |
| `E_TRIAL_NOT_EXPORTABLE` | 3 | manual trial is not on the Pareto front |
| `E_TRIAL_CHOICE_CONFLICT` | 3 | both `trial_id` and `pareto_trial_index` set |
| `E_HERETIC_SEARCH_FAILED` | 8 | Heretic exited non-zero during the search (other search failures keep `E_HERETIC_CONFIG`, `E_HERETIC_BUDGET_EXHAUSTED`, `E_HERETIC_TIMEOUT`) |
| `E_EXPORT_ADDED_TRIALS` | 8 | export-only changed the trial set |
| `E_EXPORT_TRIAL_MISMATCH` | 8 | Heretic restored a different trial than selected |
| `E_STUDY_LOCKED` | 12 | another process holds the study lock |
| `E_STUDY_ENVIRONMENT_CHANGED` | 7 | study exists under another tool environment |

## Migration notes

- The study directory hash now includes the tool environment, so studies created before this
  change are not found by new forks (they start a new study). Existing runs are unaffected.
- `abliteration.pareto_trial_index` defaults to `null` (policy selection). An explicit value,
  including `0`, is now a manual choice.
- Candidate directories are named `c-<export fingerprint>-t<trial index>`.
- `heretic_study_tool.py update-settings` is replaced by `set-settings`.
- The realtools test config now loads `configs/profiles/qwen3_5.yaml`, lists the gates it disables
  in `selection.optional_gates` (Step 1) and uses `fp8_dynamic` (implementation-log finding 18).

## Verification

| Check | How | Result |
|---|---|---|
| Selection on the A100 study | unit test on the recorded scores (`tests/fixtures/a100_study/`); realtools test with the **real** study tool on a copy of the A100 Optuna journal | trial `index` 3 (`number` 2, refusals 0.03, KL 0.0916), Pareto position 2; the journal copy is byte-identical afterwards |
| A100 replay, zero new trials | fake engine replaying the 12 recorded trials | 12 trials searched once; export-only restores trial 3 |
| Search-only → select → export-only | realtools, **real** Heretic `171bb98b` on CPU, `tiny-random/qwen3.8`, 5 trials | `Running trial 1…5`, no save; then `Resuming existing study.`, no `Running trial`, `Restoring model from trial 5`, saved; trial-set digest unchanged |
| Budget increase in a fork | realtools, real engine, 5 → 8 | `continue`: `Running trial 6…8` only, then export-only; parent study files unchanged |
| `--trial-id` fork at Pareto position 1 | realtools, real engine | `Restoring model from trial 6`, zero new trials |
| `replay` | realtools, fixed seed, second run in the same workspace | first history action `replay`, `replay_of` the earlier study |
| Selection reuse (KL, refusal, policy, manual), rerun reuse, kill recovery at four points, study lock, fork lock, environment change | fake engine (`tests/integration/test_study_lifecycle.py`); the kill tests hard-kill the pipeline process | as specified above |

Not verified: GPU runs and the time/memory of the in-memory restore that search-only still does
(C2); kill recovery with the real engine (fakes only); SIGINT during a real export-only invocation.
