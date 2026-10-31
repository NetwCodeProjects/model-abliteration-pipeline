# Project A — next changes: trustworthy promotion, constraint-based selection, export behavior

You are continuing work on **llm-surgeon** (Project A), a CLI that runs Heretic
abliteration and FP8/GGUF quantization with receipts, resume and artifact manifests.

- Repository: `llm-surgeon/` (GitHub `NetwCodeProjects/llm-surgeon`, branch `main`).
- Read first: `README.md`, `docs/architecture.md`, `docs/stages.md`, `docs/evaluation.md`,
  `docs/heretic-adapter.md`, `docs/artifact-schema.md`, `docs/limitations.md`,
  `docs/implementation-log.md`, `docs/PROJECT_A_AGENT_HANDOFF.md` (original specification; its rules still apply).
- Evidence from the last real GPU run (A100, Qwen3.5-2B): `../a100-evidence/` (logs, receipts,
  metrics, manifests of run `20261029T091307Z-0f7e92` and its fork `20261029T092717Z-0858a9`).
- Planning notes: `../future-planning.md` (observability is **out of scope** for this task).

Goal: **a promoted artifact never claims more than its evidence supports, and the path search →
selection → export → assessment is explicit, recorded and replayable.** Do not add engines, backends
or CLI features beyond what is listed here.

## Ground rules

- Do not invent engine flags, TOML keys or APIs. Heretic is pinned to `171bb98b`; verify behavior in
  its source and, where a step depends on runtime behavior, with the real engine on CPU
  (`tests/realtools`, model `tiny-random/qwen3.8`, needs Linux/WSL2 — Heretic headless fails on Windows).
- Stage IDs 00–10 and existing error IDs/exit codes are public contracts. Add new error IDs; never
  rename existing ones. Bump schema versions only for breaking changes, with migration notes.
- Every behavior gets CPU tests (unit + mocked integration with `tests/fixtures/` fakes). Keep fakes
  faithful: when a fake encodes tool behavior, cite the real-tool evidence in a comment.
- Never claim GPU results you did not obtain. Docs describe only behavior covered by a test or a
  recorded real run.
- Record findings in `docs/implementation-log.md`. Commit after each step; before each commit run
  `ruff check`, `ruff format --check`, `python scripts/generate_schema.py --check` and the full suite.
- Write design decisions (tables below) into the docs **before** implementing the code that depends on them.

## Step 0 — Feasibility gate: search-only and export-only Heretic operation (do first)

Steps 2–3 depend on two engine behaviors. Prove them with the real pinned Heretic before building on them.

1. **Search without saving.** Find a verified, non-interactive way to finish an optimization run
   without exporting a model (e.g. `model_action` behavior and trial/action loop termination in
   `src/heretic/main.py`). Must work after a normal finish and after a SIGINT budget stop.
2. **Export-only.** Export a chosen recorded trial from an existing study while adding **zero** trials.
   Known hazard in the current code: `scripts/runtime/heretic_study_tool.py update-settings` sets
   `n_trials = max(requested, recorded)`; with `checkpoint_action=continue` Heretic then runs the
   difference, so the current "export a different trial" path is not export-only.

The contract is the acceptance criteria below. *How* to meet it (which settings, study state or
engine entry point) is for you to determine from the engine source and verify by running it.
Do not assume a mechanism before you have checked it.

Acceptance criteria (real engine, tiny model):
- export-only adds zero trials (trial count and journal trial set identical before/after);
- the exported weights correspond to the requested **study trial identifier** (Optuna `number` and
  Heretic `index` user attr), confirmed from Heretic's "Restoring model from trial N" output and the
  recorded parameters;
- the same holds for a budget-interrupted (unfinished) study;
- if either behavior cannot be achieved without modifying Heretic, stop and report the blocker
  instead of proceeding with Steps 2–3.

## Step 1 — Promotion semantics and the decision table

Problem (verified in code): `evaluation/qualification.py::evaluate_gates` returns `passed: None` for
unmeasured metrics and `stages/stage07_select.py` promotes when nothing *failed*; with
`qualification: off` or reproduction mode the master is promoted normally with only a warning.

Design first, in `docs/evaluation.md`, a **promotion decision table** covering every combination of:
- gate kind: required / optional;
- gate outcome: passed / failed / unmeasured / disabled;
- profile: `selection.profile: qualified | research` (default `qualified`);
- override: `selection.allow_ungated_promotion` (per gate or global — decide and document);
- candidate origin: policy selection / manual selection (Step 2);
- run mode: optimize / exact reproduction / approximate reproduction / quantize-only.

Each row yields exactly one outcome — `promoted:qualified`, `promoted:research`, `promoted:ungated`
or `not_promoted` — plus one machine-readable reason. Fixed rules (implement these, do not re-decide):
- a `null` threshold on a **required** gate is a **config error** (new error id, rejected at config
  validation before any work). Only optional gates can be `disabled`, and only by a `null` threshold;
- `research` accepts *missing* evidence only, never a *measured failure*: a measured failure needs the
  explicit override and yields `promoted:ungated`;
- optional gates never change the outcome; their failures and gaps are reported as warnings.

Decide and document: the required gate set per run mode, and whether the override is per gate or global.

**Aggregation across gates.** A real candidate has several gates at once (e.g. one passed, one
unmeasured, one failed and overridden). Evaluate every gate first, then combine the required gates
with this precedence. The weakest claim wins:
1. any required gate `failed` without override → `not_promoted`;
2. else any required gate `bypassed_by_override` → `promoted:ungated`;
3. else any required gate `unmeasured` → `promoted:research` under `research`, `not_promoted` under `qualified`;
4. else → `promoted:qualified`.

The reason is the rule that decided the outcome, plus the gate that triggered it. When several
gates trigger the same rule, use the first in a documented canonical gate order. The record lists
**all** gates that triggered any rule, not only the deciding one.

Implement the table as one pure function (inputs → outcome + reason + per-gate statuses). Test it
two ways: a table-driven test that enumerates every single-gate row, and explicit mixed-gate cases.
The mixed cases include at least:
- passed + unmeasured + overridden failure → `ungated`;
- un-overridden failure + unmeasured under `research` → `not_promoted`;
- two failures → the reason uses canonical order;
- optional failure + all required passed → `qualified` with a warning.

Statuses per gate: `passed`, `failed`, `unmeasured`, `disabled`, `bypassed_by_override` (with the
config key that allowed it).

Acceptance criteria:
- no path produces `promoted:qualified` with any required gate unmeasured, failed or bypassed;
- every promotion path stores outcome, reason and per-gate statuses in the assessment record
  (Step 5), `metrics/selection.json` and the report;
- report shows a prominent evidence-status line per artifact and the remaining uncertainty
  (fast qualification is a smoke test; deep evaluation is Project B).

## Step 2 — Constraint-based selection with separate fingerprints

Problem (A100): Heretic's `trial_index` indexes the Pareto front sorted by fewest refusals; Pareto
position 0 (study trial `index` 8, KL 11.8) destroyed the model while position 2 (trial `index` 3,
refusals 0.03, KL 0.0916) qualified. Recovery needed a fork.

Implement inside stage 05 (public stage ID unchanged) three internal phases. Keep **study identity**
separate from **invocation settings**:
- **study identity** (decides which physical study is used): model source, native search config
  *excluding budget*, seed, engine commit, tool environment (Step 3), datasets. The existing
  `adapters/heretic.py::study_hash` already ignores `n_trials` and export choice; keep that property
  and extend it with the tool environment;
- **search invocation** fingerprint: study identity + budget (`n_trials`, time limit). A changed
  budget with the same study identity **continues the existing study** and never creates a new one;
- **selection** fingerprint: study identity + the study's recorded trial set + `selection.policy`,
  optimizer constraints, manual choice;
- **export** fingerprint: selection result (study trial identifier) + export settings.

Record all four in the stage receipt.

Reuse rules:
- a changed selection fingerprint with unchanged study identity and budget re-runs selection and
  export only (zero new trials, per Step 0);
- a changed export fingerprint re-runs export only.

Tests must cover, independently:
- KL threshold change;
- refusal threshold change;
- policy change;
- manual trial change;
- budget increase: this one *does* continue the search. Record it as `continue` (Step 3) and assert
  that the study identity and journal path are unchanged.

Selection:
- filter **completed** trials by `selection.max_optimizer_refusal_rate` / `selection.max_optimizer_kl`;
  rank by `selection.policy` (default `min_refusals_within_kl`; also `min_kl_within_refusals`);
- write `metrics/selection-candidates.json`: every completed trial with study trial identifier, Pareto
  position (informational only), scores, eligibility and the ranking rationale;
- no eligible trial → no export, new error id, closest trials listed.

Manual selection (`--trial-id` new, `--pareto-index` kept for compatibility):
- manual choice **selects** the candidate; it does not bypass promotion requirements (Step 1);
- record the stable study trial identifier; a Pareto index is resolved to a trial identifier once,
  at selection time, and only the identifier is used afterwards;
- an ineligible manual choice is exported only with an explicit, recorded override.

Holdout hygiene: selection uses optimizer metrics only. Do not build multi-candidate qualification
here; if it is added later, it must use a separate final holdout (document this).

Acceptance criteria: replaying the A100 study (journal in `../a100-evidence/` if present; otherwise a
fixture with the same trial identifiers and scores) selects **study trial index 3 with its recorded
scores** — Pareto position 2 is only its position in this fixture — with zero new trials.

## Step 3 — Study history and environment identity

Problem: forks silently continue the parent study; `update-settings` overwrites settings without history.

**Locking.** Run locks do not protect a study: several runs or forks could reference the same
physical study. Two rules apply:
- **Study lock.** Every engine invocation on a study, and every write to its journal, settings or
  history, holds a study-level lock (e.g. `<study dir>/.lock`, same stale-lock semantics as the
  run lock). Always acquire the run lock first, then the study lock. A second holder fails fast
  with a new error id.
- **Forks own independent copies.** A fork copies the parent journal into its own study directory
  (current behavior: `study_copied_from_run`) and never writes to the parent's study. Its history
  begins with a `fork-continue` entry recording the parent study id and the trial count at copy time.

Implement `study-history.jsonl` per study, append-only, written under the study lock:
- write an **intent** entry (invocation id, action, planned settings/budget changes, before-state:
  trial count, settings hash) **before** modifying the study; then a **completion** entry (trials added,
  selection, export target) or an **interruption/failure** entry;
- actions: `new`, `continue` (adds trials), `replay` (same resolved inputs, new study),
  `select`, `export` (zero trials), `fork-continue`;
- on resume, an intent without completion is surfaced and reconciled against the journal's actual state.
- Study identity includes the tool environment fingerprint (torch/CUDA build and transformers version
  from the heretic component record). Continuing a study under a different environment is refused
  (new error id) unless explicitly forked into a new study; document why.

Acceptance criteria:
- tests that kill the process (fake engine) between each of: settings update, trial completion,
  selection, export. Each test verifies the history and the resumed outcome;
- tests for environment change refusal;
- tests showing that a concurrent invocation on the same study is refused, and that a fork never
  modifies the parent's journal or history;
- the history alone reconstructs the decision sequence.

## Step 4 — Model-specific knowledge in versioned profiles

Problem: Qwen3.5 findings sit in global defaults (FP8 ignores `re:.*norm.*`, `re:.*conv1d.*`;
`quantization.gguf.converter_transformers: 5.18.0`).

Implement `configs/profiles/qwen3_5.yaml` with those choices, the `mtp.` graft recommendation and the
mmproj expectation, and a header listing what was exercised (tiny-random/qwen3.8 on CPU/WSL,
Qwen3.5-2B on A100) and what is unverified (27B, B200, FP8_BLOCK on vLLM). Restore neutral global
defaults; reference the profile from `configs/pipeline.yaml` and `examples/qwen-b200.yaml`; include the
profile hash in stage keys and assessment records.

Acceptance criteria: the Qwen pipeline's resolved config is unchanged in effect; a non-Qwen config no
longer inherits Qwen exclusions; tests for both.

## Step 5 — Assessments as separate records; master vs export comparison

Problem (verified in code): `artifacts/manifest.py::ArtifactRegistry.update` overwrites a manifest's
`validation`/`inference`/`evaluation` in place; a reused export's sidecar manifest is rewritten by later
runs, losing earlier provenance.

Assessment records:
- keep the artifact manifest about **bytes and lineage**; add append-only **assessment records**
  (`workspace/assessments/<artifact_id>/<assessment_id>.json` + run copy) with: artifact id, run id,
  suite id + version + hash, runtime backend + version + environment fingerprint, generation contract
  (below), results, outcome per Step 1, timestamp;
- the manifest references assessments by id; any "current status" shown is derived and states which
  assessment it comes from. Define how the schema changes (additive vs version bump) and document it in
  `docs/artifact-schema.md`.

Comparison contract (document it in `docs/evaluation.md`), recorded per run:
- prompt and rendered chat-template text (hash always; stored text unless redacted, see below), thinking mode, max new tokens,
  stop sequences, decoding parameters, context length and truncation behavior;
- per backend, list contract fields it cannot honor (e.g. chat template rendering differences between
  transformers, vLLM and llama.cpp) — these are reported, not hidden.
- **Redaction applies to assessment records.** Today, `logging.redact_prompts` stores hashes instead
  of prompt and response text in qualification records (`stages/stage06_qualify.py`). The same
  setting must govern assessment records and comparison outputs:
  - when it is on, store only hashes (and lengths) of prompts, rendered templates and responses,
    and set `exact_text_replay: unavailable (redacted)`;
  - comparisons still work, because they use hashes and scored results, not stored text;
  - test both settings, and assert that no prompt or response text appears anywhere in the
    workspace when redaction is on.

Comparison:
- run the same small suite (subset of `configs/qualification/benign_prompts.yaml` + a few held-out
  refusal prompts) on the master and on each export;
- record counts with denominators (correct / graded, refused / prompts, integrity issues) and deltas
  vs the baseline; describe results as "exported runtime vs baseline runtime";
- baseline = selected master; for quantize-only runs = the source model (recorded as such); if no
  baseline assessment exists, comparisons are `unmeasured` — never zero deltas or passes;
- export gating stays report-only by default (`selection.export_gates`).

Acceptance criteria:
- mocked integration shows per-export deltas with denominators;
- quantize-only shows the source baseline;
- a missing baseline yields `unmeasured`;
- two runs assessing the same artifact bytes produce two assessment records, and neither
  overwrites the other;
- with redaction on, assessment records hold hashes only.

## Step 6 — Fixes from the A100 run

1. **Capability policy, not a universal rule.** Preflight reports three distinct categories:
   `unsupported_configuration` (a requested step the configured backend cannot perform — e.g. a
   backend-documented requirement), `estimated_resource_shortage` (estimates from model size, VRAM,
   RAM, disk, loading strategy and offload settings), and `unverified_execution` (allowed but not
   proven on this hardware/backend). Base each rule on the actual backend requirement and cite it; do
   not hard-code compute capability as a compatibility verdict. For the FP8 loader test: on GPUs
   without native FP8 (A100 = 8.0) vLLM uses a weight-only fallback, so record the result as
   `unverified_execution` / not representative of native FP8 rather than a pass/fail of the export.
2. **Tool-environment activation:** run tool processes with the venv `bin/` first on `PATH` and the
   venv's `site-packages/nvidia/*/lib` on `LD_LIBRARY_PATH`; add `ninja` to the vLLM component
   (A100: vLLM JIT failed on `libnvrtc-builtins.so.13.0`, then on missing `ninja`).
3. **Preservation check** (`stages/stage09_verify.py::compare_files`): transformers-5 saves merge
   processor configs into `processor_config.json` (`image_processor`, `video_processor`) and may ship
   only `tokenizer.json`; report `re-serialized` with the equivalence basis, not `missing`.

## Step 7 — B200 acceptance record (prepare only; do not claim results)

You will not have a B200. Prepare:
- `configs/acceptance/b200.yaml` and `docs/acceptance.md`: one bounded run covering clean setup, tool
  reuse on rerun, full-size load and transformation, budget interruption followed by export-only (zero
  new trials), resume after an interrupted stage, decision-table promotion, export loading and
  master/export comparison, measured time, peak memory and disk use;
- `scripts/acceptance_record.py`: turns a finished run into `acceptance-record.json` + markdown tied to
  the git commit, config hash, profile hashes and tool versions, listing every unresolved check;
- a test against a mocked run.

## Out of scope

Observability exporters and the Lambda guest agent (`../future-planning.md`), new quantization formats
or backends, broader qualification suites beyond Step 5's comparison set, multi-candidate
qualification, Project B, automatic publication.

## Final report expected

Per step: what changed, decisions taken (link the doc tables), test commands with actual output,
new/changed error IDs and config keys, schema changes and migration notes, implementation-log
entries, what could not be verified (and why), and exact commands for the B200 acceptance run. If
Step 0 fails, report the blocker and stop before Steps 2–3.
