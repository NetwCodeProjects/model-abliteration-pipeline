# Assessments and master/export comparison

An **artifact manifest** describes bytes and lineage. Everything measured *about* an artifact
afterwards (promotion gates, loader checks, comparisons) is an **assessment record**. Assessment
records are append-only, and they are never rewritten by later runs. The validation, inference and
evaluation status in a manifest is derived from the records, and the manifest names the record each
part comes from.

## Problem this fixes

Before this change, stage 09 rewrote the manifest's `validation`, `inference` and `evaluation` in
place through `ArtifactRegistry.update`. An export reused by a later run (stage 08 finds it by parent
artifact and recipe) got its sidecar manifest rewritten with the later run's results. The earlier
run's results were lost, and nothing recorded which run a status came from.

## Assessment records

Location (written once, with exclusive create; an existing file is never replaced):

- `workspace/assessments/<artifact_id>/<assessment_id>.json`: the store, which is authoritative;
- `runs/<run>/assessments/<assessment_id>.json`: run copy.

`assessment_id` = `asm-` + the first 24 hex digits of the sha256 of the canonical record without its
id. The record includes the run id and creation time, so two runs that assess the same bytes produce
two records.

| Field | Content |
|---|---|
| `schema_version` | `1.0` (`schemas/assessment.schema.json`) |
| `assessment_id`, `artifact_id`, `role`, `run_id`, `stage_id`, `created_utc` | identity |
| `kind` | `promotion` (stage 07), `verification` (stage 09), `comparison` (stage 09) |
| `suite` | `{id, version, sha256}` of the prompt set and scorer that were used; `null` when no prompts were involved |
| `runtime` | `{backend, version, environment_fingerprint}`; the fingerprint is the sha256 of the tool environment's recorded versions and the GPU names |
| `generation_contract` | see [Comparison contract](evaluation.md#comparison-contract); `null` without generation |
| `contract_gaps` | contract fields the backend could not honor, with the reason; reported, never hidden |
| `redaction` | `{redacted, exact_text_replay}`; `exact_text_replay` is `available`, `unavailable (redacted)` or `not applicable` |
| `results` | kind-specific measured values (counts with denominators; no text when redacted) |
| `checks` | `[{check, result, value, threshold}]` with `result` from Step 1: `passed`, `failed`, `unmeasured`, `disabled` |
| `outcome` | for `promotion`, the decision-table outcome (`promoted:qualified`, …). For the other kinds, the weakest claim wins: `failed` if any check failed, else `unmeasured` if any check is unmeasured, else `passed` (disabled checks are ignored) |
| `derives` | the manifest status this record establishes: `{validation, inference, evaluation}`, each an object or `null` (not established by this record) |

## Derived status in the manifest

`validation`, `inference` and `evaluation` keep their schema 1.0 meaning, but their values are
computed. Start from the manifest's current values: the registration defaults (every file hashed, no
loader test, `not-evaluated`), or the values a schema 1.0 manifest already held. Then apply the
`derives` of every record of the artifact in `(created_utc, assessment_id)` order, so the latest
record that measured a part wins. The fold is idempotent.

`derived_status` names the record behind each of three parts (`null` = registration default or a
carried-over 1.0 value):

| Part | Manifest fields | Established by |
|---|---|---|
| `validation` | `validation.integrity_passed` | `verification` (file re-hash) |
| `inference` | `validation.inference_passed`, `validation.modalities_tested`, `inference.*` | `verification` when a representative loader test ran; `promotion` when the candidate was qualified (reload, `text`, transformers) |
| `evaluation` | `evaluation.*` | `promotion` (qualification run id and provenance, or `optimizer-only`); `verification` of an export with a loader test (`integrity-only`) |

A `comparison` record establishes no part. It is listed in `assessments` and shown in the report.
Because the parts are separate, a later run whose loader test did not run (for example
`verify_backend: none`) updates the integrity status but keeps the earlier loader result. The
manifest says which record that result came from.

Manifests are recomputed from the store on every registration and every new record, under a
per-artifact lock (`workspace/assessments/<artifact_id>/.lock`), so a re-registration never drops
references. A manifest copy that is not rewritten (the sidecar of an earlier run's directory with the
same bytes) lists the records known when it was written; the store directory is always complete.

## Schema change: artifact manifest 1.0 → 1.1 (additive)

- `schema_version` may be `1.1`. New manifests are written as `1.1`.
- New optional top-level fields `assessments` and `derived_status`. No field is removed or renamed,
  and every 1.0 field keeps its type and meaning. `extensions.verification` (stage 09 details in the
  manifest) is no longer written; the details are in the verification record and in
  `runs/<run>/metrics/verification.json`.
- **Migration:** a 1.0 manifest is valid under the 1.1 schema. Readers that ignore unknown top-level
  keys need no change. Validators pinned to the 1.0 schema (`additionalProperties: false`) must accept
  `1.1`. Existing 1.0 manifests are upgraded the next time the artifact is registered or assessed.
  Older runs have no records, so their status stays the registration default plus whatever the 1.0
  manifest held until the first new record replaces it.

## Comparison

Stage 09 runs one small suite on a baseline and on every export, all with the same rendered prompts:

- **Suite** `comparison` v1: the first `evaluation.comparison.benign_prompts` (8) `fast` prompts of
  `evaluation.benign_prompts_file`, plus the first `evaluation.comparison.heldout_refusal_prompts` (4)
  held-out refusal prompts (`evaluation.heldout_refusal_prompts`, disjoint from Heretic's splits).
  The scorer is the qualification scorer (`keyword_refusal_heuristic_v1`, substring correctness,
  integrity heuristics). The suite hash covers prompt ids and hashes, dataset revisions and splits,
  limits, expected answers, refusal markers and the scorer id.
- **Baseline:** the selected master, or the source model for quantize-only runs (recorded as
  `baseline.role: source`). The baseline runs with transformers in the `heretic` environment. In
  quantize-only runs with comparison enabled, that environment is therefore provisioned too.
- **One rendering.** The baseline tokenizer's chat template renders every prompt once (thinking mode
  from `evaluation.enable_thinking`). vLLM and llama.cpp receive that rendered text and never apply a
  template themselves, so chat-template differences between backends cannot enter the comparison.
- **Results:** counts with denominators: benign correct / graded, held-out refused / prompts,
  responses with integrity issues / responses. Deltas are export minus baseline, as counts and as
  rate differences, and are described as "exported runtime vs baseline runtime".
- **No baseline, no delta:** if the baseline did not run or did not load, every export comparison
  is `unmeasured` with the reason. The same applies to an export whose backend did not run
  (for example `fp8.verify_backend: none`). Deltas are never reported as zero, and checks never pass
  without a measurement.

### Export gates (`selection.export_gates`)

| Check | Config | Default | Fails when |
|---|---|---|---|
| `export_loaded` | — | — | the export's runtime did not load or produced no output |
| `export_integrity` | `require_integrity` | true | a prompt that was clean on the baseline has integrity issues on the export |
| `benign_correct_drop` | `max_benign_correct_drop` | 0.25 | baseline benign rate − export benign rate exceeds it |
| `heldout_refusal_increase` | `max_heldout_refusal_increase` | 0.25 | export refusal rate − baseline refusal rate exceeds it |

`mode: report` (default) records the outcome and warns. `mode: enforce` counts an export whose
comparison is `failed` or `unmeasured` as a verification failure (exit 11, or partial with
`execution.allow_partial_exports`). The keys live under `selection` but are not part of stage 07's
key, so changing them never re-promotes a master.

## Redaction (`logging.redact_prompts`)

The setting governs qualification records, assessment records, comparison outputs and stage 09
smoke samples. When it is on:

- records store sha256 hashes and lengths of prompts, rendered templates and responses; no text;
  `exact_text_replay: unavailable (redacted)`;
- generation inputs and outputs (spec files, raw generations, prompt files and the generating tools'
  stdout/stderr logs) are written to a temporary directory outside the workspace and deleted when the
  step ends. Receipts record that these files were discarded;
- comparisons still work: they use scored results and hashes, never stored text.

Not covered: the input datasets themselves (cache) and prompts that the operator writes into the
configuration (`quantization.smoke_prompts` in the resolved config). A hard kill can leave the
temporary directory behind in the system temp location.

## Error ids

| Id | Exit | When |
|---|---|---|
| `E_ASSESSMENT_EXISTS` | 1 | a record with the same id already exists (never overwritten; indicates a bug) |
| `E_ASSESSMENT_INVALID` | 1 | a record failed schema validation |

Comparison failures are not errors in report mode. In enforce mode they use the existing
`E_VERIFY_FAILED`.

## Verification

| Behavior | Covered by |
|---|---|
| per-export deltas with denominators; quantize-only source baseline; missing baseline → `unmeasured`; two runs → two records, neither overwritten; redaction leaves no prompt or response text in the workspace | `tests/integration/test_assessment_pipeline.py` (fake runtimes) |
| record ids, exclusive create, derivation order, schema | `tests/unit/test_assessments.py` |
| llama.cpp flags for the rendered prompt (`-f`, `--no-escape`, `--no-display-prompt`, `-no-cnv`) | pinned source (`d854d98`) and one real run on a Qwen3.5-0.8B GGUF: prompt token ids identical to the HF tokenizer's, no prompt echo, ` [end of text]` marker (implementation log) |
| vLLM `LLM.generate` on rendered text | not run with real vLLM here (no GPU); fake only |
