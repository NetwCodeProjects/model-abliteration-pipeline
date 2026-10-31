# Implementation log

Engineering record of what was inspected, decided, found and fixed, so installers and adapters can be
refined. Newest entries are appended at the end of each section.

## 2026-10-26 — initial implementation (v1.0.0)

### Plan (as executed)

1. Inspect upstream sources at their current heads (Heretic, llama.cpp, llm-compressor,
   compressed-tensors, vLLM, Hugging Face repos) before writing adapters.
2. Milestone 1: config spec + schema, CLI, orchestrator, receipts, manifest schema, mock fixtures.
3. Milestone 2: host probe/profiles, components, HF pinning and verified downloads, baselines, dry-run.
4. Milestone 3: Heretic adapter (stages 04–07), qualification, gates.
5. Milestone 4: FP8 / GGUF exporters, verification, reports.
6. Milestone 5: negative paths, resume/crash scenarios, docs, real-tool verification on CPU (WSL2).

### Upstream facts verified (with evidence)

| Fact | Evidence | Consequence |
|---|---|---|
| Heretic latest release v1.4.0 (`6ea3b8d`) has no `modifiers/` package, no ARA, no headless settings | `git ls-tree v1.4.0 src`, `git log -S model_action` → `0146b27` (2026-06-27) | Pin master `171bb98b` (2.0.0.dev0); documented as unreleased |
| ARA merged in `662e4ba` (2026-10-01, PR #446); v1.2.0 contains no ARA code | `git grep` on v1.2.0 | Reference card's "v1.2.0 with ARA" ⇒ unreleased build; exact reproduction impossible |
| `coder3101/Qwen3.8-27B-heretic-ara@7904b4f2` has no `reproduce/reproduce.json` | HF API file list | `E_EXACT_REPRODUCTION_UNAVAILABLE` unless `allow_approximate` |
| Heretic merged export drops 15 `mtp.*` tensors of Qwen3.8-27B | index.json comparison of source vs reference (1199 vs 1184 tensors) | Tensor diff recorded; optional verbatim graft |
| Heretic reads `config.toml` from cwd; headless keys as in upstream e2e tests | `config.py`, `tests/*/config.toml` | Run-private cwd per study |
| Continued studies restore all non-excluded settings (incl. `n_trials`, `trial_index`) | `main.py` L405–408 | Study settings updated via Optuna before `continue` |
| In `--reproduce` mode, settings come from the file; if `trial_index` is unset the save loop never terminates | `main.py` trial/action loops | Patch `trial_index=0`, `model_action=save`, clear `upload_*` |
| Heretic exits 0 after "Configuration contains N errors" | `run()` returns normally | Outputs validated; exit code never trusted |
| SIGINT → `study.stop()` → export of configured trial | `objective_wrapper` | Budget = SIGINT, then SIGTERM/SIGKILL after grace |
| Heretic's `uv.lock` pins CPU torch wheels (explicit index) | `pyproject.toml` `[tool.uv.sources]` | Export lock as constraints minus torch family; install torch at lock version with `--torch-backend` |
| llama.cpp b11889 registers `Qwen3_5ForConditionalGeneration` for text (QWEN35) and mmproj (qwen3vl) | `conversion/qwen.py`, `conversion/qwen3vl.py` | GGUF path + mmproj for the target |
| `llama-quantize` aborts without imatrix for IQ1/IQ2/IQ3_XXS/Q2_K_S | `src/llama-quant.cpp: tensor_requires_imatrix` | Fail those formats up front |
| llm-compressor 0.14.0.1 `model_free_ptq` copies non-safetensors files, works without a model class; pins transformers 5.15–5.17 | entrypoint source, `setup.py` | Data-free FP8; separate venv from Heretic (needs transformers ~5.18) |
| Official Qwen3.8 FP8: `quant_method=fp8`, e4m3, dynamic activations, `weight_block_size [128,128]`, vision excluded | `config.json` of `Qwen/Qwen3.8-27B-FP8@017b9c7a` | Default `fp8_block` + ignore list; not claimed byte-identical |
| vLLM 0.32.0 registers `Qwen3_5ForConditionalGeneration`; pins torch 2.13.0 | `registry.py`, `requirements/cuda.txt` | Separate vLLM venv without `--torch-backend` |
| Held-out prompts: `harmful_behaviors` train=416/test=104 rows; Heretic uses train[:400], test[:100] | datasets-server sizes | Qualification uses train[400:416] + test[100:104] |

### Issues found and fixed during development

1. **YAML 1.1 booleans** — `qualification: off` became `false` and the spec's own key `execution.yes`
   became `True`. Fix: YAML 1.2-style SafeLoader (only true/false are booleans) + explicit error hint.
2. **Fork re-ran Heretic** after adding a quantization format: stage 01's key changed and cascaded.
   Fix: dependencies on stage 01 are keyed by the fingerprints of the components a stage uses.
3. **Introspection output dir missing** in stage 04 (would also fail on the real host). Fix: mkdir.
4. **Preflight reuse check** conflicted with post-execution validation. Fix: `always_rerun` flag.
5. **`main.sh` python selection**: `sys.exit()` ran before `import yaml` in the probe, so a Python
   without PyYAML was chosen. Fix: import first; inspection commands search for a Python with PyYAML.
6. **Overwriting finalized exports on fork**: Fix: refuse to overwrite; reuse an existing export when
   parent artifact and recipe ID match; `.partial` always cleaned on failure.
7. **CRLF risk** on Windows checkouts for `main.sh`. Fix: `.gitattributes` forcing LF; exec bit in index.
8. **Real Heretic: journal file name overflow** — Heretic names the Optuna journal after the
   sanitized `settings.model`; an absolute HF-cache snapshot path produced
   `OSError: [Errno 22] Invalid argument` (Windows) and can exceed NAME_MAX on Linux for deep paths.
   Fix: Heretic sees `model` → symlink (NTFS junction on Windows) in its private cwd.
9. **Real Heretic on Windows**: `NoConsoleScreenBufferError` — questionary prompts are constructed
   eagerly even when headless values are set, and prompt_toolkit needs a console. Not fixable without
   modifying Heretic; documented: headless runs require Linux (target platform).
10. **GPU smoke guard on CPU experiments**: stage 01 correctly rejected a CPU-only torch wheel
    ("CUDA is not available"). Kept for GPU hosts; skipped only with explicit `torch_backend=cpu`.
11. **Real llama-quantize validation**: `--help` prints "usage" first, beyond the 60-line tail kept in
    memory. Fix: validators search the complete captured stdout/stderr logs.
12. **Torch version drift**: `--torch-backend` alone installs the newest torch. Fix: torch/torchvision
    pinned to the versions in Heretic's lock (2.14.1 / 0.29.1 at the pin), CUDA variant from the backend.
13. **Unit test network access**: a reinstall test triggered a real `git clone`. Fix: test uses a
    component whose installer fails offline first.

14. **Real llm-compressor on Qwen3.5** (`tiny-random/qwen3.8`): `model_free_ptq` aborted on
    `linear_attn.conv1d.weight` (shape (1280,1,4)) and `mtp.pre_fc_norm_embedding.weight` (1-D) — only
    names ending in `norm` are skipped automatically. Fix: default ignore adds `re:.*norm.*`,
    `re:.*conv1d.*` (as in upstream's `examples/model_free_ptq/qwen3.5_int8.py`).
15. **Real llama.cpp converter on Qwen3.5**: `ValueError: Tokenizer class Qwen3_5Tokenizer does not exist`
    with llama.cpp's pinned transformers 4.57.6. Fix: converter venv installs transformers 5.18.0
    (`quantization.gguf.converter_transformers`, upstream README recommends upgrading for new models).
16. **Thinking models in smoke tests**: real Qwen3.5-0.8B (your `E:\LLM-models` GGUF, IQ4_XS made by the
    adapter's own argv) answered `<think> Thinking Process: …` without closing the block in 24 tokens;
    the answer-oriented integrity check reported `empty_visible_output`. Fix: loader smoke tests judge
    reasoning text too (`include_thinking=True`); qualification still requires a visible answer.
17. **Toolchain note (WSL2 Ubuntu 24.04)**: host has CUDA 12.0 `nvcc` with GCC 13, an unsupported
    pairing for CUDA 12.0, so the WSL verification used `build_cuda: off` (not attempted with CUDA).

18. **FP8_BLOCK on tiny dimensions**: on `tiny-random/qwen3.8` (hidden 16/64) llm-compressor 0.14.0.1
    `FP8_BLOCK` failed with "non contiguous tensor `linear_attn.in_proj_qkv.weight`", while `FP8_DYNAMIC`
    succeeded on the same master. The 27B model's dimensions are multiples of 128 (hidden 5120, GDN qkv
    10240), so FP8_BLOCK is expected to work there — **unverified until the B200 run**.
19. **Heretic re-serializes tokenizer/generation configs**: on the real run the master's
    `tokenizer_config.json` and `generation_config.json` differ from the source (tokenizer.json is
    JSON-equivalent, chat template identical). Stage 09 reports this as a warning; not altered.

### Real-tool verification (no GPU claims)

Host: Windows 11 + WSL2 Ubuntu 24.04.4 (kernel 6.18, 8 cores, 15 GB RAM, GTX 1070 Ti visible but
unused: `torch_backend: cpu`, `build_cuda: off`). No B200 was available; nothing below is a GPU result.

| What | Result |
|---|---|
| `./main.sh --help`, `--dry-run` on Linux before bootstrap (system PyYAML) | OK; profile `ubuntu` detected ("CUDA alone does not imply Lambda Stack"); no files written |
| Bootstrap via `python3 -m venv` + `pip --require-hashes -r requirements/control.txt` (no uv on host) | OK |
| Stage 01 real installs: Heretic `171bb98b` (torch 2.14.1+cpu from lock pins, transformers 5.18.0), llm-compressor 0.14.0.1, llama.cpp b11889 CPU build + converter venv | OK after fixes 11, 15; validation incl. `heretic --help`, `import heretic.modifiers.ara` |
| Resume of the failed run (`--resume … --yes`) | fp8 + heretic components **reused** (validated), only llama.cpp rebuilt; without `--yes` → exit 15 as designed |
| Stages 02–07 with real Heretic ARA (3 trials, tiny template) on `tiny-random/qwen3.8@346bb76e` (`Qwen3_5ForConditionalGeneration`) | OK: study journal via `model` symlink, Pareto export, trial 1 exported, real transformers reload + qualification, BF16 master, 15 `mtp.*` tensors grafted back |
| Stage 08 real exports | GGUF q8_0 + iq4_xs with mmproj (after fixes 15, 20); FP8_DYNAMIC (FP8_BLOCK see 18) |
| Stage 09 real loader tests | llama-completion + llama-mtmd-cli image test passed for both GGUFs (`mentions_red: true`); FP8 left `inference_passed: null` (vLLM not run) |
| Final run `20261026T091348Z-4bb489` | all stages succeeded, exit 0 |
| Adapter argv on a real Qwen3.5-0.8B BF16 GGUF (`E:\LLM-models`) | IQ4_XS in 50 s (492 MiB), metadata intact, completion + mtmd ran (found fix 16) |
| Windows-native real Heretic | stages 00–04 OK; stage 05 blocked by `NoConsoleScreenBufferError` (issue 9) |

20. **Real converter `--mmproj` naming**: b11889 writes exactly the `--outfile` name (no `mmproj-`
    prefix), so the projector was treated as a broken model GGUF and the image test never ran. Fix:
    explicit `mmproj-…` name and projector detection by GGUF metadata (`general.architecture == clip`).

### Still to verify on the B200 (`tests/gpu`, docs/deployment-lambda.md)

CUDA torch for sm_100 via `--torch-backend auto`; Heretic ARA on Qwen3.8-27B (memory, time per trial,
budget stop + export); held-out KL with two 27B models loaded; FP8_BLOCK on 27B + vLLM 0.32.0 load;
llama.cpp CUDA build with `CMAKE_CUDA_ARCHITECTURES=100`; 27B GGUF conversion size/time; A4000 fit.

## 2026-10-29 — Step 0 feasibility (next-changes specification)

Real pinned Heretic, tiny model, CPU/WSL2; details and evidence in [step0-feasibility.md](step0-feasibility.md),
reproducible with `tests/realtools/step0/run_all.sh`. Search-only (`model_action = ""`) and
export-only (stored `n_trials = len(study.trials)`, `checkpoint_action = continue`) both add zero
trials, including after a SIGINT budget stop and after a SIGKILL. Export-only produced the same
bytes as a direct export. The current fork-export path adds trials to an interrupted study: 6 were added
in check C-hazard. Only Pareto-front trials can be exported. Exporting sets `finished = true` and
overwrites the stored budget. No pipeline code was changed.

## 2026-10-29 — Step 1: promotion decision table

- `evaluation/promotion.py::decide`: a pure function. Required gates are combined so that the weakest claim
  wins: failed, then bypassed_by_override, then unmeasured (research or not_promoted), then qualified.
  Ties use the canonical gate order. Outcomes: `promoted:qualified`, `promoted:research`,
  `promoted:ungated`, `not_promoted`.
- `evaluate_gates` now reports all six gates. It used to drop the independent gates when there was no
  qualification, so `qualification: off` could promote with only a warning.
- New config keys: `selection.profile`, `optional_gates`, `override_gates`.
  `allow_ungated_promotion` is kept as the "every gate" shorthand.
- New error IDs: `E_GATE_REQUIREMENT_MISSING` (exit 3), `E_GATES_UNMEASURED` (exit 9).
- **Behavior change:** reproduction runs and `qualification: off` no longer promote under the default
  `qualified` profile. They need `profile: research` (labelled), or optional optimizer gates for
  reproduction. A null threshold on a required gate is now a config error, where it used to disable
  the gate silently.
- The report shows an evidence-status line for each artifact. The master manifest carries
  `extensions.promotion` and `extensions.evidence_status`; both are additive.

## 2026-10-29 — Step 4: model profile for Qwen3.5

- `configs/profiles/qwen3_5.yaml` now holds the Qwen3.5 FP8 exclusions (`in_proj_a/b/ba`, `mtp.fc`, `*norm*`,
  `conv1d`) and the converter's `transformers==5.18.0`. Its header records the `mtp.` graft recommendation,
  the mmproj expectation, and what was exercised or is unverified.
- Built-in defaults are neutral: the FP8 ignore list keeps only `lm_head`, embeddings, the vision tower
  and router gates, and `converter_transformers` is `null` (llama.cpp's own pin).
- `configs/pipeline.yaml`, `examples/qwen-b200.yaml` and `examples/ubuntu-quantize-only.yaml` reference
  the profile. Tests check that their effective settings are unchanged.
- Each profile's path and sha256 are stored in `resolved-config.json`, kept across resume, and shown in
  the report. **Deviation from the specification:** the profile file hash is *not* part of stage keys.
  Stage keys already depend on the effective merged values, and keying on the file would invalidate a
  finished 27B study after a comment-only edit. Assessment records (Step 5) will carry the profile hashes.

## 2026-10-29 — Step 6: fixes from the A100 run

- **Capability policy** (`environment/capabilities.py`, stage 00). Findings fall into three categories:
  `unsupported_configuration`, `estimated_resource_shortage` and `unverified_execution`. Each finding
  cites its basis. The existing VRAM and disk warnings are now findings. New error id:
  `E_CAPABILITY_UNSUPPORTED` (exit 4).
- **FP8 on GPUs without native FP8.** Stage 09 still runs the vLLM loader test there, but records it with
  `representative: false` and leaves `inference_passed` null. A failure does not make the run partial.
- **Tool-environment activation.** `Run.tool_env(record)` and `ComponentRecord.activate` put the venv
  `bin/` and the `site-packages/nvidia/*/lib` folders first on the search paths for every tool process
  (stages 04, 05, 06, 08 and 09). No package was added: the A100 vLLM install log shows that
  `ninja==1.13.2` was already in the venv, just not on `PATH`.
- **Preservation check.** Evidence: the small tokenizer and processor files of `Qwen/Qwen3.5-2B@15852e8c`
  were loaded and re-saved with transformers 5.18.0 (Heretic venv, WSL). The result matched the A100
  observation exactly:
  - `preprocessor_config.json` and `video_preprocessor_config.json` are merged into
    `processor_config.json`. Every setting is preserved; `processor_class` moved and the
    `…ImageProcessorFast` name lost its suffix.
  - `vocab.json` and `merges.txt` are embedded unchanged in `tokenizer.json`.
  - `tokenizer.json` gains 7 audio/TTS added tokens, and its decoder and pre-tokenizer flags change.

  Encode and decode were identical on 8 probe strings, but the added tokens change tokenization of those
  strings, so `tokenizer.json` is still reported as `differs`, now with a summary. The first two cases
  are `re-serialized` with their basis. The real processor files are kept as test fixtures in
  `tests/fixtures/transformers5_qwen35/`.

## 2026-10-30 — Steps 2–3: constraint-based selection, export-only and study history

Design: [study-lifecycle.md](study-lifecycle.md) (committed before the code). Stage 05 now runs
search-only → selection → export-only under a study lock, and records every step in
`study-history.jsonl`.

- **Selection** (`evaluation/selection.py`, pure). Completed trials meeting the optimizer constraints
  are ranked by `selection.policy`; `metrics/selection-candidates.json` lists every trial. `--trial-id`
  is new; `--pareto-index` is resolved to a trial identifier once. `abliteration.pareto_trial_index`
  now defaults to `null`. New error ids: `E_NO_ELIGIBLE_TRIAL`, `E_MANUAL_TRIAL_INELIGIBLE`,
  `E_TRIAL_NOT_FOUND`, `E_TRIAL_NOT_EXPORTABLE`, `E_TRIAL_CHOICE_CONFLICT`.
- **Export-only** per the Step 0 constraints. `heretic_study_tool.py set-settings` replaces
  `update-settings` (the C-hazard). The export is verified by the trial-set digest and the restore
  line: `E_EXPORT_ADDED_TRIALS`, `E_EXPORT_TRIAL_MISMATCH`. A crash during the search is now
  `E_HERETIC_SEARCH_FAILED` (it was reported as `E_HERETIC_NO_OUTPUT`).
- **History, locks, environment.** Intents are written before the study is modified and are reconciled
  on resume. A fork copies the journal under the parent's lock. The tool environment is part of the
  study identity, and a change is refused (`E_STUDY_ENVIRONMENT_CHANGED`). A new run whose inputs
  match another study is recorded as `replay`.
- **Behavior change.** With the default thresholds, policy selection avoids the high-KL trial that
  Pareto position 0 used to export. The tests that relied on that trial now use an explicit manual choice.

Findings while implementing:

21. **Budget-derived `n_startup_trials` changed the study identity.** With
    `abliteration.n_startup_trials: null` the value is `max_trials // 3`, so a budget increase created a
    new study instead of continuing. Found by the budget-increase test. Fix: stage 04 records
    `budget_derived_keys` and the identity excludes them. A continued study keeps its stored value anyway.
22. **Stale-lock detection on Windows.** `FileLock._pid_alive` treated any PID that `OpenProcess` could
    open as alive. A terminated process stays openable while a handle to it exists, so a killed
    pipeline's lock blocked the resumed run (exit 12, intermittently). Fix: also require
    `GetExitCodeProcess == STILL_ACTIVE`, with a regression test.
23. **Hard-kill tests on Windows.** A venv's `python.exe` is a launcher that starts the real interpreter
    as a child, so `os.getppid()` in a fake tool is not always the pipeline. The test driver exports its
    PID (`FAKE_PIPELINE_PID`).
24. **The realtools test predated Steps 1 and 4.** Null thresholds now need `optional_gates`, the tiny
    Qwen3.5 model needs the `qwen3_5` profile for FP8, and `fp8_dynamic` is needed (finding 18). With
    random weights, the word-salad check (`d1849b0`) can fail integrity. That makes stage 09's master
    smoke test fail and the run partial (exit 20), which the test now accepts in that exact case only.

Real-tool verification (WSL2, CPU, real Heretic `171bb98b`, `tiny-random/qwen3.8`, seed 1234; workspace
`/home/exo/pa-ws`, runs `20261031T012711Z-57676b`, fork `…012908Z-c98c78`, fork `…013006Z-8fab33`):

| Check | Engine output | History |
|---|---|---|
| search-only, 5 trials | `Running trial 1…5 of 5`, `Restoring model from trial 5`, no save | `replay` (an earlier run had the same inputs), 5 added |
| export-only | `Resuming existing study.`, no `Running trial`, `Restoring model from trial 5`, saved | `export`, before: 5 trials and stored `n_trials` 5; 0 added |
| fork, `--max-trials 8` | `Running trial 6…8 of 8` only, then export-only of trial 5 | `fork-continue`, `continue` (3 added), `select`, `export` (0 added) |
| fork of that fork, `--trial-id 6` | `Restoring model from trial 6`, saved | `export` at Pareto position 1, 0 added |
| real study tool on the A100 journal (copy) | Pareto `[8, 9, 3, 5]` | selection picks trial 3 (refusals 0.03, KL 0.0916) |

The kill-recovery, lock and environment-change tests use the fake engine only.

## 2026-10-31 — `--set KEY=VALUE` (approved deviation)

The change specification allowed no CLI features beyond `--trial-id`. That left fork-time changes of
the selection thresholds and policy reachable only through the programmatic API (the Steps 2–3
limitation). The user approved one generic override flag instead of per-key flags. `--set` is
repeatable; values are YAML scalars or lists, and every key and type passes the same config validation
as the YAML (exit 3). A malformed pair, or a key also set by a dedicated flag or given twice, is a usage
error (`E_SET_INVALID`, exit 2). On `--resume` a changed key needs `--fork`, as with any override. The
fork selection tests now use the real flag instead of patching `cli.to_overrides`.

## 2026-10-31 — Step 5: assessment records, master vs export comparison, redaction

Design first: `docs/assessments.md` (commit `704ed79`), comparison contract in `docs/evaluation.md`,
schema history in `docs/artifact-schema.md`.

- Measurements are append-only assessment records (`workspace/assessments/<artifact_id>/`, exclusive
  create, run copy). Kinds: `promotion` (stage 07), `verification` and `comparison` (stage 09).
  `ArtifactRegistry.update` is gone; `attach` writes a record and re-derives the manifest under a
  per-artifact lock. Manifest schema 1.1 (additive): `assessments`, `derived_status`.
- Stage 09 compares the baseline (master; source or input path for quantize-only) with every export
  on one suite whose prompts are rendered once by the baseline tokenizer. Counts with denominators,
  deltas, export gates (`selection.export_gates`, report by default), `unmeasured` without a baseline.
- `--set` (approved earlier today) lets a fork change export gates; they are excluded from stage
  07's key, so only stages 09–10 re-run.

Findings:

25. **The GGUF loader smoke test applies no chat template** (`llama-completion -p … -no-cnv`), while the
    vLLM smoke test uses `llm.chat` with the model's template and its default thinking mode, and
    transformers forces `enable_thinking=false`. The three loader tests never saw the same input. The
    comparison therefore renders each prompt once (baseline tokenizer) and passes the text unchanged
    to vLLM (`LLM.generate`) and llama.cpp (`-f`). The loader smoke tests are unchanged and their
    records say which template each one used.
26. **llama.cpp prompt handling at `d854d98`** (source, then a real run): `-f` drops one trailing
    newline, so the file holds the rendered text plus one newline. Escape processing is on by
    default and also applies to `-f`, so `--no-escape` is needed. `--no-display-prompt` keeps the
    prompt out of stdout, which then ends with ` [end of text]` after an end-of-generation token.
    Real check (WSL, CPU, Qwen3.5-0.8B BF16 GGUF from `/mnt/e/LLM-models`, prompt rendered by the
    tiny Qwen3.5 tokenizer with `enable_thinking=False` and containing a literal `a\nb`): exit 0,
    stdout `Paris [end of text]`, prompt not echoed, the 33 prompt token ids (`--verbose-prompt`)
    identical to the HF tokenizer's ids, including `<think>`, `</think>` and the trailing `\n\n`.
27. **Redaction covered only `qualification.json`.** The spec and raw files of every generation step,
    the generating tools' stdout logs (llama.cpp echoes the prompt) and stage 09's smoke `samples`
    held prompt and response text in the workspace. All generation I/O now goes through
    `StageContext.generation_dir`, a temporary directory outside the workspace when redaction is on.
    The integration test scans the whole workspace for prompt and response text.
28. **Stage 07's key covered all of `selection`**, so any new selection key would re-promote the
    master. Its key now lists the selection keys explicitly, without `export_gates`. Existing runs
    see a changed stage 07 key once (re-promotion on resume).
29. **Quantize-only runs now provision the `heretic` environment** when comparison is enabled: it is
    the transformers runtime that measures the source baseline. `evaluation.comparison.enabled: false`
    avoids it.

Verification: unit tests (`tests/unit/test_assessments.py`) and fake-runtime integration tests
(`tests/integration/test_assessment_pipeline.py`) for every acceptance criterion. Real tools: only the
llama.cpp invocation (finding 26). The vLLM path and a full stage 09 comparison have not run with real
tools (no GPU here).


## 2026-10-31 — Renamed to llm-surgeon

The project and its GitHub repository are now `llm-surgeon` (`https://github.com/NetwCodeProjects/llm-surgeon`;
GitHub redirects the old URL). Distribution `llm-surgeon`, import package `llm_surgeon` (was
`model_pipeline`), console script `llm-surgeon` (was `model-pipeline`), `main.sh` runs
`python -m llm_surgeon`, schema `$id`s and `--version` use the new name. Paths in earlier entries of this
log were updated to the new package directory. Stored data is unaffected: receipts, manifests and
assessment records never contain the package name. `uv.lock` and `requirements/*.txt` were regenerated;
only the project entry changed.
