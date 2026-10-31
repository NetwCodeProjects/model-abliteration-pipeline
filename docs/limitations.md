# Known limitations (v1)

## Verification status

| Path | Verified how | Not verified |
|---|---|---|
| CLI, config, planning, receipts, resume/fork, locks, manifests, partial exports | 120+ CPU unit tests, mocked 00–10 integration tests (fake engines) | — |
| Heretic adapter (install, introspection, native config, headless run, study export) | Real pinned Heretic (ARA) on CPU in WSL2/Ubuntu 24.04 with `tiny-random/qwen3.8` (same `Qwen3_5ForConditionalGeneration` architecture), full 00–10 run, see implementation log | **Qwen3.8-27B on a B200** (memory, runtime, ARA quality) |
| llama.cpp convert/quantize/smoke | Real pinned llama.cpp build + converter on the tiny model in WSL2/Ubuntu 24.04 (see log) | CUDA build for `sm_100`; 27B conversion time/size |
| FP8 (llm-compressor) | Real `model_free_ptq` FP8_DYNAMIC on CPU for the tiny model; FP8_BLOCK fails on its non-128-divisible dims (see log) | vLLM loading on a B200 (vLLM cannot run on the development host) |
| Lambda Stack profile | Unit tests with simulated dpkg/apt markers | A real Lambda Stack image |

Nothing in this repository claims real B200 results. `tests/gpu/` contains the commands to obtain them.

## Functional limits

- **No compatibility resolver** (by design): unsupported architectures fail inside the tools; the
  platform reports their errors.
- **Exact reproduction of `coder3101/Qwen3.8-27B-heretic-ara` is impossible** (no reproduce.json,
  unreleased engine build). Approximate reproduction uses published, 4-decimal-rounded parameters and
  pinned defaults for unpublished settings.
- Heretic's merged export **drops `mtp.*` tensors** (multi-token-prediction head). Recorded in manifests;
  `selection.graft_missing_source_tensors: ['mtp.']` restores them verbatim (they were not abliterated,
  so speculative-decoding acceptance may differ slightly).
- Heretic's merged save re-serializes `tokenizer_config.json`/`generation_config.json` (reported by stage 09).
- Heretic headless mode requires Linux (prompt_toolkit console issue on Windows).
- `trust_remote_code` models cannot run through Heretic non-interactively.
- Only Pareto-optimal trials can be exported headlessly (Heretic's `trial_index` indexes the front);
  a manual `--trial-id` naming a dominated trial is refused. One trial is exported per stage-05 execution;
  choosing another (fork) re-runs export only.
- Selection thresholds and policy can only be changed in a fork through the programmatic API: the CLI has
  no flag for them (the change specification allows no new CLI features beyond `--trial-id`). From the
  command line, a new run with an edited YAML starts a new study and repeats the search.
- A study recorded under another tool environment is not continued (`E_STUDY_ENVIRONMENT_CHANGED`); after
  re-provisioning, `--new-run` repeats the search.
- Budget enforcement is wall-clock + SIGINT; a trial in progress at the deadline is discarded by Heretic.
- `baseline_mode: local` and calibrated static FP8 are not implemented.
- FP8 smoke tests exercise text only; image inputs are verified only for GGUF (+mmproj) with a synthetic
  image and a "did it run" criterion (content correctness is recorded, not gated).
- Qualification is a smoke test (≈30 prompts, keyword refusal heuristic) — not a benchmark.
- GGUF fit on a 16 GB RTX A4000 is not guaranteed; tested context is recorded, not promised.
- No automatic reboot, driver install only on the ubuntu profile and only when explicitly allowed.
- Single host, single run per run directory; no distributed queue, HTTP API or uploads (non-goals).
