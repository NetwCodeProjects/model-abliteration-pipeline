# Architecture

## Components

```text
main.sh ──► model_pipeline.cli ──► config (spec, precedence, validation) ──► orchestrator
                                                                             │
          ┌──────────────────────────────────────────────────────────────────┤
          ▼                                                                  ▼
   stages 00..10 (StageContext, prerequisites, execute,             services (injectable)
   validate_outputs, receipts)                                      HostProbe · HubClient · CommandRunner
          │                                                         ResourceSampler · confirm
          ▼
   adapters ── heretic (native TOML, reproduce.json, outcome parsing, study export)
            ── hf_hub (ref normalization, pinning, verified snapshots)
            ── fp8 (llm-compressor recipes) ── llama_cpp (verified argv builders) ── toml_io
   environment ── detect (read-only probe, profile detection) ── lambda_stack / ubuntu profiles
               ── components (pinned tool venvs + llama.cpp build, fingerprint + validation)
   artifacts ── manifest (schema 1.0, registry) ── safetensors_meta ── gguf_meta ── schema_lite
   evaluation ── baselines (published metrics) ── qualification (scoring, gates)
   scripts/runtime/*.py  ── executed *inside* tool venvs (env_check, heretic_introspect,
                            heretic_study_tool, qualify, fp8_quantize, vllm_smoke)
```

| Module | Responsibility |
|---|---|
| `cli.py` | argparse contract, mutual exclusivity, CLI → dotted config overrides; stdlib-only import so `--help` always works |
| `configspec.py` | Single declarative spec → defaults, strict validation, `schemas/pipeline.schema.json`, operational-key list |
| `config.py` | YAML 1.2-style loading, profile merging, precedence, cross-field validation, hashes, diffs |
| `orchestrator.py` | Run creation/auto-resume/fork, locking, signals, dependency + reuse decisions, status, failure reports, cleanup |
| `planning.py` | Component requirements, disk estimates, dry-run report (pure reads) |
| `stages/` | One module per public stage ID |
| `subprocesses.py` | Bounded, logged, redacted, signal-aware execution without shells |
| `telemetry.py` | JSONL + text logs, secret redaction, resource sampling |

## Run lifecycle

1. **Resolve config.** New run: defaults ⊕ profiles ⊕ YAML ⊕ CLI. Resume: the stored
   `runs/<id>/configs/resolved-config.json` ⊕ CLI overrides; any non-operational change is rejected
   unless `--fork`.
2. **Dry run** stops here and prints the plan.
3. **Confirm** (compatibility notice + planned downloads/installs) unless `--yes`.
4. **Lock** the run directory (`.lock`, stale-PID aware); install SIGINT/SIGTERM handlers.
5. **For each selected stage:** applicability → dependency receipts → `stage_key` → reuse if the
   receipt succeeded, the key matches and `validate_outputs` passes → otherwise execute and write an
   atomic receipt (`succeeded | failed | skipped | blocked | interrupted`).
6. **Finish:** status file, summary, and a best-effort incomplete report on failure. Exit codes are stable.

## Data flow and lineage

```text
HF repo@sha ──(02 verify)──► source [art-…] ──(05 Heretic)──► candidate ──(06 qualify, 07 gates)──► master [art-…, parent=source]
                                    │                                                                  │
                                    └──────────── quantize-only ──────────────┐                       │
                                                                               ▼                       ▼
                                                         (08) fp8 / gguf:<type> [art-…, parent=master or source]
                                                         (09) verified + loader-tested, manifests updated
```

Artifact IDs are content-addressed (`art-` + first 24 hex of the sha256 over the sorted
`(path, size, sha256)` inventory), so identical bytes produced by different runs share an ID.

## Stage keys and invalidation

`stage_key = sha256(stage id, config subset declared by the stage, extra inputs such as template/baseline
file hashes, dependency keys)`. Changing a key invalidates that stage and, through the chained keys, all
descendants. A dependency on stage 01 is keyed by the fingerprints of *only the components the stage
uses* (adding a GGUF format does not invalidate abliteration). Operational keys (budgets, `--yes`,
logging, …) are excluded from all fingerprints.

## Design trade-offs

- **No compatibility resolver.** The platform never classifies architectures or keeps an allowlist; it
  runs the pinned tools and reports their real errors (with logs) clearly. Structural checks only
  (config.json + safetensors present, hashes, headers).
- **Adapters, not forks.** Heretic, llama.cpp and llm-compressor are used unmodified; all integration
  goes through their documented CLIs, configs and public Python APIs at pinned revisions.
- **Separate tool environments.** Heretic master needs transformers ~5.18, llm-compressor 0.14 pins
  ≤5.17, vLLM pins torch 2.13: one shared env is impossible, so each tool gets its own venv.
- **Control plane is stdlib + PyYAML.** `--help`/`--dry-run` work before anything is installed.
- **Receipts over directories.** Existence of a directory is never evidence; only validated receipts are.
- **Hardlinked master.** The master shares inodes with the candidate (no extra 55 GB) and is made
  read-only; every later write uses atomic rename to new inodes, so the link can never mutate the master.

## Project B boundary

Project A ends at manifests. Project B consumes `workspace/manifests/*.json` (or
`runs/<run>/reports/manifest-index.json`) and the files they list; it must not import
`model_pipeline`. Deep capability benchmarks (HumanEval+, SWE-bench, …) are out of scope here.
