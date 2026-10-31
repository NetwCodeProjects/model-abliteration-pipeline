# Project A — Model-Agnostic Abliteration & Quantization Pipeline
## Implementation handoff and master coding-agent prompt

**Status:** implementation specification, v1.0 draft  
**Initial validation target:** `Qwen/Qwen3.8-27B`, original Hugging Face checkpoint (BF16 where available)  
**Initial infrastructure:** one NVIDIA B200 on Lambda Cloud, either Lambda Stack or fresh Ubuntu  
**Scope:** model acquisition, environment setup, Heretic abliteration, lightweight qualification, BF16 preservation, FP8/GGUF quantization, integrity checks, reports and manifests  
**Out of scope:** public website, hosted uploads, user billing, comprehensive coding benchmark framework (Project B), automatic architecture-compatibility resolver, unconditional publication to Hugging Face

---

# MASTER PROMPT FOR THE CODING AGENT

You are the lead platform engineer and ML infrastructure engineer implementing **Project A**, a robust, reproducible, configurable CLI platform for model abliteration and quantization. Deliver a working repository, tests, documentation, configuration examples, and an execution-ready setup path. **Do not merely propose an architecture**: implement the project incrementally, validate it, and clearly record what you could and could not verify on the current machine.

## 0. Read-first requirements and operating principles

1. First inspect the repository, README, agent instructions (e.g., `AGENTS.md`), and any existing code. Preserve useful work. If beginning in an empty directory, scaffold the specified structure. Write a brief implementation plan before making large changes.
2. Inspect the actual upstream Heretic version, its `--help`, `config.default.toml`, release or Git revision, output format, checkpoint/resume capabilities, CLI invocation, licenses, and any ARA functionality before implementing the adapter. **Do not invent engine arguments, TOML keys, APIs, model support, or ARA implementation details.** The available Heretic implementation may differ from model-card descriptions. If a required ARA feature cannot be verified, mark it unsupported and provide an honest blocker and a working default Heretic pathway; do not pretend standard directional ablation is ARA.
3. Inspect the actual llama.cpp revision, supported quantization formats, Hugging Face-to-GGUF conversion command, multimodal-projector handling, and imatrix requirements before implementing exporters. Do not claim every model or modality is supported.
4. Model-agnostic *interface* does not imply universal compatibility. **Do not build a model compatibility resolver, curated allowlist, architecture classifier, or promise of automatic compatibility.** The user chooses a model and is responsible for confirming compatibility with Heretic and desired quantizers. Perform normal structural, path, config, dependency and runtime error checks only, with clear failures.
5. Never overwrite the original downloaded checkpoint. Abliterate starting from original higher-precision weights when available; save a separate master checkpoint before quantization. Do not confuse Heretic's memory-saving bitsandbytes loading with deliverable deployment quantization.
6. Keep project B (independent quantization/coding capability research) in a separate repository. Project A exports standardized manifests and runs only necessary abliteration-qualification and artifact-loading checks.
7. Use pinned source revisions, repeatable configs, checksums, safe downloads, bounded runs, durable status files, and explicit stage gates. Support reusing software and downloaded models between runs.
8. Prefer noninteractive execution. No surprise shell prompts; no unapproved driver replacement or host reboot. No automatic public upload. All invasive actions are explicit and documented. Do not curl remote scripts directly into a privileged shell without obtaining user approval.
9. Never claim successful GPU/model tests that were not executed. Provide mocks and CPU-level integration tests for CI and clear GPU-specific smoke-test commands for the B200.
10. Store all scripts, schemas, examples, unit tests, and documentation in the repository. Record engineering fixes in `docs/implementation-log.md` so we can refine the installation/orchestration process.

## 1. Product goals and non-goals

### Goals
- Single command entry point: `./main.sh` (or `bash main.sh`).
- Configuration-first, model-neutral flow: model selected in YAML or overridden with `--model-url`.
- Support **Lambda Stack** and **fresh Ubuntu** explicitly; also support safe `auto` environment detection.
- One-time, idempotent dependency provisioning, model/download cache, optional `--setup-only`.
- Pluggable Heretic adapter and independent quantization exporter adapters.
- Download models/datasets and tools as needed; preserve hashes/revisions.
- Published baseline metadata by default, **no mandatory re-running of original or published reference models**.
- Heretic optimization and optional reproduction profile; configurable trials/time budget; stage-wise resumability.
- Lightweight qualification by default, with optional extended profile. Import optimizer metrics without falsely calling them independent validation.
- Preserve immutable selected BF16 master, then produce configured FP8 and GGUF quantizations where supported.
- Emit machine-readable versioned artifact manifests, per-stage receipts, logs, human-readable run report and actionable errors.
- Ability to run `--quantize-only` on an existing checkpoint without abrading or re-running Heretic.

### Explicit non-goals for v1
- No public website, users/tenants, distributed task queue, HTTP API, upload server, billing, or SaaS work.
- No automatic model architecture compatibility checker/resolver.
- No universal support guarantee for arbitrary models, vision encoders, fp8 schemes or quant formats.
- No giant benchmark sweep or coding-capability statistical framework: that is Project B.
- No unconditional fine-tuning, training, merging, or automatic publication.
- No silent conversion of unsupported modes into something else.

## 2. Platform behavior, entry point, and CLI contract

Provide a real, tested parser (Python can own parsing behind `main.sh`) and `--help`. The following interface is the target. You may add carefully documented flags, but preserve these:

```bash
./main.sh
./main.sh --model-url Qwen/Qwen3.8-27B --lambda-stack
./main.sh --model-url 'https://huggingface.co/Qwen/Qwen3.8-27B' --ubuntu
./main.sh --config configs/pipeline.yaml --dry-run
./main.sh --setup-only --lambda-stack
./main.sh --through 2
./main.sh --from 5 --through 6 --resume RUN_ID
./main.sh --resume RUN_ID
./main.sh --model-url org/model --quantize-only --quantizations fp8,iq4_xs,iq3_s
./main.sh --model-path /absolute/path/to/safetensors-model --quantize-only
./main.sh --experiment reproduce --qualification fast
./main.sh --experiment optimize --max-trials 50 --max-runtime-minutes 120
./main.sh --yes
```

Required behavior:
- `--model-url` accepts a Hugging Face repository ID or a *canonical* Hugging Face model root URL. Normalize safely; reject non-Hugging-Face arbitrary URLs for now. No shell interpretation of URL text. Optionally accept separate `--revision` (commit/tag); resolve and **record immutable commit SHA**.
- `--model-path` accepts a local supported model directory, mutually exclusive with `--model-url`; validate existence without model architecture classification.
- `--lambda-stack` and `--ubuntu` are mutually exclusive. Both override `environment.profile`; do not install Lambda Stack just because the flag was chosen. `--lambda-stack` means use/reuse already-provisioned Lambda Stack tools; `--ubuntu` means bring up dependencies on standard Ubuntu, with guarded host-level steps.
- `--env auto|lambda-stack|ubuntu` may supplement the two shortcuts. Auto mode detects prerequisites defensively; never mistake 'CUDA exists' for 'Lambda Stack exists'. If detection is ambiguous, use a documented safe fallback or fail with an explicit selection message.
- Config precedence for a *new run*: CLI > YAML > selected model/engine profile (when explicitly configured) > built-in defaults. Write `resolved-config.yaml` and its hash. **On resume**, reuse the stored resolved config; reject incompatible overrides or fork a new run with invalidated descendants.
- `--from`, `--through`, `--only`, `--resume`, `--dry-run`, `--setup-only`, `--quantize-only`, `--quantizations`, `--experiment`, `--qualification`, `--max-trials`, `--max-runtime-minutes`, `--yes`, `--output-dir` all documented and tested.
- `--dry-run` MUST perform no downloads, installs, GPU work, model loading, filesystem mutations beyond explicitly documented negligible diagnostic output (ideally none), or network access. Print effective stage plan, intended requirements, target outputs, and estimated disk requirement when estimable.
- Treat resume after partial failure safely. Do not skip a stage because a directory exists; require matching config fingerprint and validated stage receipt.
- Stage ranges require dependency receipts. `--from 8` must not accidentally re-abliterate; `--quantize-only` must take an explicit selected input artifact or model path and record provenance as supplied/unknown where necessary.
- Explicit nonzero exit statuses, stable stage IDs, and predictable error codes/messages.

## 3. Example user configuration

Implement `configs/pipeline.example.yaml` and an opinionated `configs/pipeline.yaml`, plus a formal JSON Schema or equivalent validated typed schema. Example semantics:

```yaml
schema_version: '1.0'
model:
  url: 'Qwen/Qwen3.8-27B'
  revision: null                 # resolve commit SHA and pin
  local_path: null
  trust_remote_code: false      # opt-in only with warning
  hf_token_env: 'HF_TOKEN'      # never store value in logs

environment:
  profile: auto                 # auto | lambda-stack | ubuntu
  install_missing: true
  allow_driver_changes: false
  allow_reboot: false
  python_manager: uv
  gpu_device: auto
  cache_root: './workspace/cache'

abliteration:
  enabled: true
  engine: heretic
  experiment: reproduce        # reproduce | optimize; see behavior below
  engine_revision: null        # resolve a documented pinned release/commit
  heretic_config: 'configs/heretic/reproduce.toml'
  max_trials: 50
  max_runtime_minutes: 180
  save_intermediate: true

evaluation:
  baseline_mode: published_only  # published_only | local
  published_baselines: 'configs/baselines/qwen3.8-27b.yaml'
  qualification: fast           # off | fast | extended
  independent_smoke_prompts: 30
  extended_kl: false
  extended_correctness: false

quantization:
  enabled: true
  input: selected_master
  formats: ['fp8', 'iq4_xs', 'iq3_s']
  fp8:
    scheme: auto               # must be resolved to an actual supported recipe
    calibration_manifest: null
  gguf:
    llama_cpp_revision: null
    imatrix_dataset: null
    preserve_multimodal: true  # if unsupported, fail/mark incomplete; don't claim preserved

execution:
  resume: true
  stop_on_failure: true
  allow_partial_exports: true
  yes: false
  dry_run: false

storage:
  root: './workspace'
  retain_intermediate: true
  artifact_checksums: sha256
```

**Do not blindly hand these keys to Heretic**. This YAML is *our orchestration contract*. Translate supported fields into inspected native CLI/TOML settings; fail on unsupported requested functionality. Be strict about option validation and path handling. Do not conflate `reproduce` with literal reproduction if the published method, exact data, engine revision, or recipe is missing: label outputs as `approximate_reproduction` or block exact reproduction and explain why.

## 4. Stage graph and acceptance gates

Stages 00 through 10 remain stable public contracts:

| ID | Stage | Deliverable / pass gate |
|---|---|---|
| 00 | `preflight` | OS/driver/GPU/CPU/RAM/disk/NVIDIA health checks, detected software, bounded disk/VRAM warning, report. No automatic architecture resolver. |
| 01 | `environment` | Idempotent, pinned environment preparation, Heretic/runtime/quantization tools as requested, installation manifest, working imports and GPU test. |
| 02 | `acquire` | Resolve/download exact model revision, datasets and tool assets as required; verify download integrity and cache; immutable source manifest. |
| 03 | `published_baselines` | Default: ingest published model-card metrics (source links, date, evaluator and comparability caveat), do **not** run reference model inference. Optional local baseline mode later. |
| 04 | `prepare_engine` | Inspect pinned Heretic version, validate native config syntax and paths, prepare experiment arguments, confirm required commands exist; **no architecture compatibility resolver**. |
| 05 | `abliterate` | Run Heretic with bounded search, durable telemetry, logs, recovered engine artifacts, output candidate checkpoint and metadata. Handle signals and partial failures. |
| 06 | `qualify` | Fast: import engine metrics plus 20–30 independent generation/integrity checks, reload saved checkpoint. Extended: held-out refusals/KL/correctness (optional); separately label provenance. |
| 07 | `select_master` | Apply configured gates to candidates; if none qualify, do not silently promote one. Preserve BF16 master (or explicitly record actual precision) and parent lineage. |
| 08 | `quantize` | Generate requested FP8/GGUF variants using verified backend-specific recipe. Outputs independent and resumable; error on unsupported formats without corrupting other artifacts. |
| 09 | `verify_exports` | SHA-256 per file, file inventories, expected loader smoke tests when supported, tokenizer/config/processor preservation checks, optional modality smoke tests and explicit unverified flags. |
| 10 | `report_export` | Versioned artifact manifests, run summary, metrics provenance, environment/tool versions, cost/time estimates marked as estimates, failures, README artifact usage examples. |

Each stage must implement a common `StageContext`, `check_prerequisites()`, `execute()`, `validate_outputs()` and receipt writer (names may vary). Stage receipt includes stage ID, run ID, start/end UTC, effective config hash, input artifact IDs, tool revisions, output paths/hash, outcome, warnings, diagnostics and resume status. Implement atomic file writes and locks to avoid concurrent runs clobbering the same work area.

Stages must only run when needed; e.g., quantize-only should not install unrelated abliteration packages or download optimizer datasets. Missing optional paths should produce clear errors instead of guessed placeholders.

## 5. Heretic integration, experiment profiles and metrics

- Use an **adapter** rather than changing Heretic source unless technically necessary and documented. Pin exact upstream Git commit/tag/installed package version. Capture stdout, stderr and resource telemetry while keeping logs usable.
- Run official `heretic --help` and inspect upstream `config.default.toml`; map settings deliberately. Treat all published ARA parameter names as *research metadata* until a matching implementation has been verified. Verify Qwen3.8-27B loading and hybrid architecture support by actually running it on GPU; user assumes model/engine compatibility risk, platform reports runtime errors.
- Published reference: `coder3101/Qwen3.8-27B-heretic-ara`; store published refusal/KL values only with direct source provenance and evaluator metadata, not as local results. Do not infer benchmark equivalence from naming.
- `reproduce` attempts a pinned, documented reproduction of a published configuration. If source implementation, datasets, seed, or exact method is unavailable, record the limitation and use clearly identified approximate reproduction **only when explicitly permitted**. Do not silently substitute standard Heretic.
- `optimize` uses verified native Heretic optimization on the untouched source checkpoint; bounded max trials/time; trial metadata and failure recovery. Run may stop cleanly at budget and retain best completed candidate.
- Refusal count is not accuracy. Separate refusal grader outputs, responses to held-out prompts, benign quality/correctness, distribution shift (KL) and generation integrity. Any imported Heretic internal KL is *optimizer metric*, not automatically held-out validation.
- At minimum, record `model_revision`, `engine_revision`, prompt datasets/revisions and hashes, chat template, thinking mode, max tokens, sampling/seed, scorer version, optimizer objective, per-trial scores, and final checkpoint identity when available.
- Avoid claiming a full scientific reproduction or statistically meaningful generalization from short smoke tests. Stage 06 `fast` gates are quality assurance only; Project B will do deep coding evaluation separately.

## 6. Quantization and model precision

- Acquire the original high-precision Hugging Face checkpoint as the transformation source, rather than starting with the official FP8 derivative, unless users explicitly override source and understand the tradeoff.
- Preserve a master in its actual saved precision. If Heretic exports only a quantized, merged or converted format, record it honestly and do not label it BF16.
- Generate FP8 for the B200 (with explicit backend and scaling/calibration recipe); support a documented variant (e.g. a verified weight-only or block-scaled approach) **only where the toolchain can actually produce and load it**. `fp8` is not a universal single format. Record scheme, scale format, quantizer revision and calibration provenance; do not claim bytewise identity with official Qwen FP8.
- GGUF pathway: verify llama.cpp support; convert high-precision HF checkpoint to supported GGUF source format; optionally build imatrix; quantize requested variants such as `q8_0`, `q6_k`, `q5_k_m`, `q4_k_m`, `iq4_xs`, `iq3_s` only when supported by pinned converter; keep original safetensors intact.
- Multimodal: where model has vision/video/audio, verify projector/processor conversions separately as applicable. Never mark multimodal preservation as passed without a real test. If a requested format lacks compatibility, mark output as unsupported/failed and report it; do not generate a falsely complete package.
- Approximate A4000 targets: 16 GB VRAM standard RTX A4000; `iq4_xs` and `iq3_s` candidates, but fit depends on final file, GPU/CPU offload, KV-cache and runtime. No promise of full-GPU loading. Document tested context size and peak VRAM.
- Run a small comparison of text generation on each produced artifact. Log quantization time, peak memory, file size, load success and output validity. Do **not** embed Project B's HumanEval+/SWE-bench capability tests.
- Quantization should be independent and rerunnable per output type with a preserved source model.

## 7. Environment provisioning: Lambda Stack and Ubuntu

- `lambda-stack`: inspect driver, CUDA runtime/toolkit, torch and nvcc; reuse host GPU stack when compatible; project-isolated Python packages as needed. Never run whole-system upgrades by default.
- `ubuntu`: report missing packages, require explicit opt-in for apt/sudo/driver installation, install minimal reproducible prerequisites only; avoid silently rebooting. Support a `--setup-only` path and repeatable setup.
- Shared project Python environment via `uv` with version pinning/lockfile, where supported. Resolve CUDA/PyTorch compatibility from the actual environment (B200 Blackwell support); avoid installing CPU-only torch accidentally.
- Keep quantization backends separate where native build dependencies conflict. Pin binary/build provenance, including compiler options and CUDA arch.
- First successful setup should write a reusable component manifest; later runs skip validated versions but can install missing components. Never treat file existence alone as proof of a working executable.
- Make Hugging Face authentication optional with `HF_TOKEN` and private-model gated-repo errors intelligible; do not print secrets.
- Require checks for free disk with a configurable buffer; distinguish ephemeral instance disk from durable storage. Avoid auto-uploading user models to any public endpoint.

## 8. Artifact format and integration boundary with Project B

Export **one manifest per produced artifact**, including the original unmodified model. Schema version `1.0`; use stable artifact IDs and exact lineage. Manifest minimum:

```json
{
  "schema_version": "1.0",
  "artifact_id": "content-addressed-or-stable-unique-id",
  "parent_artifact_id": null,
  "run_id": "run-id",
  "model_family": "qwen3.8-27b",
  "source_model": "Qwen/Qwen3.8-27B",
  "source_revision": "resolved-hf-commit-sha",
  "transformation": {
    "kind": "original|abliteration|quantization",
    "method": "exact-engine-or-quantization-method",
    "config_hash": "sha256-hash",
    "tool_revision": "git-commit-or-version"
  },
  "precision": "bf16|fp8|iq4_xs|iq3_s|...",
  "artifact": {
    "format": "safetensors|gguf|...",
    "uri": "local-path-or-storage-uri",
    "files": [{"relative_path": "filename", "size_bytes": 0, "sha256": "..."}]
  },
  "tokenizer": {"revision": "...", "hash": "..."},
  "processor": {"revision": null, "hash": null},
  "quantization": {"recipe_id": null, "calibration_revision": null, "imatrix_hash": null},
  "inference": {"backend": null, "backend_revision": null, "tested_context": null},
  "validation": {"integrity_passed": false, "inference_passed": null, "modalities_tested": []},
  "evaluation": {"qualification_run_id": null, "metrics_provenance": "not-evaluated"}
}
```

Populate actual values, never placeholders in finalized manifests. Use `null` for unavailable optional fields, not fake hashes. For sharded models, hash every shard and accompanying config/tokenizer files. Keep schema machine-readable, versioned and tested. Preserve tokenizer and multimodal processor provenance. Expose paths/checksums so Project B can consume artifacts **without importing any Project A internals**.

## 9. Directory/repository layout

```text
llm-surgeon/
├── main.sh
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE-THIRD-PARTY.md
├── .gitignore
├── configs/
│   ├── pipeline.yaml
│   ├── pipeline.example.yaml
│   ├── baselines/
│   │   └── qwen3.8-27b.yaml
│   ├── heretic/
│   │   ├── reproduce.toml
│   │   └── optimize.toml
│   └── quantization/
│       └── default.yaml
├── schemas/
│   ├── pipeline.schema.json
│   └── artifact.schema.json
├── src/llm_surgeon/
│   ├── cli.py
│   ├── config.py
│   ├── orchestrator.py
│   ├── stages/
│   │   ├── stage00_preflight.py
│   │   ├── stage01_environment.py
│   │   ├── stage02_acquire.py
│   │   ├── stage03_baselines.py
│   │   ├── stage04_prepare.py
│   │   ├── stage05_abliterate.py
│   │   ├── stage06_qualify.py
│   │   ├── stage07_select.py
│   │   ├── stage08_quantize.py
│   │   ├── stage09_verify.py
│   │   └── stage10_report.py
│   ├── adapters/
│   │   ├── heretic.py
│   │   ├── hf_hub.py
│   │   ├── fp8.py
│   │   └── llama_cpp.py
│   ├── environment/
│   │   ├── lambda_stack.py
│   │   └── ubuntu.py
│   ├── artifacts/
│   ├── evaluation/
│   ├── subprocesses.py
│   ├── telemetry.py
│   └── errors.py
├── scripts/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── gpu/
├── docs/
│   ├── architecture.md
│   ├── stages.md
│   ├── cli.md
│   ├── configuration.md
│   ├── deployment-lambda.md
│   ├── deployment-ubuntu.md
│   ├── heretic-adapter.md
│   ├── quantization.md
│   ├── evaluation.md
│   ├── artifact-schema.md
│   ├── troubleshooting.md
│   ├── security-and-trust.md
│   ├── limitations.md
│   └── implementation-log.md
└── examples/
    ├── qwen-b200.yaml
    ├── ubuntu-quantize-only.yaml
    └── resume.md
```

Put large downloaded weights, datasets, environments and builds outside Git, defaulting to configurable `./workspace/` ignored in Git. Structure:

```text
workspace/
├── cache/
├── downloads/
├── tools/
├── datasets/
├── models/{source,references,candidates,selected}/
├── runs/<run-id>/{logs,receipts,configs,metrics,checkpoints,reports}/
└── exports/{bf16,fp8,gguf}/
```

Avoid recursive copying of gigantic checkpoints when a verified immutable reference/hardlink/snapshot strategy suffices; prefer atomic rename/copy as appropriate and record storage semantics.

## 10. Logging, failures, repeatability and operational guardrails

- Structured JSONL logging + readable log per run/stage, plus stdout progress. Redact access tokens, credentials, local secret files and sensitive model prompts if requested.
- Stage status: pending/running/succeeded/failed/skipped/blocked; atomic receipts and status updates. SIGINT/SIGTERM cleanup; child process termination; preserve partial artifacts without falsely marking complete.
- Dry run reports plan and predicted downloads when metadata readily known without violating no-network guarantee; uncertain size is `unknown`, not guessed.
- GPU memory, RAM, disk free, elapsed time, peak utilization metrics when available. Resource measurements must be factual; cost estimates require configurable hourly rate and clear distinction between measured runtime and projected cost.
- Download resume using official HF cache machinery; avoid trusting incomplete or corrupted artifacts.
- Filesystem locking, path traversal protections, immutable source snapshots, storage quota and free-space checks, bounded optimizer budget.
- Optional cleanup command should always be dry-run first and must never delete sources or finalized exported models without explicit confirmation.
- Document code-execution risks from `trust_remote_code`, third-party model source code, native converters and evaluation-generated content. Make opt-in security behavior explicit.
- Third-party license obligations must be documented; check Heretic AGPL terms rather than falsely assuming all components are permissively licensed.

## 11. Documentation deliverables

Implement all files referenced in the repository tree with substantial, concrete content rather than placeholders:

- `README.md`: project overview, compatibility disclaimer, features, prerequisites, 5-minute quick start, one-B200 example, Ubuntu example, run/resume/quantize-only, troubleshooting pointer, outputs.
- `docs/architecture.md`: component responsibilities, run lifecycle, data flow, design tradeoffs, dependency graph, no-resolver decision, Project B boundary.
- `docs/stages.md`: all stage gates, inputs, outputs, receipts, restart/invalidation rules.
- `docs/cli.md`: every flag, examples, precedence, exit codes, mutually exclusive flags.
- `docs/configuration.md` and schema: field types/defaults, override behavior, validation errors, explanation of engine-specific versus platform options.
- `docs/deployment-lambda.md`: supplied Lambda Stack versus fresh Ubuntu selection on Lambda, B200 preflight, storage persistence risks, secrets and commands.
- `docs/deployment-ubuntu.md`: Ubuntu prerequisites, guarded install steps, CUDA driver guidance, troubleshooting.
- `docs/heretic-adapter.md`: exact inspected CLI/TOML mapping, pinned version, status of ARA support, reproduction limitations, sample engine invocation.
- `docs/quantization.md`: BF16 source rule, FP8 recipes/backends, GGUF conversion, optional imatrix, multimodal limitations, VRAM caveats.
- `docs/evaluation.md`: published baseline provenance versus optimizer metrics versus independent fast/extended qualification and Project B boundary.
- `docs/artifact-schema.md`: manifest semantics, example and compatibility/reproducibility commitments.
- `docs/security-and-trust.md`: risk profile of arbitrary weights/remote code/untrusted repositories and sandbox guidance.
- `docs/implementation-log.md`: record unexpected errors and fixes so installer scripts can improve.
- `docs/troubleshooting.md` and `docs/limitations.md`: honest known failure modes.

Compatibility notice text (include in README/config comments and at initial run; `--yes` acknowledges only where applicable):

> This platform does not automatically verify that your model architecture is supported by the selected abliteration engine or quantization backend. Verify compatibility with the engines' official documentation before execution. Unsupported models or formats may fail during model loading, transformation, or export. The platform manages execution, dependencies, logs, and artifacts but does not guarantee compatibility with every model.

## 12. Test suite and acceptance criteria

### Unit tests (required on CPU)
- Config parsing, defaults and schema; CLI override precedence; mutual exclusivity and invalid values.
- Hugging Face URL normalization and malicious input rejection; revision pinning unit behavior.
- Stage dependency graph, `--from`/`--through`/`--only`, resume state, invalidation, config fingerprint.
- `--dry-run` causes no network, disk mutations, installs, GPU imports or downloads (use spies/mocks).
- Atomic receipts, checksums/file inventories for sharded outputs, manifest schema round trip.
- Missing required files, unsupported export formats, corrupted downloads, failed subprocess, interrupts, partial output handling.
- Adapter command construction using fixtures representing inspected upstream help and native config (no fabricated flags).
- Model source immutability and artifact-parent lineage.
- Environment selection, Lambda Stack/Ubuntu profile behavior and idempotent setup.

### Integration tests (mocked/no GPU)
- Full 00–10 mock pipeline with recorded outputs, stage receipts, fake Heretic and quantization binaries.
- Quantize-only skipping Heretic.
- Failed quantization of one format preserves validated other outputs when partial outputs enabled.
- Resume after Stage 05 simulated crash and after Stage 08 simulated crash.
- Published-only Stage 03 performs zero model inference/download of reference checkpoint.
- Stage 06 fast tests use held-out prompts and clearly differentiate imported optimizer metrics versus measured smoke checks.

### GPU smoke tests (run only when B200 exists, not claimed otherwise)
- `nvidia-smi`, CUDA/PyTorch B200 smoke, model load and single generation from source model, Heretic engine invocation/preparation.
- Small actual Heretic operation/trial before full optimization budget (bounded and explicitly authorized).
- Saved candidate reload and integrity validation.
- FP8 backend actual loading test; GGUF actual conversion/loader test and explicit tested context.
- Optional multimodal smoke test if exported modality support is claimed.

### Acceptance criteria
1. `./main.sh --help` works on Ubuntu without a GPU.
2. `./main.sh --config configs/pipeline.yaml --dry-run` describes a valid plan with **zero side effects**.
3. CLI flags override YAML; `--lambda-stack` and `--ubuntu` reject simultaneous use.
4. Setup can be rerun without unnecessary reinstall; both environment profiles implemented and testable.
5. Model ID/URL override changes the actual resolved source and cannot contaminate resumed runs.
6. Published-only baselines do not spend GPU time evaluating original/reference models.
7. Heretic adapter uses verified upstream interface and reports if ARA reproduction cannot be guaranteed.
8. Stage 06 fast distinguishes optimizer metrics from independent measurements.
9. BF16 master remains untouched during quantization and all outputs trace to the original commit.
10. Quantization exporters produce real artifacts where supported, and cleanly report unsupported cases.
11. A manifest is generated for source and each export, with complete checksums and parent lineage.
12. Stage resumability, precise failure reasons, and clean partial-format handling pass mocked integration tests.
13. Project B is **not implemented** or embedded in this repository.
14. All documentation above exists, accurate to verified tool versions; `docs/implementation-log.md` records encountered issues.
15. Final agent response includes what was implemented, actual test output, unverified assumptions, blockers, repository paths, and exact commands to run on the B200.

## 13. Implementation order and agent execution instructions

Build in this order, committing after meaningful milestones if the repository workflow permits:

**Milestone 1 — Skeleton and contracts.** README, architecture doc, directory structure, config schema, argument parsing, orchestration interfaces, stage IDs, manifest schema, mock stage test fixtures. Run CPU tests.

**Milestone 2 — Provenance and setup.** Environment detectors/provisioners, revision resolution/download/cache, reproducibility receipts, baseline metadata import. Implement `--setup-only`, `--through 2`, `--dry-run`. Test idempotence and no side effects.

**Milestone 3 — Abliteration.** Verified Heretic adapter, stage 04 native config translation, stages 05–07, fast qualification, restart and error handling. Provide a minimal bounded real-GPU test path and no unfounded success assertions.

**Milestone 4 — Quantization and artifact packages.** Real verified FP8 and GGUF pathways, output-type isolation, stage 09 integrity/inference validation, manifest export, stage 10 reporting.

**Milestone 5 — Hardening and documentation.** Negative-path tests, mock full pipeline, interruption/resume scenarios, additional docs, implementation log, CLI examples, license notices, final user handoff.

**If GPU/B200 is unavailable during implementation:** complete CPU-testable orchestration and mock integration; ensure GPU-required stages fail with useful prerequisites; provide explicit runbook and do not claim successful real-model abliteration. If third-party functionality is unsupported or unavailable, isolate that limitation behind an honest adapter and preserve useful working components.

**Do not pause to ask for routine preferences.** Follow these defaults unless truly blocked by destructive or credentialed operations. Do not claim background work or future results. Deliver real code, tests and docs in the current engineering session.

## 14. Useful primary references for verification

- Heretic source and current README: https://github.com/p-e-w/heretic
- Lambda Stack software and provisioning: https://lambda.ai/lambda-stack-deep-learning-software
- Original Qwen model: https://huggingface.co/Qwen/Qwen3.8-27B
- Official FP8 reference (not source checkpoint): https://huggingface.co/Qwen/Qwen3.8-27B-FP8
- Published ARA reference to investigate: https://huggingface.co/coder3101/Qwen3.8-27B-heretic-ara
- llama.cpp: https://github.com/ggml-org/llama.cpp
- llama.cpp quantization docs: https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md

**Final mandate:** Engineer a reusable, reliable platform for running supported abliteration and quantization engines against user-selected models. No model compatibility resolver, no heavy default published-baseline reruns, no second research repository, and no false success claims. Start implementing now.
