# Configuration

The YAML file is **our orchestration contract**. Stage 04 translates the `abliteration.*` keys into
Heretic's own TOML vocabulary (validated against the installed Heretic); stage 08 translates
`quantization.*` into llm-compressor recipes and llama.cpp arguments. Engine-native settings that we do
not model live in the native template (`abliteration.heretic_config`, default
`configs/heretic/<experiment>.toml`), never in the pipeline YAML.

- **Schema:** `schemas/pipeline.schema.json` (JSON Schema 2020-12) is generated from
  `src/model_pipeline/configspec.py` (`python scripts/generate_schema.py`; a unit test fails if it is stale).
- **Strictness:** unknown keys, wrong types, out-of-range numbers and unknown enum values are errors
  (`E_CONFIG_INVALID`, exit 3) listing every problem with its dotted path.
- **YAML 1.2 booleans:** only `true`/`false` are booleans. `off`, `no`, `yes` stay strings
  (PyYAML's YAML 1.1 default would silently turn `qualification: off` and the key `yes:` into booleans).
- **Paths:** relative paths in config files resolve against the repository root; CLI paths against the
  current directory.
- **Profiles:** `profiles: [file.yaml, …]` (relative to the config file) are merged *below* the file.
- **Precedence:** CLI > YAML > profiles > defaults (new runs); stored config + CLI (resume).
- **Operational keys** (marked below) never change fingerprints and may be changed on resume.

Cross-field rules: `model.url` xor `model.local_path` (one required unless `--setup-only`);
`model.revision` only with `url`; formats must be `fp8` or a verified GGUF type; `fp8.calibration_manifest`
is rejected (data-free schemes only); `quantization.input=path` needs `input_path`; abliteration disabled ⇒
`input` becomes `source`; `baseline_mode=local` is unsupported in v1; `experiment=reproduce` needs
`reproduce.source` or `reproduce.reference_repo`; `torch_source=system` only with `lambda-stack`.

## Engine-specific vs platform options

| Concern | Where it lives |
|---|---|
| Trials, budget, seed, Pareto choice, export | `abliteration.*` (platform; mapped to Heretic `n_trials`, `seed`, `trial_index`, …) |
| Scorers, modifier (ARA/Abliteration), ARA hyper-parameters, datasets, dtypes, device map, batch size | native template TOML |
| Reproduction source / approximation permission | `abliteration.reproduce.*` |
| FP8 scheme and ignore list | `quantization.fp8.*` → llm-compressor `model_free_ptq` |
| GGUF types, imatrix, multimodal projector | `quantization.gguf.*` → llama.cpp tools |

## Reference

| Key | Type | Default | Operational | Description |
|---|---|---|:-:|---|
| `schema_version` | string (1.0) | `"1.0"` |  | Config schema version. |
| `profiles` | array | `[]` |  | Optional model/engine profile YAML files (relative to this config) merged below this file. |
| `model.url` | string\|null | `null` |  | Hugging Face repo ID or canonical https://huggingface.co/<org>/<name> URL. |
| `model.revision` | string\|null | `null` |  | Branch, tag or commit; always resolved and pinned to a commit SHA. |
| `model.local_path` | string\|null | `null` |  | Local model directory (mutually exclusive with url). |
| `model.trust_remote_code` | boolean | `false` |  | Allow third-party model code in our own loaders. Heretic cannot be told this non-interactively; see docs/security-and-trust.md. |
| `model.hf_token_env` | string | `"HF_TOKEN"` |  | Name of the environment variable holding the HF token (the value is never stored). |
| `model.family` | string\|null | `null` |  | model_family for manifests; defaults to the lower-cased repo name. |
| `environment.profile` | string (auto\|lambda-stack\|ubuntu) | `"auto"` |  | Host profile. |
| `environment.install_missing` | boolean | `true` |  | Install missing project-local components (tool venvs, llama.cpp build). |
| `environment.allow_system_packages` | boolean | `false` |  | Allow 'sudo apt-get install' of missing host packages (also needs --yes or a TTY confirmation). |
| `environment.allow_driver_changes` | boolean | `false` |  | Allow the guarded NVIDIA driver install step on the ubuntu profile. |
| `environment.allow_reboot` | boolean | `false` |  | Reserved. v1 never reboots; it stops and prints the reboot command. |
| `environment.python_manager` | string (uv) | `"uv"` |  |  |
| `environment.python_version` | string | `"3.12"` |  | Python for tool environments (uv-managed). |
| `environment.torch_backend` | string | `"auto"` |  | uv --torch-backend for tool environments; 'auto' follows the installed driver. |
| `environment.torch_source` | string (pip\|system) | `"pip"` |  | 'system' reuses the host torch via --system-site-packages (lambda-stack only, experimental). |
| `environment.gpu_device` | string | `"auto"` |  | 'auto' or a CUDA_VISIBLE_DEVICES value. |
| `environment.cache_root` | string\|null | `null` |  | Cache directory; defaults to <storage.root>/cache. |
| `environment.min_free_disk_gb` | number | `50` |  | Hard floor of free disk space; stages refuse to start below it. |
| `environment.disk_buffer_gb` | number | `100` |  | Safety margin added to disk estimates before warning. |
| `environment.uv_executable` | string\|null | `null` |  | Explicit uv binary; default: control venv uv, then PATH. |
| `abliteration.enabled` | boolean | `true` |  |  |
| `abliteration.engine` | string (heretic) | `"heretic"` |  |  |
| `abliteration.experiment` | string (reproduce\|optimize) | `"optimize"` |  |  |
| `abliteration.engine_repo` | string | `"https://github.com/p-e-w/heretic.git"` |  |  |
| `abliteration.engine_revision` | string\|null | `null` |  | Git commit/tag; null uses the verified pin 171bb98b8b9a. |
| `abliteration.heretic_config` | string\|null | `null` |  | Native Heretic TOML template; null uses configs/heretic/<experiment>.toml. |
| `abliteration.max_trials` | integer | `50` |  | Heretic n_trials (optimize). |
| `abliteration.n_startup_trials` | integer\|null | `null` |  | Heretic n_startup_trials; null = min(template value, max_trials//3). |
| `abliteration.max_runtime_minutes` | integer\|null | `180` | yes | Wall-clock budget; on expiry Heretic is interrupted and saves the configured Pareto trial. |
| `abliteration.shutdown_grace_minutes` | integer | `90` | yes | Time allowed for saving after a budget stop before termination. |
| `abliteration.seed` | integer\|null | `null` |  | Heretic seed; null generates one and records it. |
| `abliteration.pareto_trial_index` | integer | `0` |  | Index into Heretic's sorted Pareto front (0 = fewest refusals). |
| `abliteration.save_intermediate` | boolean | `true` |  | Keep candidate checkpoints after promotion. |
| `abliteration.reproduce.source` | string\|null | `null` |  | reproduce.json: local path, https URL or hf://<repo>@<rev>/<path>. |
| `abliteration.reproduce.reference_repo` | string\|null | `null` |  | Published reference to probe for reproduce/reproduce.json. |
| `abliteration.reproduce.reference_revision` | string\|null | `null` |  |  |
| `abliteration.reproduce.allow_approximate` | boolean | `false` |  | Permit approximate reproduction from published parameters when exact data is unavailable. |
| `abliteration.reproduce.ignore_environment_mismatches` | boolean | `false` |  | Passed to Heretic ignore_mismatches for exact reproduction attempts. |
| `evaluation.baseline_mode` | string (published_only\|local) | `"published_only"` |  |  |
| `evaluation.published_baselines` | string\|null | `null` |  | YAML with published metrics and provenance. |
| `evaluation.qualification` | string (off\|fast\|extended) | `"fast"` |  |  |
| `evaluation.independent_smoke_prompts` | integer | `30` |  | Upper bound on fast-qualification prompts (benign + held-out). |
| `evaluation.benign_prompts_file` | string | `"configs/qualification/benign_prompts.yaml"` |  |  |
| `evaluation.heldout_refusal_prompts` | array | `[{"dataset": "mlabonne/harmful_behaviors", "commit": "01c...` |  | Held-out prompts disjoint from Heretic's default train[:400]/test[:100] splits. |
| `evaluation.heldout_harmless_prompts` | array | `[{"dataset": "mlabonne/harmless_alpaca", "commit": "02c6a...` |  | Held-out harmless prompts for extended KL. |
| `evaluation.extended_kl` | boolean | `false` |  |  |
| `evaluation.extended_correctness` | boolean | `false` |  |  |
| `evaluation.enable_thinking` | boolean | `false` |  | Chat-template thinking mode during qualification. |
| `evaluation.max_new_tokens` | integer | `256` |  |  |
| `evaluation.seed` | integer | `0` |  |  |
| `evaluation.refusal_markers` | array | `["i can't", "i cannot", "i can not", "i won't", "i will n...` |  |  |
| `selection.max_optimizer_refusal_rate` | number\|null | `0.2` |  |  |
| `selection.max_optimizer_kl` | number\|null | `0.1` |  |  |
| `selection.max_heldout_refusal_rate` | number\|null | `0.35` |  |  |
| `selection.min_benign_pass_rate` | number\|null | `0.7` |  |  |
| `selection.require_integrity` | boolean | `true` |  |  |
| `selection.require_reload` | boolean | `true` |  |  |
| `selection.optimizer_refusal_score` | string | `"Refusals"` |  | Heretic score name used as refusal rate. |
| `selection.optimizer_kl_score` | string | `"KL divergence"` |  | Heretic score name used as KL. |
| `selection.allow_ungated_promotion` | boolean | `false` |  | Promote even if gates fail (master is labelled 'ungated'). |
| `selection.graft_missing_source_tensors` | array | `[]` |  | Tensor-name prefixes absent from the candidate to copy verbatim from the source (e.g. 'mtp.'). |
| `quantization.enabled` | boolean | `true` |  |  |
| `quantization.input` | string (selected_master\|source\|path) | `"selected_master"` |  |  |
| `quantization.input_path` | string\|null | `null` |  |  |
| `quantization.formats` | array | `["fp8", "iq4_xs", "iq3_s"]` |  |  |
| `quantization.fp8.scheme` | string (auto\|fp8_block\|fp8_dynamic) | `"auto"` |  |  |
| `quantization.fp8.calibration_manifest` | string\|null | `null` |  | Not used by the data-free schemes offered in v1; setting it is an error. |
| `quantization.fp8.ignore` | array | `["lm_head", "re:.*embed_tokens$", "re:.*visual\\..*", "re...` |  |  |
| `quantization.fp8.backend` | string (llmcompressor) | `"llmcompressor"` |  |  |
| `quantization.fp8.llmcompressor_version` | string | `"0.14.0.1"` |  |  |
| `quantization.fp8.compressed_tensors_version` | string | `"0.19.0"` |  |  |
| `quantization.fp8.device` | string | `"cuda:0"` |  |  |
| `quantization.fp8.verify_backend` | string (vllm\|none) | `"vllm"` |  |  |
| `quantization.fp8.vllm_version` | string | `"0.32.0"` |  |  |
| `quantization.fp8.vllm_max_model_len` | integer | `8192` |  |  |
| `quantization.fp8.vllm_gpu_memory_utilization` | number | `0.85` |  |  |
| `quantization.gguf.llama_cpp_repo` | string | `"https://github.com/ggml-org/llama.cpp.git"` |  |  |
| `quantization.gguf.llama_cpp_revision` | string\|null | `null` |  | null uses the verified pin b11889. |
| `quantization.gguf.build_cuda` | string (auto\|on\|off) | `"auto"` |  |  |
| `quantization.gguf.cuda_architectures` | string | `"auto"` |  | 'auto' derives from nvidia-smi compute capability (B200 -> 100). |
| `quantization.gguf.converter_transformers` | string\|null | `"5.18.0"` |  | transformers for the converter venv. llama.cpp pins 4.57.6, which cannot load Qwen3.5 tokenizers (upstream README: upgrade for newer models). null keeps the upstream pin. |
| `quantization.gguf.intermediate_type` | string (bf16\|f16\|f32) | `"bf16"` |  |  |
| `quantization.gguf.imatrix_dataset` | string\|null | `null` |  | Calibration text: local file or hf://datasets/<repo>@<rev>/<file>. |
| `quantization.gguf.imatrix_chunks` | integer\|null | `null` |  |  |
| `quantization.gguf.preserve_multimodal` | boolean | `true` |  |  |
| `quantization.gguf.mmproj_type` | string (f16\|bf16\|f32) | `"f16"` |  |  |
| `quantization.gguf.smoke_context` | integer | `4096` |  |  |
| `quantization.gguf.smoke_gpu_layers` | string | `"auto"` |  | 'auto' = all layers on a CUDA build, 0 on CPU builds. |
| `quantization.gguf.smoke_timeout_minutes` | integer | `30` | yes |  |
| `quantization.gguf.keep_intermediate` | boolean | `true` |  |  |
| `quantization.smoke_prompts` | array | `["What is the capital of France? Answer in one word.", "W...` |  |  |
| `quantization.smoke_max_tokens` | integer | `64` |  |  |
| `execution.resume` | boolean | `true` | yes | Auto-resume the latest incomplete run with an identical config fingerprint. |
| `execution.stop_on_failure` | boolean | `true` | yes |  |
| `execution.allow_partial_exports` | boolean | `true` |  |  |
| `execution.yes` | boolean | `false` | yes |  |
| `execution.dry_run` | boolean | `false` | yes |  |
| `execution.hourly_rate_usd` | number\|null | `null` | yes | Used only for clearly-labelled cost estimates. |
| `storage.root` | string | `"./workspace"` |  |  |
| `storage.retain_intermediate` | boolean | `true` |  |  |
| `storage.artifact_checksums` | string (sha256) | `"sha256"` |  |  |
| `storage.rehash_on_verify` | boolean | `true` | yes |  |
| `storage.link_strategy` | string (auto\|hardlink\|copy) | `"auto"` |  |  |
| `logging.console_level` | string (debug\|info\|warning) | `"info"` | yes |  |
| `logging.redact_prompts` | boolean | `false` |  | Store hashes instead of prompt/response text in qualification records. |
| `logging.resource_sample_seconds` | integer | `15` | yes |  |
