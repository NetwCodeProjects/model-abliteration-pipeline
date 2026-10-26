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

### Real-tool verification (no GPU claims)

See the section appended below for the CPU/WSL2 runs.
