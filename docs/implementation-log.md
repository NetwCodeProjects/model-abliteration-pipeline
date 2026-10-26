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
