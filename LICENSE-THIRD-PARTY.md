# Third-party components and license obligations

This repository's own license has not been chosen yet (see `pyproject.toml`). It does **not** vendor or modify the
engines below; stage 01 downloads them at pinned revisions into `workspace/tools/` and runs them as
separate processes. Their licenses still apply to anything you redistribute.

| Component | Pinned version | License | How it is used | Obligations to note |
|---|---|---|---|---|
| [Heretic](https://github.com/p-e-w/heretic) | git `171bb98b8b9a41c081631b5e678f7133128973e0` | **AGPL-3.0-or-later** | Installed in its own venv; invoked as the `heretic` CLI; `scripts/runtime/heretic_*.py` import its public modules inside that venv | If you distribute Heretic, or a modified version, or offer a modified version over a network, you must provide the corresponding source under AGPL. Unmodified CLI use is unaffected. `configs/heretic/*.toml` and `tests/fixtures/heretic_config.default.171bb98b.toml` reproduce upstream configuration values (attributed in-file). |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | tag `b11889` (`d854d98`) | MIT | Built from source; `convert_hf_to_gguf.py` + `gguf-py` in a converter venv | Keep the MIT notice when redistributing binaries. |
| [llm-compressor](https://github.com/vllm-project/llm-compressor) | 0.14.0.1 | Apache-2.0 | FP8 `model_free_ptq` in the `fp8` venv | NOTICE retention on redistribution. |
| [compressed-tensors](https://github.com/vllm-project/compressed-tensors) | 0.19.0 | Apache-2.0 | FP8 checkpoint format | — |
| [vLLM](https://github.com/vllm-project/vllm) | 0.32.0 | Apache-2.0 | FP8 loader smoke test in the `vllm` venv | — |
| PyTorch, transformers, optuna, peft, datasets, accelerate, bitsandbytes | resolved from Heretic's `uv.lock` / tool pins | BSD / Apache-2.0 / MIT | Inside tool venvs | — |
| uv | 0.13.5 (control venv) | Apache-2.0 / MIT | Environment provisioning | — |
| huggingface_hub, PyYAML, psutil, tomli | see `uv.lock` | Apache-2.0 / MIT / BSD | Control plane | — |

## Models and datasets

Model weights carry their own licenses (e.g. `Qwen/Qwen3.8-27B` is Apache-2.0 per its model card).
Prompt datasets used by default (`mlabonne/harmful_behaviors`, `mlabonne/harmless_alpaca`) do not
state a license on the Hub; check before redistributing derived data. Abliterated derivatives inherit
the base model's license terms; publication is never automatic in this platform.
