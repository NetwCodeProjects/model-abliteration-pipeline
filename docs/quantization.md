# Quantization

## Source rule and precision honesty

- Abliteration starts from the **original high-precision checkpoint** (`Qwen/Qwen3.8-27B`, BF16), never
  from the official `-FP8` derivative, unless you point `--model-url`/`--model-path` elsewhere yourself.
- Heretic's `quantization = "bnb_4bit"` is memory-saving *loading* during the search; it is not a
  deployment format. The default template uses `"none"`.
- The master's precision is read from the saved safetensors headers (`bf16`, `f16`, `f32` or
  `mixed:…(dominant …)`) and recorded; the export directory is named after the actual precision.
- Quantization reads the master (or source) and never writes to it; its quick fingerprint is compared
  before and after stage 08 (`E_INPUT_MUTATED`), and the master files are read-only.

## FP8 (B200)

Backend: llm-compressor **0.14.0.1** `model_free_ptq` (data-free; operates directly on safetensors
shards, no model class needed — suited to new hybrid/multimodal architectures), output in
**compressed-tensors** format, loadable by vLLM.

| `quantization.fp8.scheme` | Preset | Weights | Activations |
|---|---|---|---|
| `auto` → `fp8_block` | `FP8_BLOCK` | FP8 E4M3, static, 128×128 block scales | FP8 E4M3, dynamic, groups of 128 |
| `fp8_dynamic` | `FP8_DYNAMIC` | FP8 E4M3, static, per-output-channel scales | FP8 E4M3, dynamic, per token |

Default `ignore` (kept in source precision): `lm_head`, `embed_tokens`, the vision tower (`visual.*`,
`vision_tower`), MoE router gates (`mlp.gate`), Gated-DeltaNet `in_proj_a/b/ba` (not 128-divisible),
`mtp.fc`, every `*norm*` tensor and the Gated-DeltaNet `conv1d` weights. llm-compressor only skips names
*ending* in `norm` automatically; on a real Qwen3.5 checkpoint it tried to quantize the 3-D `linear_attn.conv1d.weight`
and the 1-D `mtp.pre_fc_norm_embedding.weight` and aborted (same ignores as upstream's Qwen3.5 `model_free_ptq` example). This mirrors the official Qwen3.8 FP8
release's choices (block 128×128, dynamic activations, vision excluded) but the files are **not**
byte-identical to it. The recipe (`recipe_id`, scheme, scale format, ignore list, versions) is stored in
the manifest; `calibration_revision` is `null` because no calibration data is used. Calibrated static
FP8 is not implemented (`calibration_manifest` is rejected).

Verification: stage 09 loads the export with **vLLM 0.32.0** (`max_model_len` 8192 by default) and
generates greedy answers to the smoke prompts (text only; image inputs are not tested for FP8, so
`modalities_tested` is `["text"]` at most).

## GGUF (llama.cpp b11889)

1. `convert_hf_to_gguf.py <master> --outtype bf16` (converter venv uses transformers 5.18.0 instead of
   llama.cpp's pinned 4.57.6, which cannot load `Qwen3_5Tokenizer` — verified failure) → one high-precision intermediate per master and
   converter revision (`workspace/models/gguf-intermediate/…`, reused across formats/runs).
   `Qwen3_5ForConditionalGeneration` is registered for the text model (`QWEN35`) at the pin.
2. Vision projector: if `config.json` has `vision_config` and `preserve_multimodal: true`,
   `convert_hf_to_gguf.py --mmproj --outtype f16`. If that fails, every GGUF format **fails**
   (`E_MMPROJ_FAILED`) instead of producing a text-only package silently; set
   `preserve_multimodal: false` to export text-only on purpose.
3. Optional importance matrix: `llama-imatrix -m <bf16.gguf> -f <calibration.txt>` from
   `quantization.gguf.imatrix_dataset` (local file or `hf://datasets/<repo>@<rev>/<file>`).
4. `llama-quantize [--imatrix f] <bf16.gguf> <out.gguf> <TYPE>` per format, each into its own
   `exports/gguf/<family>-<parent>-<type>/` with the mmproj file hardlinked alongside.

Supported types are exactly llama-quantize's table at the pin (q8_0, q6_k, q5_k_m, q4_k_m, iq4_xs,
iq3_s, …). `iq1_*`, `iq2_*`, `iq3_xxs` and `q2_k_s` **require an imatrix** (llama.cpp aborts otherwise);
the pipeline fails those formats up front with `E_IMATRIX_REQUIRED`. `iq4_xs`/`iq3_s` run without one
but are better with it.

Verification: GGUF metadata (architecture, tokenizer, chat template) is read with a pure-Python
parser; `llama-completion -no-cnv --temp 0 -c <smoke_context> -ngl <layers>` must produce non-degenerate
text; with an mmproj, `llama-mtmd-cli --image <generated red square>` must run (only then `image` is
listed in `modalities_tested`).

## Approximate sizes and the RTX A4000 (16 GB)

| Format | ≈ bpw | ≈ size for 27B | Fits 16 GB fully on GPU? |
|---|---:|---:|---|
| q8_0 | 8.5 | 29 GB | no |
| q4_k_m | 4.85 | 16.5 GB | no |
| iq4_xs | 4.25 | 14.5 GB | weights alone nearly fill the card; needs partial offload or a tiny context |
| iq3_s | 3.44 | 11.8 GB | likely with short context and quantized KV cache; not guaranteed |

Fit depends on the final file, context length, KV-cache type (`-ctk/-ctv`), the mmproj (~1 GB) and
runtime overhead. The platform records the context it actually tested (`inference.tested_context`) and
the peak GPU memory it sampled; it never promises full-GPU loading.

## Independence and resumability

Each format is a separate job with a sub-receipt (`receipts/08-quantize/<fmt>.json`) and a recipe ID.
A failing format is cleaned up (`.partial` removed) and does not affect others; with
`execution.allow_partial_exports: true` the run ends with exit 20 and the valid exports keep their
manifests. Re-running reuses any finalized export with the same parent artifact and recipe ID.
