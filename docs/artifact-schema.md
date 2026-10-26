# Artifact manifests (schema 1.0)

One manifest per artifact — the unmodified source, the promoted master and every export — validated
against `schemas/artifact.schema.json` (JSON Schema 2020-12) before it is written. Locations:

- `workspace/manifests/<artifact_id>.json` (global store)
- `<artifact dir>.manifest.json` sidecar next to platform-owned artifacts (not for HF-cache or
  user-supplied directories)
- `runs/<run>/reports/manifests/<role>.<artifact_id>.json` and `reports/manifest-index.json`

## Example (master)

```json
{
  "schema_version": "1.0",
  "artifact_id": "art-3f0c9a1e5b7d2c4a6e8f0b1d",
  "parent_artifact_id": "art-9b1e2d3c4f5a6b7c8d9e0f1a",
  "run_id": "20261026T120000Z-a1b2c3",
  "model_family": "qwen3.8-27b",
  "source_model": "Qwen/Qwen3.8-27B",
  "source_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  "transformation": {"kind": "abliteration", "method": "heretic:ARA:optimize",
                     "config_hash": "<sha256 of the generated config.toml>", "tool_revision": "171bb98b8b9a41c081631b5e678f7133128973e0"},
  "precision": "bf16",
  "artifact": {"format": "safetensors", "uri": "/…/workspace/exports/bf16/qwen3.8-27b-optimize-<run>",
               "files": [{"relative_path": "model-00001-of-00011.safetensors", "size_bytes": 4993459520, "sha256": "…"}]},
  "tokenizer": {"revision": "1d4bf0f2…", "hash": "<sha256 over tokenizer files>"},
  "processor": {"revision": "1d4bf0f2…", "hash": "<sha256 over processor configs>"},
  "quantization": {"recipe_id": null, "calibration_revision": null, "imatrix_hash": null},
  "inference": {"backend": "transformers", "backend_revision": "transformers 5.18.0", "tested_context": 412},
  "validation": {"integrity_passed": true, "inference_passed": true, "modalities_tested": ["text"]},
  "evaluation": {"qualification_run_id": "20261026T120000Z-a1b2c3", "metrics_provenance": "fast-qualification"},
  "extensions": {"label": "optimize", "ungated": false, "gates": [], "heretic_parameters": {}, "tensor_diff_vs_source": {}, "storage_semantics": "hardlink"}
}
```

## Field semantics

| Field | Meaning |
|---|---|
| `artifact_id` | `art-` + 24 hex of sha256 over sorted `(relative_path, size, sha256)`; identical bytes ⇒ identical ID |
| `parent_artifact_id` | direct input artifact (`null` for the source); the chain always ends at the source |
| `source_model` / `source_revision` | original repo and **commit SHA**; both `null` for locally supplied models (`extensions.provenance = "supplied"`) |
| `transformation.kind` | `original` · `abliteration` · `quantization` |
| `transformation.method` | e.g. `huggingface_hub.snapshot_download (…)`, `heretic:ARA:approximate_reproduction`, `llmcompressor.model_free_ptq:FP8_BLOCK`, `llama.cpp:convert_hf_to_gguf(bf16)+llama-quantize:IQ4_XS` |
| `transformation.config_hash` | sha256 of the exact transformation recipe/config |
| `precision` | measured (`bf16`, `fp8`, `iq4_xs`, `mixed:…`), never assumed |
| `artifact.files` | **every** file incl. all shards, config, tokenizer, processor, mmproj; sha256 always present |
| `tokenizer.hash` / `processor.hash` | sha256 over the respective files; `null` when not shipped as files (GGUF embeds the tokenizer) |
| `quantization.*` | `recipe_id`, calibration origin, imatrix sha256; `null` when not applicable |
| `inference.*` | backend actually used for the loader test and the context length actually tested |
| `validation.integrity_passed` | files re-hashed and structurally valid |
| `validation.inference_passed` | `true`/`false` from a real loader test, **`null` = not tested** |
| `validation.modalities_tested` | only modalities exercised by a real test (`["text","image"]` only after an mmproj test) |
| `evaluation.metrics_provenance` | `not-evaluated`, `integrity-only`, `optimizer-only`, `fast-qualification`, `extended-qualification` |
| `extensions` | optional details; consumers must tolerate unknown keys here |

## Commitments

- No placeholders: a manifest with an unhashed file or malformed field is refused (`E_MANIFEST_INVALID`).
- `schema_version` changes on any breaking change; additive information goes into `extensions`.
- Re-verification: any consumer can recompute `artifact_id` from `artifact.files` and compare file hashes.
- Project B reads manifests only; it does not import `model_pipeline`.
