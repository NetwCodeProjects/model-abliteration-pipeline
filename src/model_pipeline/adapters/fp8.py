"""FP8 export via llm-compressor's data-free ``model_free_ptq`` (compressed-tensors format).

"fp8" is not one format: we offer two verified presets and record exactly which one ran.
  fp8_block   - FP8 E4M3 weights with 128x128 block scales, dynamic FP8 activations (per token
                group of 128). Default ('auto') — matches the block-wise style of official Qwen FP8
                releases, but is not byte-identical to them.
  fp8_dynamic - FP8 E4M3 weights with per-output-channel scales, dynamic per-token activations.
Both are calibration-free. Modules matching ``quantization.fp8.ignore`` stay in the source dtype.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import ResolvedConfig
from ..fsutil import sha256_obj

SCHEMES = {
    "fp8_block": {
        "preset": "FP8_BLOCK",
        "weights": "float8 e4m3, symmetric, static, block 128x128 scales",
        "activations": "float8 e4m3, symmetric, dynamic, group size 128",
    },
    "fp8_dynamic": {
        "preset": "FP8_DYNAMIC",
        "weights": "float8 e4m3, symmetric, static, per-channel scales",
        "activations": "float8 e4m3, symmetric, dynamic, per-token",
    },
}


def recipe(config: ResolvedConfig) -> dict[str, Any]:
    scheme = config.get("quantization.fp8.scheme")
    scheme = "fp8_block" if scheme == "auto" else scheme
    spec = SCHEMES[scheme]
    ignore = list(config.get("quantization.fp8.ignore"))
    body = {
        "backend": "llmcompressor",
        "entrypoint": "llmcompressor.model_free_ptq",
        "scheme": scheme,
        "preset": spec["preset"],
        "ignore": ignore,
        "device": config.get("quantization.fp8.device"),
        "llmcompressor_version": config.get("quantization.fp8.llmcompressor_version"),
        "compressed_tensors_version": config.get("quantization.fp8.compressed_tensors_version"),
        "calibration": None,
        "scale_format": {"weights": spec["weights"], "activations": spec["activations"]},
    }
    body["recipe_id"] = f"llmcompressor-{body['llmcompressor_version']}:model_free_ptq:{spec['preset']}:{sha256_obj(body)[:10]}"
    return body


def quantize_argv(python_cmd: list[str], script: Path, input_dir: Path, output_dir: Path, recipe_file: Path, result_file: Path) -> list[str]:
    return [*python_cmd, str(script), "--input", str(input_dir), "--output", str(output_dir), "--recipe", str(recipe_file), "--result", str(result_file)]


def check_output_config(config_json: dict[str, Any], expected_preset: str) -> list[str]:
    problems = []
    qc = config_json.get("quantization_config") or (config_json.get("text_config") or {}).get("quantization_config")
    if not qc:
        return ["config.json has no quantization_config"]
    if qc.get("quant_method") != "compressed-tensors":
        problems.append(f"quant_method is {qc.get('quant_method')!r}, expected 'compressed-tensors'")
    groups = qc.get("config_groups") or {}
    if not groups:
        problems.append("quantization_config.config_groups is empty")
    for name, group in groups.items():
        weights = group.get("weights") or {}
        if weights.get("num_bits") != 8 or weights.get("type") != "float":
            problems.append(f"group {name}: weights are not 8-bit float")
        if expected_preset == "FP8_BLOCK" and weights.get("block_structure") != [128, 128]:
            problems.append(f"group {name}: block_structure {weights.get('block_structure')} != [128, 128]")
    return problems
