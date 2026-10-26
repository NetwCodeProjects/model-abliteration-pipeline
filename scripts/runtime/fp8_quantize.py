"""Data-free FP8 quantization with llm-compressor's model_free_ptq (runs inside the fp8 environment).

Reads the recipe written by stage 08 and records what actually ran. model_free_ptq copies all
non-safetensors files (config, tokenizer, processor, chat template) and writes compressed-tensors
weights plus an updated config/index.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    result: dict = {"ok": False, "device": recipe.get("device")}
    try:
        import torch
        from llmcompressor import model_free_ptq

        result["llmcompressor"] = md.version("llmcompressor")
        result["compressed-tensors"] = md.version("compressed-tensors")
        if result["llmcompressor"] != recipe["llmcompressor_version"]:
            raise RuntimeError(f"llmcompressor {result['llmcompressor']} installed, recipe requires {recipe['llmcompressor_version']}")
        started = time.time()
        model_free_ptq(
            model_stub=args.input,
            save_directory=args.output,
            scheme=recipe["preset"],
            ignore=recipe["ignore"],
            device=recipe["device"],
        )
        result["duration_s"] = round(time.time() - started, 1)
        if torch.cuda.is_available():
            result["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-4000:]
    Path(args.result).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
