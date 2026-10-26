"""Fake llama.cpp tools for CPU tests: fake_llama.py <convert|quantize|imatrix|completion|mtmd> ARGS.

FAKE_QUANT_FAIL=<TYPE> makes llama-quantize fail for that type; FAKE_MMPROJ_FAIL=1 breaks --mmproj.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _fakeutil import count, minimal_gguf  # noqa: E402


def arg(flag: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1]


def main() -> int:
    tool = sys.argv[1]
    rest = sys.argv[2:]
    if "--version" in rest:
        print("version: 11889 (d854d98) fake")
        return 0
    if "--help" in rest:
        print(f"usage: llama-{tool} [options] (fake)")
        return 1
    if "--print-supported-models" in rest:
        print("Qwen3_5ForConditionalGeneration")
        return 0
    count(f"llama-{tool}")
    if tool == "convert":
        out = Path(arg("--outfile"))
        if "--mmproj" in rest:
            if os.environ.get("FAKE_MMPROJ_FAIL"):
                print("NotImplementedError: vision tower not supported", file=sys.stderr)
                return 1
            minimal_gguf(out.parent / ("mmproj-" + out.name), {"general.architecture": "clip", "general.type": "mmproj"})
        else:
            minimal_gguf(
                out,
                {
                    "general.architecture": "qwen35",
                    "tokenizer.ggml.model": "gpt2",
                    "tokenizer.chat_template": "{{ messages }}",
                    "general.file_type": arg("--outtype"),
                },
            )
        return 0
    if tool == "quantize":
        args = [a for a in rest if a != "--imatrix"]
        if "--imatrix" in rest:
            args.remove(arg("--imatrix"))
        src, dst, qtype = Path(args[0]), Path(args[1]), args[2]
        if os.environ.get("FAKE_QUANT_FAIL", "").upper() == qtype.upper():
            print(f"llama_model_quantize: failed to quantize: {qtype} (fake failure)", file=sys.stderr)
            return 1
        minimal_gguf(
            dst,
            {
                "general.architecture": "qwen35",
                "tokenizer.ggml.model": "gpt2",
                "tokenizer.chat_template": "{{ messages }}",
                "general.file_type": qtype,
                "fake.source_size": str(src.stat().st_size),
            },
        )
        return 0
    if tool == "imatrix":
        shutil.copyfile(arg("-f"), arg("-o"))
        return 0
    if tool == "completion":
        print(arg("-p") + " The answer is Paris, and 17 * 23 = 391.")
        return 0
    if tool == "mtmd":
        print("Red.")
        return 0
    print(f"unknown fake tool {tool}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
