"""Load an FP8 (compressed-tensors) checkpoint with vLLM and generate a few greedy completions.

Runs inside the vllm environment. Writes {"ok", "outputs", "load_seconds", ...}. Text-only smoke
test: multimodal inputs are not exercised here, so no image modality is claimed for FP8 exports.

``--suite``: the comparison suite as rendered by the baseline tokenizer ([{"id", "rendered"}]). It is
passed to ``LLM.generate`` as plain text, so vLLM applies no chat template of its own
(docs/assessments.md); results go to "suite" with the same load.
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--suite", help="JSON list of {id, rendered} for the comparison suite")
    parser.add_argument("--suite-max-tokens", type=int, default=128)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    result: dict = {"ok": False, "max_model_len": args.max_model_len}
    try:
        from vllm import LLM, SamplingParams

        result["vllm"] = md.version("vllm")
        started = time.time()
        llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=False)
        result["load_seconds"] = round(time.time() - started, 1)
        started = time.time()
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        outputs = llm.chat(conversations, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
        result["generate_seconds"] = round(time.time() - started, 1)
        result["outputs"] = [o.outputs[0].text for o in outputs]
        result["ok"] = all(text.strip() for text in result["outputs"])
        if args.suite:
            suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
            generated = llm.generate([s["rendered"] for s in suite], SamplingParams(temperature=0.0, max_tokens=args.suite_max_tokens))
            result["suite"] = [
                {
                    "id": s["id"],
                    "response": g.outputs[0].text,
                    "finish_reason": g.outputs[0].finish_reason,
                    "new_tokens": len(g.outputs[0].token_ids),
                    "prompt_tokens": len(g.prompt_token_ids or []),
                }
                for s, g in zip(suite, generated, strict=True)
            ]
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-4000:]
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
