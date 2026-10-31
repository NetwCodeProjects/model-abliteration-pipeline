"""Generate responses for qualification / smoke tests (runs inside the heretic environment).

Loads the checkpoint from disk with transformers (a real reload of what Heretic saved), applies the
model's chat template (thinking mode as configured), decodes greedily and writes raw generations.
Scoring happens in the control plane (llm_surgeon.evaluation.qualification) so it is unit-tested.
Optional extended KL: mean KL(source || candidate) of next-token distributions on held-out prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

SLICE_RE = re.compile(r"^\[(-?\d*):(-?\d*)\]$")


def load_model(path: str, trust: bool):
    config = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    cls = AutoModelForImageTextToText if "vision_config" in config else AutoModelForCausalLM
    model = cls.from_pretrained(path, dtype="auto", device_map="auto", trust_remote_code=trust)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_prompts(spec: dict, limit: int | None) -> tuple[list[str], dict]:
    name = spec["dataset"]
    if Path(name).is_file():
        lines = [line.strip() for line in Path(name).read_text(encoding="utf-8").splitlines() if line.strip()]
        split = spec.get("split")
        if split:
            m = SLICE_RE.match(split)
            if m:
                start = int(m.group(1)) if m.group(1) else None
                stop = int(m.group(2)) if m.group(2) else None
                lines = lines[start:stop]
        info = {"dataset": name, "local": True, "sha256": hashlib.sha256(Path(name).read_bytes()).hexdigest()}
        prompts = lines
    else:
        import datasets

        ds = datasets.load_dataset(name, spec.get("config"), split=spec["split"], revision=spec.get("commit"))
        prompts = [str(x) for x in ds[spec["column"]]]
        info = {"dataset": name, "commit": spec.get("commit"), "split": spec["split"], "rows": len(prompts)}
    if limit is not None:
        prompts = prompts[:limit]
    return prompts, info


def render(tokenizer, prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def generate(model, tokenizer, prompts: list[str], max_new_tokens: int, enable_thinking: bool, batch_size: int = 8) -> list[dict]:
    rows = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        texts = [render(tokenizer, p, enable_thinking) for p in chunk]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        try:
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            finish_error = None
        except RuntimeError as exc:
            if "probability tensor" in str(exc) or "nan" in str(exc).lower():
                out, finish_error = None, "nan"
            else:
                raise
        for i, prompt in enumerate(chunk):
            # rendered/prompt_tokens: the comparison feeds this exact text to the other backends (docs/assessments.md)
            n_prompt = int(inputs["attention_mask"][i].sum())
            if out is None:
                rows.append(
                    {
                        "prompt": prompt,
                        "rendered": texts[i],
                        "prompt_tokens": n_prompt,
                        "response": "",
                        "finish_reason": finish_error,
                        "new_tokens": 0,
                        "total_tokens": None,
                    }
                )
                continue
            new = out[i, inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(new, skip_special_tokens=True)
            n_new = int((new != tokenizer.pad_token_id).sum())
            rows.append(
                {
                    "prompt": prompt,
                    "rendered": texts[i],
                    "prompt_tokens": n_prompt,
                    "response": text,
                    "finish_reason": "length" if n_new >= max_new_tokens else "stop",
                    "new_tokens": n_new,
                    "total_tokens": n_prompt + n_new,
                }
            )
    return rows


@torch.no_grad()
def next_token_logprobs(model, tokenizer, prompts: list[str], enable_thinking: bool) -> torch.Tensor:
    result = []
    for prompt in prompts:
        inputs = tokenizer([render(tokenizer, prompt, enable_thinking)], return_tensors="pt").to(model.device)
        logits = model(**inputs).logits[0, -1].float()
        result.append(F.log_softmax(logits, dim=-1).cpu())
    return torch.stack(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    torch.manual_seed(int(spec.get("seed") or 0))
    out: dict = {"ok": False, "load_ok": False, "transformers": transformers.__version__, "torch": torch.__version__}
    try:
        started = time.time()
        model, tokenizer = load_model(spec["model_path"], bool(spec.get("trust_remote_code")))
        out.update(load_ok=True, load_seconds=round(time.time() - started, 1), dtype=str(model.dtype))
        template = getattr(tokenizer, "chat_template", None) or ""
        out["chat_template_sha256"] = hashlib.sha256(template.encode()).hexdigest() if template else None
        thinking = bool(spec.get("enable_thinking"))
        benign = spec.get("benign") or []
        rows = generate(model, tokenizer, [b["prompt"] for b in benign], int(spec["max_new_tokens"]), thinking)
        out["benign"] = [{**b, **r, "kind": "benign"} for b, r in zip(benign, rows, strict=True)]
        heldout, datasets_info = [], []
        remaining = spec.get("heldout_limit")
        for ds_spec in spec.get("heldout_refusal_specs") or []:
            if remaining is not None and remaining <= 0:
                break
            prompts, info = load_prompts(ds_spec, remaining)
            datasets_info.append(info)
            heldout.extend(prompts)
            if remaining is not None:
                remaining -= len(prompts)
        rows = generate(model, tokenizer, heldout, int(spec["max_new_tokens"]), thinking)
        out["heldout_refusal"] = [{"id": f"heldout-{i}", **r, "kind": "heldout_refusal"} for i, r in enumerate(rows)]
        out["datasets"] = datasets_info
        totals = [r.get("total_tokens") or 0 for r in out["benign"] + out["heldout_refusal"]]
        out["max_total_tokens"] = max(totals) if totals else None
        if spec.get("extended_kl"):
            kl_prompts = []
            for ds_spec in spec.get("heldout_harmless_specs") or []:
                prompts, info = load_prompts(ds_spec, None)
                kl_prompts.extend(prompts)
                datasets_info.append(info)
            cand = next_token_logprobs(model, tokenizer, kl_prompts, thinking)
            del model
            torch.cuda.empty_cache()
            source, source_tok = load_model(spec["source_path"], bool(spec.get("trust_remote_code")))
            src = next_token_logprobs(source, source_tok, kl_prompts, thinking)
            kl = F.kl_div(cand, src, reduction="batchmean", log_target=True).item()
            out["extended_kl"] = {"value": kl, "prompts": len(kl_prompts), "definition": "KL(source || candidate) of next-token distributions, batch mean"}
        if torch.cuda.is_available():
            out["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-4000:]
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
