"""Master vs export comparison: suite, generation contract, scoring and deltas (docs/assessments.md).

Every backend receives the prompts rendered once by the baseline tokenizer, so chat-template
rendering cannot differ between backends. A comparison without a measured baseline or export is
``unmeasured``: deltas are never reported as zero and checks never pass without a measurement.
"""

from __future__ import annotations

from typing import Any

from ..artifacts.assessments import check, outcome_of
from ..fsutil import sha256_obj, sha256_text
from .qualification import SCORER_ID, integrity_issues, is_refusal, load_benign_prompts, visible_text

SUITE_ID = "comparison"
SUITE_VERSION = "1"
STOP = "end-of-sequence tokens of the model only; no stop strings"
TRUNCATION = "none: a backend whose configured context is smaller than required gets a contract gap"
LLAMA_CPP_BASIS = "llama.cpp d854d98 tools/completion/completion.cpp"
# Contract fields a backend cannot honor, independent of the run (docs/evaluation.md#what-each-backend-cannot-honor).
STATIC_GAPS: dict[str, list[dict[str, str]]] = {
    "transformers": [],
    "vllm": [],
    "llama.cpp": [
        {
            "field": "tokenization",
            "detail": "the rendered text is tokenized from the GGUF vocabulary (special tokens parsed, BOS per GGUF metadata), not by the Hugging Face tokenizer",
            "basis": f"{LLAMA_CPP_BASIS}: common_tokenize(ctx, prompt, true, true); add_bos = llama_vocab_get_add_bos(vocab)",
        },
        {
            "field": "stop",
            "detail": "generation stops at the GGUF's end-of-generation tokens, which can include tokens generation_config.json does not list",
            "basis": f"{LLAMA_CPP_BASIS}: llama_vocab_is_eog",
        },
    ],
}


def suite_spec(config) -> dict[str, Any]:
    fast = [p for p in load_benign_prompts(config.get("evaluation.benign_prompts_file")) if p.get("category") == "fast"]
    return {
        "benign": fast[: int(config.get("evaluation.comparison.benign_prompts"))],
        "heldout_refusal_specs": config.get("evaluation.heldout_refusal_prompts"),
        "heldout_limit": int(config.get("evaluation.comparison.heldout_refusal_prompts")),
        "max_new_tokens": int(config.get("evaluation.comparison.max_new_tokens")),
        "enable_thinking": bool(config.get("evaluation.enable_thinking")),
        "markers": [m.lower() for m in config.get("evaluation.refusal_markers")],
    }


def suite_ref(spec: dict[str, Any]) -> dict[str, str]:
    content = {
        "benign": [[p["id"], sha256_text(p["prompt"]), p["expect_any"]] for p in spec["benign"]],
        "heldout": [spec["heldout_refusal_specs"], spec["heldout_limit"]],
        "markers": spec["markers"],
        "scorer": SCORER_ID,
    }
    return {"id": SUITE_ID, "version": SUITE_VERSION, "sha256": sha256_obj(content)}


def text_fields(name: str, text: str | None, redact: bool) -> dict[str, Any]:
    text = text or ""
    fields: dict[str, Any] = {f"{name}_sha256": sha256_text(text), f"{name}_chars": len(text)}
    if not redact:
        fields[name] = text
    return fields


def rendered_prompts(baseline_raw: dict[str, Any]) -> list[dict[str, Any]]:
    """The suite as the baseline rendered it: id, kind, prompt and rendered text (never written when redacted)."""
    rows = baseline_raw.get("benign", []) + baseline_raw.get("heldout_refusal", [])
    return [
        {"id": r["id"], "kind": r["kind"], "prompt": r.get("prompt", ""), "rendered": r["rendered"], "prompt_tokens": r.get("prompt_tokens")}
        for r in rows
        if r.get("rendered")
    ]


def contract(spec: dict[str, Any], baseline_raw: dict[str, Any] | None, redact: bool) -> dict[str, Any]:
    prompts = rendered_prompts(baseline_raw or {})
    longest = max((p["prompt_tokens"] or 0 for p in prompts), default=None)
    body = {
        "chat_template": {
            "rendered_by": "baseline tokenizer: transformers apply_chat_template(add_generation_prompt=True)",
            "template_sha256": (baseline_raw or {}).get("chat_template_sha256"),
        },
        "thinking_mode": spec["enable_thinking"],
        "max_new_tokens": spec["max_new_tokens"],
        "stop_sequences": [],
        "stop": STOP,
        "decoding": {"strategy": "greedy", "temperature": 0, "seed": spec.get("seed", 0)},
        "context": {
            "max_prompt_tokens": longest,
            "required_tokens": longest + spec["max_new_tokens"] if longest is not None else None,
            "truncation": TRUNCATION,
        },
    }
    rows = [{"id": p["id"], "kind": p["kind"], **text_fields("prompt", p["prompt"], redact), **text_fields("rendered", p["rendered"], redact)} for p in prompts]
    return {"contract_id": sha256_obj({**body, "prompts": [[r["id"], r["rendered_sha256"]] for r in rows]}), **body, "prompts": rows}


def backend_gaps(backend: str, context_limit: int | None, required_tokens: int | None) -> list[dict[str, str]]:
    gaps = [dict(g) for g in STATIC_GAPS.get(backend, [])]
    if context_limit is not None and required_tokens is not None and context_limit < required_tokens:
        gaps.append(
            {
                "field": "context",
                "detail": f"configured context {context_limit} < required {required_tokens} tokens"
                + ("; llama.cpp shifts the context instead of failing" if backend == "llama.cpp" else ""),
                "basis": "contract.context.required_tokens",
            }
        )
    return gaps


def score(rows: list[dict[str, Any]] | None, spec: dict[str, Any], loaded: bool, redact: bool, reason: str | None = None) -> dict[str, Any]:
    """Counts with denominators for one runtime. ``rows``: id, kind, response, finish_reason (``None`` = did not run)."""
    if rows is None or not loaded:
        return {"measured": False, "loaded": loaded, "reason": reason or ("did not load" if rows is not None else "did not run")}
    expect = {p["id"]: [e.lower() for e in p["expect_any"]] for p in spec["benign"]}
    per_prompt, correct, graded, refused, heldout = [], 0, 0, 0, 0
    for row in rows:
        response = row.get("response") or ""
        issues = integrity_issues(response, row.get("finish_reason"))
        item: dict[str, Any] = {"id": row["id"], "kind": row["kind"], "integrity_issues": issues, **text_fields("response", response, redact)}
        if row["kind"] == "benign":
            wanted = expect.get(row["id"]) or []
            item["correct"] = any(e in visible_text(response).lower() for e in wanted) if wanted else None
            if wanted:
                graded += 1
                correct += bool(item["correct"])
        else:
            item["refused"] = is_refusal(response, spec["markers"])
            heldout += 1
            refused += item["refused"]
        per_prompt.append(item)
    return {
        "measured": True,
        "loaded": True,
        "benign_correct": {"count": correct, "of": graded},
        "heldout_refused": {"count": refused, "of": heldout},
        "integrity_issues": {"count": sum(1 for p in per_prompt if p["integrity_issues"]), "of": len(per_prompt)},
        "per_prompt": per_prompt,
    }


def _rate(pair: dict[str, int]) -> float | None:
    return pair["count"] / pair["of"] if pair["of"] else None


def _delta(name: str, base: dict[str, Any], exp: dict[str, Any]) -> dict[str, Any]:
    b, e = base[name], exp[name]
    rb, re_ = _rate(b), _rate(e)
    return {
        "baseline": f"{b['count']}/{b['of']}",
        "export": f"{e['count']}/{e['of']}",
        "delta": e["count"] - b["count"] if b["of"] == e["of"] else None,
        "rate_delta": round(re_ - rb, 6) if rb is not None and re_ is not None else None,
    }


def compare(baseline: dict[str, Any] | None, export: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    """Deltas (export minus baseline) and export-gate checks; everything is unmeasured without both sides."""
    loaded = (
        check("export_loaded", "passed", True)
        if export.get("measured")
        else check("export_loaded", "failed" if export.get("reason") == "did not load" else "unmeasured", False, reason=export.get("reason"))
    )
    if not (baseline and baseline.get("measured") and export.get("measured")):
        side = export if baseline and baseline.get("measured") else baseline
        why = "export not measured" if side is export else "no baseline measurement"
        if side and side.get("reason"):
            why += f" ({side['reason']})"
        checks = [loaded] + [check(name, "unmeasured", reason=why) for name in ("export_integrity", "benign_correct_drop", "heldout_refusal_increase")]
        return {"deltas": None, "unmeasured_reason": why, "checks": checks, "outcome": outcome_of(checks)}
    deltas = {name: _delta(name, baseline, export) for name in ("benign_correct", "heldout_refused", "integrity_issues")}
    clean = {p["id"] for p in baseline["per_prompt"] if not p["integrity_issues"]}
    new_issues = sorted(p["id"] for p in export["per_prompt"] if p["integrity_issues"] and p["id"] in clean)
    checks = [loaded]
    if gates.get("require_integrity"):
        checks.append(check("export_integrity", "failed" if new_issues else "passed", new_issues, "no new integrity issues"))
    else:
        checks.append(check("export_integrity", "disabled", new_issues, None))
    for name, key, sign in (("benign_correct_drop", "benign_correct", -1), ("heldout_refusal_increase", "heldout_refused", 1)):
        threshold = gates.get("max_" + name)
        rate_delta = deltas[key]["rate_delta"]
        if threshold is None:
            checks.append(check(name, "disabled", None, None))
        elif rate_delta is None:
            checks.append(check(name, "unmeasured", None, threshold, reason=f"no {key.replace('_', ' ')} prompts in the suite"))
        else:
            value = round(sign * rate_delta, 6)
            checks.append(check(name, "failed" if value > threshold + 1e-9 else "passed", value, threshold))
    return {"deltas": deltas, "new_integrity_issues": new_issues, "checks": checks, "outcome": outcome_of(checks)}
