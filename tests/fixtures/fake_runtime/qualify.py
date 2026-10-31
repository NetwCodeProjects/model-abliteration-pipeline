"""Fake qualification runtime: canned generations.

FAKE_QUALIFY_REFUSE_ALL=1 makes every held-out prompt a refusal. FAKE_BASELINE_FAIL=1 makes the
comparison baseline (spec purpose "comparison-baseline") fail to load, like the real script, which
writes {"ok": false, "load_ok": false, "error": ...} when from_pretrained raises. Rows carry the
rendered text and prompt token count as scripts/runtime/qualify.py returns them.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "fake_bin"))
from _fakeutil import REFUSAL, count, degraded, fake_render, heldout_refused  # noqa: E402

count("qualify")
spec = json.loads(Path(sys.argv[sys.argv.index("--spec") + 1]).read_text(encoding="utf-8"))
out = Path(sys.argv[sys.argv.index("--out") + 1])
assert Path(spec["model_path"], "config.json").exists(), "model must be reloaded from disk"
if spec.get("purpose") == "comparison-baseline" and os.environ.get("FAKE_BASELINE_FAIL"):
    out.write_text(json.dumps({"ok": False, "load_ok": False, "error": "OSError: fake baseline load failure"}), encoding="utf-8")
    sys.exit(1)
refuse_all = bool(os.environ.get("FAKE_QUALIFY_REFUSE_ALL"))
thinking = bool(spec.get("enable_thinking"))
bad = degraded(spec["model_path"])


def row(prompt: str, response: str, kind: str, **extra) -> dict:
    rendered = fake_render(prompt, thinking)
    return {
        **extra,
        "prompt": prompt,
        "rendered": rendered,
        "prompt_tokens": len(rendered.split()),
        "response": response,
        "finish_reason": "stop",
        "new_tokens": 9,
        "total_tokens": 40,
        "kind": kind,
    }


benign = [
    row(
        b["prompt"],
        f"<think>\n\n</think>\n\nThe answer is {(b.get('expect_any') or ['ok'])[0]}." if not bad else "I do not know that one.",
        "benign",
        **{k: v for k, v in b.items() if k != "prompt"},
    )
    for b in spec["benign"]
]
n = spec["heldout_limit"] if spec["heldout_limit"] is not None else 20
heldout = []
for i in range(n):
    refused = refuse_all or bad or heldout_refused(i)
    text = REFUSAL if refused else f"Here is a general, high-level overview of topic {i} with several distinct details."
    heldout.append(row(f"held-out prompt {i}", text, "heldout_refusal", id=f"heldout-{i}"))
result = {
    "ok": True,
    "load_ok": True,
    "load_seconds": 0.1,
    "dtype": "torch.bfloat16",
    "benign": benign,
    "heldout_refusal": heldout,
    "datasets": [{"dataset": s["dataset"], "commit": s["commit"], "split": s["split"]} for s in spec.get("heldout_refusal_specs") or []],
    "chat_template_sha256": "0" * 64,
    "max_total_tokens": 60,
}
if spec.get("extended_kl"):
    result["extended_kl"] = {"value": 0.02, "prompts": 100, "definition": "fake"}
out.write_text(json.dumps(result), encoding="utf-8")
