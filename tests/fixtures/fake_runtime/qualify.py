"""Fake qualification runtime: canned generations. FAKE_QUALIFY_REFUSE_ALL=1 makes every held-out prompt a refusal."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "fake_bin"))
from _fakeutil import count  # noqa: E402

count("qualify")
spec = json.loads(Path(sys.argv[sys.argv.index("--spec") + 1]).read_text(encoding="utf-8"))
out = Path(sys.argv[sys.argv.index("--out") + 1])
assert Path(spec["model_path"], "config.json").exists(), "model must be reloaded from disk"
refuse_all = bool(os.environ.get("FAKE_QUALIFY_REFUSE_ALL"))
benign = [
    {
        **b,
        "response": f"<think>\n\n</think>\n\nThe answer is {(b.get('expect_any') or ['ok'])[0]}.",
        "finish_reason": "stop",
        "new_tokens": 9,
        "total_tokens": 40,
        "kind": "benign",
    }
    for b in spec["benign"]
]
n = spec["heldout_limit"] if spec["heldout_limit"] is not None else 20
heldout = []
for i in range(n):
    refused = refuse_all or i % 10 == 0
    text = "I'm sorry, but I can't help with that." if refused else f"Here is a general, high-level overview of topic {i} with several distinct details."
    heldout.append(
        {
            "id": f"heldout-{i}",
            "prompt": f"held-out prompt {i}",
            "response": text,
            "finish_reason": "stop",
            "new_tokens": 20,
            "total_tokens": 60,
            "kind": "heldout_refusal",
        }
    )
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
