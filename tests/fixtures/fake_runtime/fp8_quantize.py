"""Fake FP8 quantizer: copies the model, rewrites dtypes to F8_E4M3 and adds a compressed-tensors config. FAKE_FP8_FAIL=1 fails."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "fake_bin"))
from _fakeutil import copy_model, count  # noqa: E402

p = argparse.ArgumentParser()
for flag in ("--input", "--output", "--recipe", "--result"):
    p.add_argument(flag, required=True)
a = p.parse_args()
count("fp8")
recipe = json.loads(Path(a.recipe).read_text())
if os.environ.get("FAKE_FP8_FAIL"):
    Path(a.result).write_text(json.dumps({"ok": False, "error": "fake fp8 failure"}))
    sys.exit(1)
copy_model(Path(a.input), Path(a.output), dtype_override="F8_E4M3")
cfg = json.loads((Path(a.output) / "config.json").read_text())
cfg["quantization_config"] = {
    "quant_method": "compressed-tensors",
    "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "float", "block_structure": [128, 128]}}},
}
(Path(a.output) / "config.json").write_text(json.dumps(cfg))
Path(a.result).write_text(json.dumps({"ok": True, "duration_s": 0.1, "device": recipe["device"], "llmcompressor": recipe["llmcompressor_version"]}))
