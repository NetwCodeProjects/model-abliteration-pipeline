"""Stdlib helpers shared by the fake tool binaries used in CPU tests (never used in production)."""

from __future__ import annotations

import json
import os
import signal
import struct
from pathlib import Path


def count(tool: str) -> None:
    directory = os.environ.get("FAKE_COUNTER_DIR")
    if directory:
        path = Path(directory) / f"{tool}.count"
        n = int(path.read_text()) if path.exists() else 0
        path.write_text(str(n + 1))


def log(tool: str, message: str) -> None:
    directory = os.environ.get("FAKE_COUNTER_DIR")
    if directory:
        with open(Path(directory) / f"{tool}.log", "a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def kill_parent_at(point: str) -> None:
    """FAKE_KILL_PARENT_AT=<point>: hard-kill the pipeline process (our parent) once, then exit.

    Simulates a host crash or OOM kill at a named point: no cleanup, receipts and history entries
    stay as they were. One-shot per FAKE_COUNTER_DIR, so the resumed pipeline is not killed again.
    """
    if os.environ.get("FAKE_KILL_PARENT_AT") != point:
        return
    marker = Path(os.environ.get("FAKE_COUNTER_DIR", ".")) / f"killed-{point.replace(':', '_')}"
    if marker.exists():
        return
    marker.write_text(point)
    # The driver exports its pid: on Windows a venv's python.exe is a launcher, so our parent may not be the pipeline.
    os.kill(int(os.environ.get("FAKE_PIPELINE_PID") or os.getppid()), getattr(signal, "SIGKILL", signal.SIGTERM))
    os._exit(137)


def pareto_front(trials: list[dict]) -> list[dict]:
    """Non-dominated COMPLETE trials sorted by objective values, like Optuna's best_trials + Heretic's menu."""
    done = [t for t in trials if t.get("state") == "COMPLETE"]

    def dominates(a, b):
        return all(x <= y for x, y in zip(a["values"], b["values"], strict=True)) and a["values"] != b["values"]

    front = [t for t in done if not any(dominates(o, t) for o in done)]
    return sorted(front, key=lambda t: tuple(t["values"]))


def read_st(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    (size,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + size])
    return header, data[8 + size :]


def write_st(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, offset = {}, 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(blob)) + blob)
        for _d, _s, raw in tensors.values():
            out.write(raw)


def copy_model(src: Path, dst: Path, drop_prefix: str | None = None, dtype_override: str | None = None, flip: bool = False) -> None:
    """Copy an HF-style model dir, optionally dropping tensors and altering bytes (to fake a transformation)."""
    dst.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for file in sorted(src.iterdir()):
        if file.suffix == ".safetensors":
            header, data = read_st(file)
            tensors = {}
            for name, meta in header.items():
                if name.startswith("__") or (drop_prefix and name.startswith(drop_prefix)):
                    continue
                start, end = meta["data_offsets"]
                raw = data[start:end]
                if flip:
                    raw = bytes((b ^ 0x01) for b in raw)
                tensors[name] = (dtype_override or meta["dtype"], meta["shape"], raw)
                weight_map[name] = file.name
            write_st(dst / file.name, tensors)
        elif file.name == "model.safetensors.index.json":
            continue
        elif file.is_file():
            (dst / file.name).write_bytes(file.read_bytes())
    index = {"metadata": {"total_size": sum((dst / f).stat().st_size for f in set(weight_map.values()))}, "weight_map": weight_map}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))


def minimal_gguf(path: Path, kv: dict) -> None:
    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    body = b""
    for key, value in kv.items():
        body += s(key) + struct.pack("<I", 8) + s(str(value))
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv)) + body)


# ---------------------------------------------------------------------------- generation fakes
BENIGN_FILE = Path(__file__).resolve().parents[3] / "configs" / "qualification" / "benign_prompts.yaml"
REFUSAL = "I'm sorry, but I can't help with that."


def fake_render(prompt: str, enable_thinking: bool) -> str:
    """Qwen-style chat rendering (the real one is the tokenizer's apply_chat_template in qualify.py)."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n")


def heldout_refused(index: int) -> bool:
    return index % 10 == 0


def degraded(model_path: str) -> bool:
    """FAKE_DEGRADE=<substring,...>: exports whose path contains one answer wrongly and refuse every held-out prompt."""
    marks = [m.strip().lower() for m in os.environ.get("FAKE_DEGRADE", "").split(",") if m.strip()]
    parts = Path(model_path).parts
    # Only the part below workspace/exports/ (e.g. "gguf/<family>-<id>-iq3_s"): test tmp paths contain test names.
    tail = "/".join(parts[parts.index("exports") + 1 :]).lower() if "exports" in parts else ""
    return any(m in tail for m in marks)


def fake_answer(rendered: str, bad: bool) -> str:
    """Answer a rendered comparison prompt the way the fake qualification runtime answers it on the baseline."""
    import re

    import yaml

    for item in yaml.safe_load(BENIGN_FILE.read_text(encoding="utf-8"))["prompts"]:
        if item["prompt"] in rendered:
            return "I do not know that one." if bad else f"The answer is {item['expect_any'][0]}."
    match = re.search(r"held-out prompt (\d+)", rendered)
    if bad or (match and heldout_refused(int(match.group(1)))):
        return REFUSAL
    return f"Here is a general, high-level overview of topic {match.group(1) if match else 'x'} with several distinct details."
