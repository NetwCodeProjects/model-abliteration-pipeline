"""Console banner: shown at the start of every run and dry run, and as the welcome screen of a bare
``./main.sh`` (no arguments). Stdlib only, so the welcome screen works before ``.venv`` exists."""

from __future__ import annotations

from typing import Any

LOGO = r"""
 _     _     __  __   ____  _   _ ____   ____ _____ ___  _   _
| |   | |   |  \/  | / ___|| | | |  _ \ / ___| ____/ _ \| \ | |
| |   | |   | |\/| | \___ \| | | | |_) | |  _|  _|| | | |  \| |
| |___| |___| |  | |  ___) | |_| |  _ <| |_| | |__| |_| | |\  |
|_____|_____|_|  |_| |____/ \___/|_| \_\\____|_____\___/|_| \_|
""".strip("\n")
TAGLINE = "MODEL ABLATION & QUANTIZATION"
WIDTH = 64
WELCOME = """
 Welcome to LLM Surgeon.

 Review:   configs/pipeline.yaml
 Preview:  ./main.sh --dry-run
 Help:     ./main.sh --help
 Run:      ./main.sh --yes
""".strip("\n")


def gpu_label(gpus: list[dict[str, Any]] | None, probed: bool = True) -> str:
    if not probed:
        return "not probed (dry run)"
    if not gpus:
        return "none detected (nvidia-smi unavailable or no GPU)"
    names = [g.get("name") or "unknown" for g in gpus]
    first = names[0]
    if all(n == first for n in names):
        return first if len(names) == 1 else f"{first} x{len(names)}"
    return ", ".join(names)


def render(version: str, gpu: str, welcome: bool = False) -> str:
    version = f"v{version}"
    lines = [LOGO, "", f" {TAGLINE}{version:>{WIDTH - len(TAGLINE)}}", "", f" GPU type: {gpu}", "", " " + "-" * WIDTH]
    if welcome:
        lines += ["", WELCOME]
    return "\n".join(lines) + "\n"
