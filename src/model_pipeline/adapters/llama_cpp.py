"""Command construction for llama.cpp tools at the pinned tag (verified flags only).

Verified against b11889 (d854d98): ``convert_hf_to_gguf.py`` options --outfile/--outtype
(f32,f16,bf16,q8_0,tq1_0,tq2_0,auto)/--mmproj; ``Qwen3_5ForConditionalGeneration`` is registered for
both the text model (QWEN35) and the vision projector (qwen3vl); ``llama-quantize [--imatrix f] in
out TYPE [nthreads]``; ``llama-imatrix -m -f -o [--chunks N] [-ngl N]``; ``llama-completion -m -p -n
--temp -s -c -ngl -no-cnv``; ``llama-mtmd-cli -m --mmproj --image -p -n``.
"""

from __future__ import annotations

import os
from pathlib import Path


def convert_argv(convert_cmd: list[str], model_dir: Path, outfile: Path, outtype: str) -> list[str]:
    return [*convert_cmd, str(model_dir), "--outfile", str(outfile), "--outtype", outtype]


def mmproj_argv(convert_cmd: list[str], model_dir: Path, outfile: Path, outtype: str) -> list[str]:
    # With --outfile the converter writes exactly that name (verified on b11889); callers pass an mmproj- name.
    return [*convert_cmd, str(model_dir), "--mmproj", "--outfile", str(outfile), "--outtype", outtype]


def imatrix_argv(imatrix_cmd: list[str], model: Path, calib: Path, out: Path, chunks: int | None, gpu_layers: str) -> list[str]:
    argv = [*imatrix_cmd, "-m", str(model), "-f", str(calib), "-o", str(out), "-ngl", gpu_layers]
    if chunks:
        argv += ["--chunks", str(chunks)]
    return argv


def quantize_argv(quantize_cmd: list[str], src: Path, dst: Path, qtype: str, imatrix: Path | None) -> list[str]:
    argv = [*quantize_cmd]
    if imatrix is not None:
        argv += ["--imatrix", str(imatrix)]
    return [*argv, str(src), str(dst), qtype.upper(), str(max(1, (os.cpu_count() or 2)))]


def completion_argv(cmd: list[str], model: Path, prompt: str, n_predict: int, ctx: int, gpu_layers: str) -> list[str]:
    return [*cmd, "-m", str(model), "-p", prompt, "-n", str(n_predict), "--temp", "0", "-s", "0", "-c", str(ctx), "-ngl", gpu_layers, "-no-cnv"]


def rendered_completion_argv(cmd: list[str], model: Path, prompt_file: Path, n_predict: int, ctx: int, gpu_layers: str) -> list[str]:
    """Generate from text already rendered by the baseline's chat template, byte for byte (d854d98, common/arg.cpp):
    ``-f`` reads the prompt from a file and drops one trailing newline (the caller appends one), ``--no-escape``
    keeps backslash sequences literal (escape processing also applies to -f), ``--no-display-prompt`` keeps
    the prompt out of stdout, and ``-no-cnv`` uses the prompt as is (no chat template)."""
    return [
        *cmd,
        "-m",
        str(model),
        "-f",
        str(prompt_file),
        "--no-escape",
        "--no-display-prompt",
        "-n",
        str(n_predict),
        "--temp",
        "0",
        "-s",
        "0",
        "-c",
        str(ctx),
        "-ngl",
        gpu_layers,
        "-no-cnv",
    ]


EOG_MARKER = "[end of text]"


def generated_text(stdout: str) -> tuple[str, str]:
    """Generated text and finish reason from llama-completion's stdout with --no-display-prompt.

    llama-completion prints " [end of text]" after an end-of-generation token (seen with d854d98 on a real
    Qwen3.5-0.8B GGUF, docs/implementation-log.md); without it the token limit ended generation."""
    text = stdout.rstrip()
    if text.endswith(EOG_MARKER):
        return text[: -len(EOG_MARKER)].rstrip(), "stop"
    return text.strip(), "length"


def mtmd_argv(cmd: list[str], model: Path, mmproj: Path, image: Path, prompt: str, n_predict: int, ctx: int, gpu_layers: str) -> list[str]:
    return [
        *cmd,
        "-m",
        str(model),
        "--mmproj",
        str(mmproj),
        "--image",
        str(image),
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "--temp",
        "0",
        "-s",
        "0",
        "-c",
        str(ctx),
        "-ngl",
        gpu_layers,
    ]


def gguf_name(family: str, tag: str, qtype: str) -> str:
    return f"{family}-{tag}-{qtype.upper()}.gguf"
