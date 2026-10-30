"""Capability policy: what the configured backends need versus what this host offers.

Findings fall into exactly three categories (reported by stage 00, see docs/stages.md):

* ``unsupported_configuration``  - a requested step the configured backend cannot perform here
  (a documented backend requirement); preflight stops before any work.
* ``estimated_resource_shortage`` - estimates from model size, VRAM, disk; a warning, never a verdict.
* ``unverified_execution``       - allowed, but not proven on this hardware/backend combination.

Every rule names the backend requirement it is based on (``basis``). Compute capability is used
only where a backend documents it as the requirement, never as a general compatibility verdict.
"""

from __future__ import annotations

from typing import Any

CATEGORIES = ("unsupported_configuration", "estimated_resource_shortage", "unverified_execution")

VLLM_FP8_BASIS = (
    "vLLM documentation, Quantization > FP8 W8A8: FP8 compute needs NVIDIA compute capability >= 8.9 "
    "(Ada, Hopper, Blackwell); on Ampere (>= 8.0) FP8 checkpoints run weight-only (W8A16) through a "
    "fallback kernel. Observed 2026-10-29 on an A100 (8.0): vLLM 0.32.0 took a JIT weight-only path."
)

# Compute capabilities on which this platform has a recorded real run, and what was covered.
EXERCISED = {
    "8.0": "A100 (2026-10-29): Heretic ARA, qualification, GGUF CUDA build; FP8 loader test not representative",
}


def _capability(gpu: dict[str, Any]) -> tuple[int, int] | None:
    try:
        major, minor = str(gpu.get("compute_capability")).split(".")
        return int(major), int(minor)
    except (TypeError, ValueError):
        return None


def native_fp8(gpus: list[dict[str, Any]]) -> bool | None:
    """True if the first GPU (vLLM's default device) computes in FP8 natively; None if unknown."""
    cap = _capability(gpus[0]) if gpus else None
    return None if cap is None else cap >= (8, 9)


def finding(category: str, rule: str, message: str, basis: str) -> dict[str, str]:
    assert category in CATEGORIES
    return {"category": category, "rule": rule, "message": message, "basis": basis}


def assess(config, gpus: list[dict[str, Any]], *, needs_gpu: bool, heretic: bool, source_size: int | None) -> list[dict[str, str]]:
    findings = []
    formats = config.get("quantization.formats") or []
    fp8_test = config.get("quantization.enabled") and "fp8" in formats and config.get("quantization.fp8.verify_backend") == "vllm"
    cap = _capability(gpus[0]) if gpus else None
    if fp8_test and cap is not None:
        label = f"{gpus[0].get('name')} (compute capability {cap[0]}.{cap[1]})"
        if cap < (8, 0):
            findings.append(
                finding(
                    "unsupported_configuration",
                    "fp8_vllm_compute",
                    f"vLLM cannot load FP8 checkpoints on {label}; set quantization.fp8.verify_backend: none or drop fp8",
                    VLLM_FP8_BASIS,
                )
            )
        elif cap < (8, 9):
            findings.append(
                finding(
                    "unverified_execution",
                    "fp8_vllm_weight_only",
                    f"{label} has no native FP8: the vLLM loader test runs the weight-only fallback, so its result is recorded "
                    "as not representative of native FP8 and does not pass or fail the export",
                    VLLM_FP8_BASIS,
                )
            )
    if needs_gpu:
        for gpu in gpus:
            gcap = _capability(gpu)
            key = f"{gcap[0]}.{gcap[1]}" if gcap else None
            if key not in EXERCISED:
                findings.append(
                    finding(
                        "unverified_execution",
                        "hardware_not_exercised",
                        f"No recorded real run on {gpu.get('name')} (compute capability {key or 'unknown'}); toolchain builds and "
                        "kernels for it are unproven by this platform",
                        f"Exercised so far: {EXERCISED}",
                    )
                )
                break
    if source_size and gpus and heretic:
        vram = max((g.get("memory_total_mib") or 0) for g in gpus) * 1024**2
        if vram and vram < 1.15 * source_size:
            findings.append(
                finding(
                    "estimated_resource_shortage",
                    "vram_vs_checkpoint",
                    f"Largest GPU has {vram / 1024**3:.1f} GiB but the BF16 checkpoint is {source_size / 1024**3:.1f} GiB; "
                    "Heretic will offload or need bnb_4bit loading (which changes the master's provenance)",
                    "Estimate: Heretic loads the full BF16 model plus LoRA adapters and activations on the GPU (factor 1.15)",
                )
            )
    return findings
