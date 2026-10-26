"""Validate a provisioned tool environment (runs *inside* that environment).

Prints one JSON line {"ok": bool, ...}. With --gpu-smoke it also runs a small BF16 matmul on CUDA,
which is the minimum proof that the torch build supports the installed GPU (e.g. B200 = sm_100).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import platform
import sys
import traceback


def version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", required=True, choices=["heretic", "fp8", "vllm"])
    parser.add_argument("--gpu-smoke", action="store_true")
    args = parser.parse_args()
    out: dict = {"ok": False, "component": args.component, "python": platform.python_version()}
    try:
        import torch

        out["torch"] = torch.__version__
        out["torch_cuda"] = torch.version.cuda
        out["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            out["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            out["capability"] = list(torch.cuda.get_device_capability(0))
            out["arch_list"] = torch.cuda.get_arch_list()
            major, minor = torch.cuda.get_device_capability(0)
            if f"sm_{major}{minor}" not in out["arch_list"] and f"compute_{major}{minor}" not in out["arch_list"]:
                out["warning"] = f"torch was not built with sm_{major}{minor}; kernels may JIT from PTX or fail"
        if args.gpu_smoke:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available to torch (CPU-only wheel or driver mismatch)")
            a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
            b = a @ a
            torch.cuda.synchronize()
            if not bool(torch.isfinite(b).all()):
                raise RuntimeError("non-finite result in GPU smoke matmul")
            out["gpu_smoke"] = "bf16 matmul ok"
        if args.component == "heretic":
            out["heretic-llm"] = version("heretic-llm")
            out["transformers"] = version("transformers")
            importlib.import_module("heretic.config")
            importlib.import_module("heretic.modifiers.ara")
            out["ara_importable"] = True
        elif args.component == "fp8":
            out["llmcompressor"] = version("llmcompressor")
            out["compressed-tensors"] = version("compressed-tensors")
            from llmcompressor import model_free_ptq  # noqa: F401

            out["model_free_ptq"] = True
        elif args.component == "vllm":
            out["vllm"] = version("vllm")
            importlib.import_module("vllm")
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001 - reported to the orchestrator
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-2000:]
    print(json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
