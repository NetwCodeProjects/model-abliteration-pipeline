# Deployment on plain Ubuntu (22.04 / 24.04)

## Prerequisites

| Need | Package / source | Required for |
|---|---|---|
| Python ≥ 3.10 + `python3-venv` (or `uv`) | `python3`, `python3-venv` | control `.venv` bootstrap |
| `python3-yaml` (optional) | apt | `--dry-run` before bootstrapping |
| git | `git` | Heretic and llama.cpp sources |
| cmake, C/C++ compiler, make | `cmake`, `build-essential` | llama.cpp build (GGUF) |
| NVIDIA driver | NVIDIA datacenter driver (B200: R570+, open kernel modules) | Heretic, FP8, vLLM |
| CUDA toolkit `nvcc` (optional) | NVIDIA CUDA repo | CUDA build of llama.cpp (CPU build otherwise) |

## Guarded host steps

Stage 01 reports missing packages and prints the exact command, e.g.

```text
ERROR [E_HOST_PREREQ] Missing host prerequisites: cmake, c++ compiler
  hint: Run manually:
  sudo -n apt-get install -y --no-install-recommends build-essential cmake
or set environment.allow_system_packages: true and confirm with --yes.
```

With `environment.allow_system_packages: true` **and** `--yes` (or an interactive `y`), stage 01 runs
that command itself via `sudo -n` (non-interactive; it fails cleanly if sudo needs a password — run it
yourself in that case). Nothing else is upgraded.

Drivers: with `environment.allow_driver_changes: true`, the ubuntu profile may run
`sudo -n ubuntu-drivers install --gpgpu`, then stops with `E_REBOOT_REQUIRED`. **v1 never reboots**;
reboot yourself and `./main.sh --resume RUN_ID`. Recommended manual route for B200: install NVIDIA's
datacenter driver ≥ 570 with the open kernel modules from NVIDIA's CUDA repository and verify with
`nvidia-smi` before running the pipeline.

## CUDA / PyTorch selection

Tool venvs install torch with `uv pip install --torch-backend auto`, which picks the CUDA wheel matching
the installed driver; Heretic's torch/torchvision versions are pinned to Heretic's own `uv.lock`.
Override with `environment.torch_backend: cu130` (or `cpu` for CPU-only experiments; this also skips the
GPU smoke validation). Stage 01 refuses an environment whose torch cannot run a CUDA matmul.

llama.cpp is built with CUDA when `nvcc` is found (`quantization.gguf.build_cuda: auto`). Old toolkits
paired with new compilers fail (e.g. CUDA 12.0 + GCC 13 on Ubuntu 24.04): set `build_cuda: off` — GGUF
conversion and quantization do not need a GPU, only the speed of smoke tests/imatrix changes.

## Commands

```bash
./main.sh --ubuntu --dry-run
./main.sh --ubuntu --setup-only --yes
./main.sh --config examples/ubuntu-quantize-only.yaml --ubuntu --yes
```

## Troubleshooting quick list

- `E_UV_MISSING`: bootstrap first (`./main.sh --bootstrap --yes`), the control venv ships uv.
- `E_BOOTSTRAP_IMPOSSIBLE`: `sudo apt-get install -y python3-venv`.
- `E_COMPONENT_VALIDATION_FAILED` with "CUDA is not available": driver/wheel mismatch; check `nvidia-smi`
  CUDA version and set `environment.torch_backend` explicitly.
- See [troubleshooting.md](troubleshooting.md).
