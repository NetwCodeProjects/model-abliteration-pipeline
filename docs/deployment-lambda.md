# Deployment on Lambda Cloud (one B200)

## Choosing the profile

| You launched… | Use | What happens |
|---|---|---|
| A Lambda Stack image (default on Lambda Cloud) | `--lambda-stack` (or `auto`) | Driver/CUDA from Lambda Stack are reused; nothing is installed system-wide; Python tools go into project venvs. A missing driver is a blocker (this profile never installs drivers or Lambda Stack). |
| A plain Ubuntu image | `--ubuntu` | Missing packages are reported with the exact `apt-get` command; installed only with `environment.allow_system_packages: true` + confirmation; drivers only with `allow_driver_changes: true` and never with an automatic reboot. |

`auto` selects `lambda-stack` **only** when Lambda Stack packaging markers exist (`dpkg` packages
`lambda-stack*`/`lambda-server*` or a `lambda` apt source); a working CUDA install alone is not taken as
evidence. Non-Ubuntu hosts require an explicit choice.

## Preflight on a B200

```bash
nvidia-smi                               # expect "NVIDIA B200", driver R570+ (open kernel modules), CUDA 12.8+
./main.sh --lambda-stack --dry-run       # plan, components, missing host tools, compatibility notice
./main.sh --lambda-stack --through 0 --yes && cat workspace/runs/*/metrics/preflight.json
```

Preflight records GPU name, memory, driver, compute capability (`10.0` → llama.cpp built for `sm_100`),
uncorrected ECC counters, CPU/RAM, mounts and free space. Stage 01 then proves the Heretic/FP8/vLLM
environments can run a BF16 matmul on the GPU (catches CPU-only torch wheels and sm_100-less builds).

## Storage persistence

Lambda instance disks are lost when the instance is **terminated**. Preflight classifies the workspace
mount (`instance-local` vs network/persistent) and warns. Options:

- Put `storage.root` (`--output-dir`) on a Lambda persistent filesystem (`/lambda/nfs/<name>`), at the
  cost of slower I/O for 55 GB checkpoints; or
- keep the workspace local and copy `workspace/exports/` + `workspace/manifests/` off the instance when
  done (`rsync -a --info=progress2`). Manifests contain all hashes for re-verification after copying.

Budget roughly 230–260 GB free for the default Qwen3.8-27B run (source 55.6 GB, candidate ~55 GB,
master hardlinked, FP8 ~30 GB, GGUF BF16 intermediate ~55 GB, IQ4_XS ~15 GB, IQ3_S ~12 GB) plus caches.

## Secrets

- `export HF_TOKEN=…` (or another variable named by `model.hf_token_env`). The value is passed to child
  processes in their environment only, never written to configs/receipts, and redacted from all logs.
- Nothing is uploaded: Heretic's `upload` action is never selected and reproduce files are patched to
  clear upload settings.

## Commands

```bash
git clone <repo> && cd model-abliteration-pipeline
./main.sh --lambda-stack --setup-only --yes                    # ~10-20 min: venvs, CUDA torch, llama.cpp (sm_100) build
./main.sh --lambda-stack --yes                                 # full run (optimize: 50 trials / 180 min budget)
./main.sh --config examples/qwen-b200.yaml --yes               # approximate reproduction of the ARA reference
./main.sh --lambda-stack --through 5 --yes                     # stop after abliteration
./main.sh --resume <RUN_ID>                                    # resume after interruption (study continues)
pytest -m gpu tests/gpu -x                                     # GPU smoke tests (bounded, explicit)
tmux new -s pipeline                                           # recommended: run inside tmux/screen
```

Cost estimates: set `execution.hourly_rate_usd` (e.g. the instance's hourly price); the report multiplies
*measured stage time* by it and labels the result as an estimate (idle time is not included).
