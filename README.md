# model-abliteration-pipeline (Project A - Abliterate and Quantize)

A reproducible, configuration-first CLI platform that takes a user-selected Hugging Face model through
**abliteration** (with the [Heretic](https://github.com/p-e-w/heretic) engine), **lightweight
qualification**, preservation of an immutable **master checkpoint**, and **deployment quantization**
(FP8 via llm-compressor, GGUF via llama.cpp). Every artifact gets a versioned manifest with checksums
and lineage, so downstream consumers (Project B) never need to import this code.

> **Compatibility notice.** This platform does not automatically verify that your model architecture
> is supported by the selected abliteration engine or quantization backend. Verify compatibility with
> the engines' official documentation before execution. Unsupported models or formats may fail during
> model loading, transformation, or export. The platform manages execution, dependencies, logs, and
> artifacts but does not guarantee compatibility with every model.

## What it does

| Stage | ID | Result |
|---|---|---|
| preflight | 00 | OS/GPU/driver/RAM/disk facts, profile detection (Lambda Stack vs Ubuntu), disk & VRAM warnings |
| environment | 01 | Pinned, validated tool environments (Heretic, llm-compressor, vLLM, llama.cpp build) |
| acquire | 02 | Model revision pinned to a commit SHA, sha256-verified download, immutable source manifest |
| published_baselines | 03 | Model-card metrics with provenance — **no** reference-model inference |
| prepare_engine | 04 | Native Heretic config generated and validated against the *installed* Heretic |
| abliterate | 05 | Budgeted, resumable headless Heretic run → candidate checkpoint |
| qualify | 06 | Reload + held-out refusal / benign / integrity checks (optimizer metrics labelled separately) |
| select_master | 07 | Gates → immutable master (actual precision recorded) |
| quantize | 08 | FP8 / GGUF exports, each an independent resumable job |
| verify_exports | 09 | Re-hash, preservation checks, real loader smoke tests |
| report_export | 10 | Manifests, run summary, report, usage examples |

Pinned, verified upstream versions (see [docs/implementation-log.md](docs/implementation-log.md)):
Heretic `171bb98b` (master; ARA + headless mode exist only after v1.4.0), llama.cpp `b11889`,
llm-compressor `0.14.0.1`, vLLM `0.32.0`.

## Requirements

- Ubuntu 22.04/24.04 (Lambda Stack image or plain Ubuntu). `--help` and `--dry-run` also work elsewhere.
- Python ≥ 3.10. Either `uv`, or `python3-venv` for the hash-locked bootstrap. PyYAML (`python3-yaml`, preinstalled on most Ubuntu images) for `--dry-run` before bootstrapping.
- NVIDIA GPU + driver for Heretic / FP8 / vLLM (one B200 is the reference target). GGUF-only quantization works on CPU.
- git, cmake, a C++ compiler for the llama.cpp build (stage 01 can `apt-get` them with `environment.allow_system_packages: true` + `--yes`).
- Disk: roughly 3–4× the BF16 model size for the full Qwen3.8-27B pipeline (≈ 55 GB source, candidate, GGUF intermediate, exports). Preflight and acquire check real numbers.

## Five-minute quick start

```bash
git clone <this repo> model-abliteration-pipeline && cd model-abliteration-pipeline
./main.sh --help
./main.sh --config configs/pipeline.yaml --dry-run          # zero side effects: plan, components, disk estimate
./main.sh --setup-only --yes                                # bootstrap .venv + provision pinned tool environments
./main.sh --yes                                             # full pipeline with configs/pipeline.yaml
```

`./main.sh` without `--yes` asks for confirmation on a terminal and refuses (exit 15) when non-interactive.

### One B200 on Lambda Cloud (Lambda Stack)

```bash
export HF_TOKEN=...                       # optional for public repos; never written to disk by the platform
./main.sh --lambda-stack --dry-run
./main.sh --lambda-stack --setup-only --yes
./main.sh --lambda-stack --yes            # Qwen/Qwen3.8-27B, optimize 50 trials / 180 min, fp8 + iq4_xs + iq3_s
```

The ARA reference (`coder3101/Qwen3.8-27B-heretic-ara`) can only be reproduced **approximately**; see
[examples/qwen-b200.yaml](examples/qwen-b200.yaml) and [docs/heretic-adapter.md](docs/heretic-adapter.md).

### Plain Ubuntu

```bash
./main.sh --ubuntu --dry-run
./main.sh --ubuntu --setup-only --yes     # reports missing host packages; apt only with allow_system_packages
./main.sh --config examples/ubuntu-quantize-only.yaml --ubuntu --yes
```

## Run, resume, quantize-only

```bash
./main.sh --through 2 --yes                         # stop after acquisition
./main.sh --resume RUN_ID                           # continue; valid receipts are reused
./main.sh --resume RUN_ID --from 5 --through 6 --rerun
./main.sh --resume RUN_ID --fork --quantizations fp8  # config change → new run reusing still-valid work
./main.sh --model-url org/model --quantize-only --quantizations fp8,iq4_xs,iq3_s --yes
./main.sh --model-path /data/model --quantize-only --yes
./main.sh --experiment optimize --max-trials 50 --max-runtime-minutes 120 --yes
./main.sh --status RUN_ID ; ./main.sh --list-runs ; ./main.sh --cleanup
```

More in [docs/cli.md](docs/cli.md) and [examples/resume.md](examples/resume.md).

## Outputs

```text
workspace/
├── cache/{hf,uv}                         HF cache (verified, read-only source blobs) and uv cache
├── tools/{components.json,envs,heretic,llama.cpp}
├── models/source/<org>--<name>/<sha>.json   verified source records (+ link to the snapshot)
├── models/candidates/<run>/<candidate>/     Heretic exports
├── models/selected/<run>/master.json        pointer to the promoted master
├── models/gguf-intermediate/                re-creatable conversion intermediates
├── manifests/<artifact_id>.json             every manifest ever produced
├── exports/{bf16,fp8,gguf}/<name>/          deliverables (+ <name>.manifest.json sidecars)
└── runs/<run>/{logs,receipts,configs,metrics,checkpoints,reports}/
```

`runs/<run>/reports/report.md` summarizes stages, artifacts (integrity, loader test, modalities
actually tested), gates, metric provenance, tool versions and measured time.

## Documentation

[architecture](docs/architecture.md) · [stages](docs/stages.md) · [cli](docs/cli.md) ·
[configuration](docs/configuration.md) · [Lambda](docs/deployment-lambda.md) · [Ubuntu](docs/deployment-ubuntu.md) ·
[Heretic adapter](docs/heretic-adapter.md) · [study lifecycle](docs/study-lifecycle.md) · [quantization](docs/quantization.md) · [evaluation](docs/evaluation.md) ·
[artifact schema](docs/artifact-schema.md) · [security](docs/security-and-trust.md) · [troubleshooting](docs/troubleshooting.md) ·
[limitations](docs/limitations.md) · [implementation log](docs/implementation-log.md) · [original specification](docs/PROJECT_A_AGENT_HANDOFF.md)

## Development

```bash
uv sync --frozen            # or ./main.sh --bootstrap --yes, then .venv/bin/pip install -r requirements/dev.txt
.venv/bin/python -m pytest  # CPU-only unit + mocked integration tests
python scripts/generate_schema.py --check
REALTOOLS_WS=/tmp/ws python tests/realtools/provision_cpu.py && REALTOOLS_WS=/tmp/ws pytest -m realtools tests/realtools
pytest -m gpu tests/gpu     # on the B200 only
```

Project B (coding-capability research) is intentionally **not** part of this repository.
Third-party licenses: [LICENSE-THIRD-PARTY.md](LICENSE-THIRD-PARTY.md) (Heretic is AGPL-3.0).
