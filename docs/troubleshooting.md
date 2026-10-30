# Troubleshooting

Every failure prints `ERROR [E_ID] message` + `hint:` and is stored in the stage receipt. Logs:
`runs/<run>/logs/run.log`, `events.jsonl`, and per-stage `logs/<id>-<name>/<tool>.{stdout,stderr}.log`.
`./main.sh --status RUN_ID` shows stage states; `runs/<run>/reports/report.md` is written even after failures.

| Symptom | Likely cause | Fix |
|---|---|---|
| `E_CONTROL_ENV_MISSING` on `--dry-run` | no `.venv` and no system PyYAML | `sudo apt-get install python3-yaml` or `./main.sh --bootstrap --yes` |
| `E_CONFIRMATION_REQUIRED` (15) | non-interactive run without `--yes` | add `--yes` |
| `E_CAPABILITY_UNSUPPORTED` (4) | a requested step the backend documents as impossible on this GPU (e.g. FP8 loader test below compute capability 8.0) | follow the hint (e.g. `quantization.fp8.verify_backend: none`); the basis is in `metrics/preflight.json` |
| `E_CONFIG_INVALID … got bool` | unquoted YAML boolean-like value | quote it: `'off'` |
| `E_ENV_UNSUPPORTED_PLATFORM` | running stages on non-Linux | use Ubuntu/WSL2; `--dry-run` works anywhere |
| `E_ENV_AMBIGUOUS` | non-Ubuntu Linux with `profile: auto` | pass `--ubuntu` or `--lambda-stack` |
| `E_DISK_FLOOR` / `E_DISK_INSUFFICIENT` | not enough free space | move `--output-dir`; lower `min_free_disk_gb` only if you know the run fits |
| `E_HOST_PREREQ` | missing git/cmake/compiler/driver | run the printed command or enable `allow_system_packages` |
| `E_COMPONENT_VALIDATION_FAILED` "CUDA is not available" | CPU-only torch or driver too old for the wheel | check `nvidia-smi`; set `environment.torch_backend` (e.g. `cu128`) |
| `E_INSTALL_STEP_FAILED` at `llama-cpp-configure/build` | toolchain mismatch (e.g. CUDA 12.0 + GCC 13) | `quantization.gguf.build_cuda: off`, or install a matching CUDA toolkit |
| `E_HF_GATED` / `E_HF_NOT_FOUND` | gated/private repo or typo | accept the license; export the token in `model.hf_token_env` |
| `E_DOWNLOAD_CORRUPT` | interrupted/corrupted blob | delete the listed files from `workspace/cache/hf/hub/...`; re-run stage 02 (downloads resume) |
| `E_HERETIC_UNKNOWN_SETTING` | template key not in the installed Heretic | remove it; see `logs/04-prepare_engine/heretic-help.stdout.log` |
| `E_HERETIC_CONFIG` | Heretic rejected its config (exit 0!) | read `logs/05-abliterate/heretic.stdout.log` |
| `E_HERETIC_BUDGET_EXHAUSTED` | budget ended before the first trial | raise `--max-runtime-minutes`; `--resume` continues the same study |
| `E_HERETIC_NO_OUTPUT` | crash/OOM, remote-code prompt, unsupported architecture | inspect the Heretic log tail in the hint; try `bnb_4bit` in the template only if memory-bound (and accept the provenance note) |
| `E_EXACT_REPRODUCTION_UNAVAILABLE` (14) | no reproduce.json for the reference | `--allow-approximate-reproduction` (labelled) or `--experiment optimize` |
| `E_GATES_FAILED` (9) | candidate failed thresholds | inspect `metrics/selection.json`; choose another Pareto trial with `--resume RUN --fork --pareto-index N` (study is reused), adjust thresholds, or override named gates (`selection.override_gates`, labelled `ungated`) |
| `E_GATES_UNMEASURED` (9) | a required gate has no evidence (qualification off/failed, reproduction without optimizer metrics) | measure it, use `selection.profile: research` (labelled), or list the gate in `selection.optional_gates`; see docs/evaluation.md |
| `E_GATE_REQUIREMENT_MISSING` (3) | a required gate's threshold is `null` (or `require_*: false`) | set a threshold, or list the gate in `selection.optional_gates` |
| `E_IMATRIX_REQUIRED` | iq1/iq2/iq3_xxs/q2_k_s without calibration data | set `quantization.gguf.imatrix_dataset` |
| `E_MMPROJ_FAILED` | vision projector not convertible | `preserve_multimodal: false` for an explicit text-only GGUF |
| Exit 20 | some exports failed or failed their loader test | see `receipts/08-quantize/*.json` and `metrics/verification.json`; fix and `--resume` (finished formats are reused) |
| `E_LOCK_HELD` (12) | another process uses the run/tools dir | wait; delete the `.lock` only if that PID is gone (same-host stale locks are reclaimed automatically) |
| `E_RESUME_CONFIG_CONFLICT` (7) | overrides change a resumed run | drop them, or add `--fork` |
| Heretic on Windows: `NoConsoleScreenBufferError` | prompt_toolkit needs a console even headless | run on Linux/WSL2 |
