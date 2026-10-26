# Security and trust

## Risk profile

| Surface | Risk | Mitigation in Project A | Your responsibility |
|---|---|---|---|
| Model weights (safetensors) | Data only, but loaders and converters parse them | safetensors only for HF checkpoints; pure-Python header parsing in the control plane; sha256/git-blob verification against the Hub | Prefer reputable sources; check licenses |
| `trust_remote_code` models | Arbitrary Python from the model repo runs in tool processes | Off by default; Heretic cannot be told to trust code and stdin is closed, so remote-code models fail rather than prompting | Only enable for audited repos; run in an isolated VM/container |
| Third-party tools (Heretic, llama.cpp, llm-compressor, vLLM) | Supply chain | Pinned commit/tag/version; llama.cpp tag→commit verified (`E_PIN_MISMATCH`); Heretic deps constrained by its own `uv.lock`; control plane hash-locked (`requirements/control.txt`) | Review pins before bumping; record changes in the implementation log |
| Native builds | Compiler/toolchain on host | No `curl | sh`; builds use host cmake/compiler only; build flags recorded | — |
| Hub/dataset content | Untrusted text, prompts | Prompts treated as data; responses stored in run metrics (optionally redacted) | Treat generated content as untrusted; don't execute it |
| Credentials | Token leakage | Token read from the env var named in config; never stored in configs/receipts; redacted (`hf_…`, values of `*TOKEN*/*SECRET*/*KEY*` env vars) from all logs | Use read-only tokens; avoid shared shells |
| Publication | Accidental upload of an abliterated model | Heretic `model_action` is always `save`; reproduce files are patched to clear `upload_*`; no Hub writes anywhere | Decide publication explicitly and outside the pipeline |
| Host changes | Surprise system modifications | apt/driver steps only with explicit config flags + confirmation, via `sudo -n`; never reboots; no system-wide upgrades | Review the printed commands |
| Concurrent runs | Corrupted work areas | Run and tools locks with stale-PID detection; atomic writes; `.partial` directories | — |
| Cleanup | Deleting valuable data | Dry-run listing first; deletion only with a matching token; sources, masters and finalized exports are never candidates | — |

## Abliterated models

Abliteration removes refusal behaviour; outputs of the resulting models are unfiltered. Keep artifacts
access-controlled, label them clearly (manifests carry `transformation.kind = abliteration`), and follow
the base model's license and your organisation's policies before any distribution.

## Sandbox guidance

- Use a dedicated cloud instance or VM; avoid running as root (privileged steps use `sudo -n`).
- Keep `HF_TOKEN` scoped read-only; unset it for public models.
- If you must use `trust_remote_code`, run inside a container without host credentials or network
  egress beyond the Hub.
- Logs may contain prompts/responses; set `logging.redact_prompts: true` to store hashes instead in
  qualification records.

## AGPL note

Heretic is AGPL-3.0. Project A runs it unmodified as a separate process; the helper scripts in
`scripts/runtime/heretic_*.py` import Heretic modules inside its venv. If you redistribute Heretic or
those helpers in modified form, review the AGPL obligations (see LICENSE-THIRD-PARTY.md).
