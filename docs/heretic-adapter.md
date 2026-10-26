# Heretic adapter

## Pinned engine

| Item | Value |
|---|---|
| Repository | https://github.com/p-e-w/heretic |
| Commit | `171bb98b8b9a41c081631b5e678f7133128973e0` (master, 2026-10-24, package version `2.0.0.dev0`) |
| License | AGPL-3.0-or-later |
| Why not a release | The latest release **v1.4.0** (`6ea3b8d`, 2026-06-14) has neither the modifier-plugin system with **ARA** (merged in `662e4ba`, 2026-10-01, PR #446) nor **headless operation** (`0146b27`, 2026-06-27, PR #392). Without headless settings Heretic prompts interactively and cannot be automated. |
| Install | `git clone` + checkout of the commit → `uv export --frozen` of Heretic's own `uv.lock` as constraints (torch family removed) → torch/torchvision at the lock's versions with the CUDA variant from `uv --torch-backend` → `uv pip install -c constraints <src>` |
| Validation | `import heretic.config`, `import heretic.modifiers.ara`, GPU matmul, `heretic --help` containing `--model` |

## Verified interface (inspected, not assumed)

- Settings are a pydantic `BaseSettings` read from **CLI > `HERETIC_*` env > `config.toml` in the
  working directory**. The adapter writes `config.toml` into a run-private directory and executes Heretic
  with that `cwd`. Stage 04 runs `scripts/runtime/heretic_introspect.py` inside the Heretic venv to list
  `Settings.model_fields` of the *installed* version and rejects unknown keys
  (`E_HERETIC_UNKNOWN_SETTING`). Plugin tables `[scorer.*]` / `[modifier.*]` are allowed.
- Headless keys (same as upstream e2e tests): `export_strategy="merge"`, `checkpoint_action`,
  `trial_index`, `model_action="save"`, `save_directory`.
- `trial_index` indexes the **Pareto front sorted ascending by objective values** in scorer order — with
  the default scorers index 0 is the fewest-refusals trial (usually the highest KL).
- SIGINT during optimization → `study.stop()`; Heretic then restores the configured trial and saves it.
  If no trial completed it exits without saving (`E_HERETIC_BUDGET_EXHAUSTED`).
- Heretic returns **exit status 0 after configuration errors** ("Configuration contains N errors").
  Success is decided only by validating the saved checkpoint.
- The Optuna journal file name is derived from `settings.model`. Long absolute paths overflow filename
  limits (observed with the real engine), so Heretic is pointed at a short `model` symlink (junction on
  Windows) inside its working directory.
- Continuing a study restores **all** non-excluded settings from the study (including `n_trials` and
  `trial_index`). Before `checkpoint_action=continue` the adapter updates the stored settings through
  Optuna's public API (`scripts/runtime/heretic_study_tool.py update-settings`), as Heretic itself does
  for "run additional trials".
- `--reproduce <file>` accepts **version-4** reproduce.json; the file's settings replace config.toml
  settings (except `exclude=True` fields such as `save_directory`), so control-flow fields are patched.
- `trust_remote_code` is not a Heretic setting; transformers prompts on stdin (closed) → remote-code
  models fail in stage 05.
- On Windows Heretic constructs `questionary` prompts eagerly and prompt_toolkit requires a console;
  headless runs need Linux (the supported target).

## Mapping

| Pipeline key | Native setting |
|---|---|
| source snapshot path | `model` (via the `model` link) |
| `abliteration.max_trials` | `n_trials` |
| `abliteration.n_startup_trials` (null → min(template, n_trials//3)) | `n_startup_trials` |
| `abliteration.seed` (null → derived from run ID, recorded) | `seed` |
| `abliteration.pareto_trial_index` | `trial_index` |
| candidate `.partial` dir | `save_directory` |
| new / continued study | `checkpoint_action = restart / continue` |
| `abliteration.max_runtime_minutes` | adapter budget: SIGINT, then SIGTERM after `shutdown_grace_minutes`, then SIGKILL |
| `abliteration.reproduce.ignore_environment_mismatches` | `ignore_mismatches` |
| everything else (scorers, ARA, datasets, dtypes, batch size) | native template `configs/heretic/<experiment>.toml` |

Template keys owned by the orchestrator (`model`, `n_trials`, `seed`, `trial_index`, `model_action`,
`save_directory`, `export_strategy`, `checkpoint_action`, `study_checkpoint_dir`, `upload_*`, …) are rejected.

## Sample invocation (generated)

```toml
# runs/<run>/checkpoints/heretic/study-<hash>/config.toml
quantization = "none"
device_map = "auto"
batch_size = 0
max_response_length = 100
n_startup_trials = 16
scorers = [{ plugin = "heretic.scorers.keyword_rate.KeywordRate", optimization = "minimize" }, { plugin = "heretic.scorers.kl_divergence.KLDivergence", optimization = "minimize" }]
modifiers = [{ plugin = "heretic.modifiers.ara.ARA" }]
model = "model"
n_trials = 50
seed = 2779340105
trial_index = 0
model_action = "save"
export_strategy = "merge"
save_directory = "/…/workspace/models/candidates/<run>/c-1a2b3c4d-p0.partial"
checkpoint_action = "restart"
study_checkpoint_dir = "checkpoints"
# … pinned dataset tables, [modifier.ARA] …
```

```bash
cd runs/<run>/checkpoints/heretic/study-<hash> && COLUMNS=400 NO_COLOR=1 TERM=dumb \
  <workspace>/tools/envs/heretic-<fp>/bin/heretic          # stdin closed, own process group
```

## ARA status

ARA **is implemented** in the pinned engine (`heretic.modifiers.ara.ARA`, the upstream default modifier)
and is verified importable in stage 01/04. Its parameters (`start_layer_index`, `end_layer_index`,
`preserve_good_behavior_weight`, `steer_bad_behavior_weight`, `overcorrect_relative_weight`,
`neighbor_count`) match the names published by the reference model card. Standard directional ablation
is the separate `heretic.modifiers.abliteration.Abliteration` plugin; the adapter never substitutes one
for the other.

## Reproduction of `coder3101/Qwen3.8-27B-heretic-ara`

| Fact (verified 2026-10-26) | Consequence |
|---|---|
| Repo `7904b4f2…` contains no `reproduce/reproduce.json` | Exact reproduction impossible → `E_EXACT_REPRODUCTION_UNAVAILABLE` (exit 14) by default |
| Card claims "Heretic v1.2.0 with ARA (PR #211)"; v1.2.0 has no ARA code | The reference was built from an unreleased branch; the pinned ARA may differ |
| Card publishes the six ARA parameters rounded to 4 decimals and "row-norm preservation" | `allow_approximate: true` synthesizes a version-4 reproduce input with these parameters, `preserve_row_magnitudes=true`, pinned default datasets, `ignore_mismatches=true` |
| lora_rank, optimizer steps, dataset slices, seed are not published | recorded as assumptions; output label `approximate_reproduction` in receipts, candidate, master manifest and report |

When a real reproduce.json exists (via `reproduce.source` or the reference repo), the label is
`exact_reproduction_attempt`; bitwise identity is reported only if the saved safetensors match the
published hashes.
