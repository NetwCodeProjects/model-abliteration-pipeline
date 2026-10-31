# Stages

Stage IDs and names are stable public contracts. Every executed stage writes
`runs/<run>/receipts/<id>-<name>.json` atomically:

```json
{
  "schema_version": "1.0", "stage_id": "05", "stage_name": "abliterate", "run_id": "…",
  "started_utc": "…", "ended_utc": "…", "duration_s": 0.0,
  "config_hash": "<sha256 of the full non-operational config>",
  "stage_config_hash": "<sha256 of this stage's config subset>", "stage_key": "<sha256>",
  "dependency_receipts": {"04": "<stage_key>"}, "input_artifact_ids": ["art-…"], "output_artifact_ids": [],
  "tool_revisions": {"heretic": "171bb98b…"}, "outputs": [{"path": "…", "kind": "…", "sha256": null, "size_bytes": 0}],
  "outcome": "succeeded|failed|skipped|blocked|interrupted", "partial": false,
  "warnings": [], "diagnostics": {}, "error": null, "resources": {"peak_gpu_memory_mib": …}, "resume": {…}
}
```

| ID | Name | Depends on | Inputs | Outputs | Gate |
|---|---|---|---|---|---|
| 00 | preflight | — | host | `metrics/preflight.json` | Linux; GPU if GPU components needed; free disk ≥ `min_free_disk_gb`; capability findings (below); warnings for ECC/ephemeral storage. **Always re-run.** |
| 01 | environment | 00 | profile, pins | `tools/components.json`, `metrics/components.json` | host prerequisites satisfied (or guarded installs succeeded); every needed component validated *now* (imports, `heretic --help`, CUDA matmul unless `torch_backend=cpu`) |
| 02 | acquire | 00 | model ref | source record + manifest, reproduce.json, prefetched pinned datasets, imatrix text | commit SHA resolved; every file sha256 / git-blob verified; source read-only |
| 03 | published_baselines | 02 | baseline YAML | `metrics/published_baselines.json` | provenance fields present; zero inference |
| 04 | prepare_engine | 01 (heretic), 02 | template TOML, reproduce input | `checkpoints/heretic/engine-plan.json`, `prepared-config.toml` | installed Heretic introspected; native keys ⊆ installed Settings; modifier importable; reproduction label decided or blocked |
| 05 | abliterate | 04 | engine plan | `models/candidates/<run>/<cand>/`, `metrics/candidate.json`, `metrics/heretic_study.json` | checkpoint written (config.json + all indexed shards); exit code not trusted |
| 06 | qualify | 05 | candidate | `metrics/qualification.json` | reload + generations completed (gates are applied in 07) |
| 07 | select_master | 05, 06 | gates | `exports/<precision>/<name>/` + manifest, `metrics/selection.json` | decision table outcome is `promoted:*` (qualified, research or ungated; docs/evaluation.md) |
| 08 | quantize | 01 (fp8/llama_cpp), 07 *or* 02 | master/source/path | `exports/fp8/…`, `exports/gguf/…` + manifests, `receipts/08-quantize/<fmt>.json` | each format succeeds independently; input fingerprint unchanged |
| 09 | verify_exports | 01, 02, 07?, 08? | registered artifacts | `metrics/verification.json`, updated manifests | re-hash matches; loader smoke tests (failures → partial; an FP8 test on a GPU without native FP8 is recorded as not representative and leaves `inference_passed` null) |
| 10 | report_export | 09 | everything | `reports/{report.md,run-summary.json,manifest-index.json}` | — |

## Restart and invalidation rules

- A stage is **reused** iff its receipt `succeeded`, its `stage_key` is unchanged, and
  `validate_outputs` passes (outputs exist with recorded sizes; for 02 the source quick-fingerprint
  must be unchanged; for 01 the component records and executables must still exist).
- `--rerun` forces re-execution of the selected range; old receipts move to `receipts/history/`.
- `--from N` never re-executes stages before N; they must have valid receipts (otherwise `E_MISSING_DEPENDENCY`, exit 7).
- Skipped-as-not-applicable receipts satisfy dependencies (e.g. quantize-only skips 03–07).
- **Stage 05 phases and resume** ([study lifecycle](study-lifecycle.md)): search-only, constraint-based
  selection, export-only, under a study lock, each recorded in `study-history.jsonl`. The journal lives in
  `checkpoints/heretic/study-<study_id>/`; an interrupted stage reconciles open history intents with the
  journal and continues the same study. A changed selection (fork) re-runs selection and export only,
  with zero new trials; a larger budget continues the search. Incomplete `.partial` exports are deleted
  before retrying.
- **Stage 08 resume:** finished formats are reused via sub-receipts or via an existing finalized export
  with the same parent artifact and recipe ID (also across runs); finalized exports are never overwritten.
- **Interrupts:** first SIGINT/SIGTERM is forwarded to the child (a search stops without saving; the
  journal is kept), the stage is recorded `interrupted` (exit 130) with an `interruption` entry in the
  study history; a second signal aborts immediately.

## Capability findings (stage 00)

`environment/capabilities.py` compares what the configured backends need with what the host offers.
Every finding has one of three categories and cites the requirement it rests on (`basis`). Findings are
written to `metrics/preflight.json` and shown as warnings.

| Category | Effect | Rules |
|---|---|---|
| `unsupported_configuration` | stops before any work (`E_CAPABILITY_UNSUPPORTED`, exit 4) | `fp8_vllm_compute`: FP8 loader test requested on a GPU below compute capability 8.0. vLLM documents no FP8 path there. |
| `estimated_resource_shortage` | warning | `vram_vs_checkpoint` (Heretic needs the BF16 model on the GPU), `disk_estimate` |
| `unverified_execution` | warning | `fp8_vllm_weight_only`: compute capability 8.0–8.8 (e.g. A100) has no native FP8, so vLLM runs a weight-only fallback. `hardware_not_exercised`: no recorded real run on this compute capability (e.g. B200 10.0, B300 10.3). |

Compute capability is used only where a backend documents it as the requirement. It is never a general
compatibility verdict.
