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
| 00 | preflight | — | host | `metrics/preflight.json` | Linux; GPU if GPU components needed; free disk ≥ `min_free_disk_gb`; warnings for disk/VRAM/ECC/ephemeral storage. **Always re-run.** |
| 01 | environment | 00 | profile, pins | `tools/components.json`, `metrics/components.json` | host prerequisites satisfied (or guarded installs succeeded); every needed component validated *now* (imports, `heretic --help`, CUDA matmul unless `torch_backend=cpu`) |
| 02 | acquire | 00 | model ref | source record + manifest, reproduce.json, prefetched pinned datasets, imatrix text | commit SHA resolved; every file sha256 / git-blob verified; source read-only |
| 03 | published_baselines | 02 | baseline YAML | `metrics/published_baselines.json` | provenance fields present; zero inference |
| 04 | prepare_engine | 01 (heretic), 02 | template TOML, reproduce input | `checkpoints/heretic/engine-plan.json`, `prepared-config.toml` | installed Heretic introspected; native keys ⊆ installed Settings; modifier importable; reproduction label decided or blocked |
| 05 | abliterate | 04 | engine plan | `models/candidates/<run>/<cand>/`, `metrics/candidate.json`, `metrics/heretic_study.json` | checkpoint written (config.json + all indexed shards); exit code not trusted |
| 06 | qualify | 05 | candidate | `metrics/qualification.json` | reload + generations completed (gates are applied in 07) |
| 07 | select_master | 05, 06 | gates | `exports/<precision>/<name>/` + manifest, `metrics/selection.json` | all evaluable gates pass (or explicit `allow_ungated_promotion`) |
| 08 | quantize | 01 (fp8/llama_cpp), 07 *or* 02 | master/source/path | `exports/fp8/…`, `exports/gguf/…` + manifests, `receipts/08-quantize/<fmt>.json` | each format succeeds independently; input fingerprint unchanged |
| 09 | verify_exports | 01, 02, 07?, 08? | registered artifacts | `metrics/verification.json`, updated manifests | re-hash matches; loader smoke tests (failures → partial) |
| 10 | report_export | 09 | everything | `reports/{report.md,run-summary.json,manifest-index.json}` | — |

## Restart and invalidation rules

- A stage is **reused** iff its receipt `succeeded`, its `stage_key` is unchanged, and
  `validate_outputs` passes (outputs exist with recorded sizes; for 02 the source quick-fingerprint
  must be unchanged; for 01 the component records and executables must still exist).
- `--rerun` forces re-execution of the selected range; old receipts move to `receipts/history/`.
- `--from N` never re-executes stages before N; they must have valid receipts (otherwise `E_MISSING_DEPENDENCY`, exit 7).
- Skipped-as-not-applicable receipts satisfy dependencies (e.g. quantize-only skips 03–07).
- **Stage 05 resume:** the Optuna journal lives in `checkpoints/heretic/study-<hash>/`; an interrupted
  stage continues the same study (`checkpoint_action=continue`) after updating its stored
  `n_trials`/`trial_index`. Incomplete `.partial` exports are deleted before retrying.
- **Stage 08 resume:** finished formats are reused via sub-receipts or via an existing finalized export
  with the same parent artifact and recipe ID (also across runs); finalized exports are never overwritten.
- **Interrupts:** first SIGINT/SIGTERM is forwarded to the child (Heretic exports the best completed
  trial on SIGINT), the stage is recorded `interrupted` (exit 130); a second signal aborts immediately.
