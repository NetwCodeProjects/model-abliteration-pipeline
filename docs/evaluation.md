# Evaluation: provenance first

Project A distinguishes four kinds of numbers and never mixes them:

| Kind | Produced by | Where | Label |
|---|---|---|---|
| **Published** | model cards (copied with source URL, retrieval date, evaluator, caveat) | stage 03 → `metrics/published_baselines.json` | `published (copied from cited sources; not measured by this platform)` |
| **Optimizer metrics** | Heretic's in-loop scorers (KeywordRate refusals, first-token KL) on Heretic's own evaluation prompts, used to *select* trials | stage 05 → `metrics/heretic_study.json`, `candidate.json` | `heretic_optimizer (… not held-out validation)` |
| **Independent fast qualification** | Project A: reload the saved candidate from disk with transformers, greedy generations on held-out prompts | stage 06 → `metrics/qualification.json` | `independent_qualification` |
| **Loader smoke tests** | vLLM / llama.cpp / transformers on exported files | stage 09 → manifests | `integrity-only` / validation fields |

## Published baselines (default, `baseline_mode: published_only`)

`configs/baselines/qwen3.8-27b.yaml` records the reference's card values (refusals 9/100 vs 92/100,
KL 0.0349), its revision, the claimed method/engine version, observations (missing `mtp.*` tensors,
shard layout) and a comparability caveat. Stage 03 validates provenance fields, warns when the run's
model/revision differs, and performs **zero** downloads or inference of reference/original models.
`baseline_mode: local` is not implemented in v1 (exit 14).

## Fast qualification (default)

- **Benign checks** (`configs/qualification/benign_prompts.yaml`, 10 `fast` prompts): answers checked
  by case-insensitive substring on the visible answer (thinking blocks removed).
- **Held-out refusal prompts:** `mlabonne/harmful_behaviors@01cead01` splits `train[400:416]` and
  `test[100:104]` — disjoint from Heretic's default `train[:400]` (direction/ARA prompts) and
  `test[:100]` (optimizer refusal scorer). Total prompts are capped at `independent_smoke_prompts` (30).
- **Scorer:** `keyword_refusal_heuristic_v1` (Project A's own marker list, first 400 visible characters).
- **Integrity:** empty output, replacement characters, low lexical diversity, repetition loops, NaN.
- **Reload:** loading from disk must succeed; dtype, load time and peak GPU memory are recorded.
- Generation settings recorded: chat template hash, `enable_thinking` (default false for short checks),
  `max_new_tokens`, greedy decoding, seed, dataset revisions.

## Extended qualification

`qualification: extended` adds the `extended` benign tier, all held-out refusal prompts, and
**held-out KL**: mean KL(source ‖ candidate) of next-token distributions on
`mlabonne/harmless_alpaca@02c6a92c test[1000:1100]` (disjoint from Heretic's `test[:100]`), computed by
loading both checkpoints (fits on one B200). `extended_kl` / `extended_correctness` can be enabled
individually with `fast`.

## Gates (stage 07)

| Gate | Default | Provenance |
|---|---|---|
| `max_optimizer_refusal_rate` | 0.2 | heretic_optimizer |
| `max_optimizer_kl` | 0.1 | heretic_optimizer |
| `max_heldout_refusal_rate` | 0.35 | independent_qualification |
| `min_benign_pass_rate` | 0.7 | independent_qualification |
| `require_integrity`, `require_reload` | true | independent_qualification |

A gate whose metric was not measured (e.g. optimizer metrics in reproduction mode) is reported as
`passed: null` with a warning — it is never silently counted as passed. Any failing gate stops the run
before promotion (`E_GATES_FAILED`, exit 9) unless `allow_ungated_promotion` is set, which labels the
master `ungated`.

## What these numbers do not mean

A 30-prompt smoke test cannot establish statistically meaningful refusal rates or capability retention;
keyword refusal detection misses soft refusals and can flag benign wording. Refusal count is not
accuracy. Deep coding/capability evaluation (HumanEval+, SWE-bench, …) is **Project B**, which consumes
the manifests produced here.
