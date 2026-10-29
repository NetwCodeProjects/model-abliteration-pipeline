#!/usr/bin/env bash
# Step 0 feasibility checks against the real pinned Heretic (see docs/step0-feasibility.md).
# Usage: REALTOOLS_WS=/path/ws [STEP0_DIR=~/step0] bash tests/realtools/step0/run_all.sh
source "$(dirname "$0")/lib.sh"
rm -rf "$S0"/{A,C,C-hazard,C-short,D1,D2,K}

echo "== A: search-only (model_action=\"\"), 5 trials"
mkcfg "$S0/A" 5 "" restart "$S0/A/out-must-not-exist" 0
run_heretic "$S0/A" "$S0/A/search.log"; engine_lines "$S0/A/search.log"
ls "$S0/A/out-must-not-exist" 2>/dev/null && echo "UNEXPECTED: model saved" || echo "no model saved"
inspect "$S0/A" > "$S0/A/before.json"

echo "== A: export-only, Pareto position 2, then 2 again, then 0 (config.toml says trial_index=0, model_action=\"\")"
for spec in "2 pos2-a" "2 pos2-b" "0 pos0"; do
  set -- $spec
  patch "$S0/A" "n_trials=5" "trial_index=$1" 'model_action="save"'
  mkcfg "$S0/A" 5 "" continue "$S0/A/export-$2" 0
  run_heretic "$S0/A" "$S0/A/export-$2.log"; engine_lines "$S0/A/export-$2.log"
done
inspect "$S0/A" > "$S0/A/after.json"; same_trials "$S0/A/before.json" "$S0/A/after.json"
"$PY" "$HERE/study.py" check-restore "$S0/A/checkpoints/model.jsonl" "$S0"/A/export-*.log

echo "== D: direct export vs export-only of the same weight-changing trial"
mkcfg "$S0/D1" 1 save restart "$S0/D1/direct" 0; run_heretic "$S0/D1" "$S0/D1/run.log"
mkcfg "$S0/D2" 1 "" restart "$S0/D2/none" 0; run_heretic "$S0/D2" "$S0/D2/search.log"
inspect "$S0/D2" > "$S0/D2/before.json"
patch "$S0/D2" "n_trials=1" "trial_index=0" 'model_action="save"'
mkcfg "$S0/D2" 1 "" continue "$S0/D2/export-only" 0; run_heretic "$S0/D2" "$S0/D2/export.log"; engine_lines "$S0/D2/export.log"
inspect "$S0/D2" > "$S0/D2/after.json"; same_trials "$S0/D2/before.json" "$S0/D2/after.json"
echo "direct $(whash "$S0/D1/direct")  export-only $(whash "$S0/D2/export-only")  source $(whash "$SNAP")"

echo "== C: budget stop (SIGINT at trial 4 of 10), then export-only"
mkcfg "$S0/C" 10 "" restart "$S0/C/none" 0; run_heretic "$S0/C" "$S0/C/search.log" INT 4; engine_lines "$S0/C/search.log"
inspect "$S0/C" > "$S0/C/before.json"
for v in hazard short; do cp -a "$S0/C" "$S0/C-$v"; ln -sfn "$SNAP" "$S0/C-$v/model"; done
patch "$S0/C" "n_trials=4" "trial_index=0" 'model_action="save"'   # 4 = len(study.trials), PRUNED included
mkcfg "$S0/C" 10 "" continue "$S0/C/export-only" 0; run_heretic "$S0/C" "$S0/C/export.log"; engine_lines "$S0/C/export.log"
inspect "$S0/C" > "$S0/C/after.json"; same_trials "$S0/C/before.json" "$S0/C/after.json"

echo "== C-short: n_trials below the recorded count (COMPLETE only = 3)"
patch "$S0/C-short" "n_trials=3" "trial_index=0" 'model_action="save"'
mkcfg "$S0/C-short" 10 "" continue "$S0/C-short/export-only" 0; run_heretic "$S0/C-short" "$S0/C-short/export.log"; engine_lines "$S0/C-short/export.log"
inspect "$S0/C-short" > "$S0/C-short/after.json"; same_trials "$S0/C/before.json" "$S0/C-short/after.json"

echo "== C-hazard: current update-settings behavior, n_trials = max(requested 10, recorded 4)"
patch "$S0/C-hazard" "n_trials=10" "trial_index=0" 'model_action="save"'
mkcfg "$S0/C-hazard" 10 "" continue "$S0/C-hazard/export" 0; run_heretic "$S0/C-hazard" "$S0/C-hazard/export.log"; engine_lines "$S0/C-hazard/export.log"
inspect "$S0/C-hazard" > "$S0/C-hazard/after.json"; same_trials "$S0/C/before.json" "$S0/C-hazard/after.json"

echo "== K: hard kill (SIGKILL at trial 3 of 10), then export-only"
mkcfg "$S0/K" 10 "" restart "$S0/K/none" 0; run_heretic "$S0/K" "$S0/K/search.log" KILL 3
inspect "$S0/K" > "$S0/K/before.json"
patch "$S0/K" "n_trials=3" "trial_index=0" 'model_action="save"'
mkcfg "$S0/K" 10 "" continue "$S0/K/export-only" 0; run_heretic "$S0/K" "$S0/K/export.log"; engine_lines "$S0/K/export.log"
inspect "$S0/K" > "$S0/K/after.json"; same_trials "$S0/K/before.json" "$S0/K/after.json"
