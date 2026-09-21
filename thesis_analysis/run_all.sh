#!/usr/bin/env bash
# Regenerates every figure, table and number macro used in the thesis.
# Inputs (read-only): RESULTS (eight full A100 runs, seeded quick runs, early V2 runs), DATA, RESULTS_V2SHIFT (V2 window-offset check), inputs/other_runs.csv, inputs/run_manifest_all.csv, RESULTS_O1 (Model 2 check)
# Outputs: out/figures/*.pdf|png, out/tables/*.tex, out/tables/numbers.tex
# Override paths with A100_RESULTS, DATA_DIR, QUICK_RESULTS, V2SHIFT_RESULTS, EARLY_RESULTS, O1_RESULTS, THESIS_OUT.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf out && mkdir -p out
for s in an_runs an_v2_skill an_economics an_ga an_data an_v1 an_magnitude an_quick an_v2shift an_inventory an_early an_o1; do
  echo "=== $s ==="
  python "$s.py"
done
