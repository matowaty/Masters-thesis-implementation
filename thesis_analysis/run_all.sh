#!/usr/bin/env bash
# Regenerates every figure, table and number macro used in the thesis.
# Inputs (read-only): A100-results/RESULTS, DATA, RESULTS (seeded quick runs), RESULTS_V2SHIFT (V2 window-offset check), inputs/other_runs.csv
# Outputs: out/figures/*.pdf|png, out/tables/*.tex, out/tables/numbers.tex
# Override paths with A100_RESULTS, DATA_DIR, QUICK_RESULTS, V2SHIFT_RESULTS, THESIS_OUT.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf out && mkdir -p out
for s in an_runs an_v2_skill an_economics an_ga an_data an_v1 an_magnitude an_quick an_v2shift; do
  echo "=== $s ==="
  python "$s.py"
done
