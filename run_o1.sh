#!/usr/bin/env bash
# Test O1 (CPU): a random generation 0 of 20 individuals, evaluated in the fast mode (20% of the data) and in the full mode,
# each with four variants of Model 2 (see quick_o1_model2.py). Two worker processes share the individuals of a mode
# (O1_CLAIM=1: the first worker that creates the folder RESULTS_O1/claim_<mode>_<i> evaluates individual i). The result of an
# individual does not depend on the worker, because all random seeds are derived from the seed and the index of the individual.
# In the run reported in the thesis the fast mode used a fixed split of the individuals (i mod 2), the full mode the claim folders;
# the results are the same in both ways. Takes about 30 minutes on two CPU cores.
set -euo pipefail
cd "$(dirname "$0")"
export DATA_DIR="${DATA_DIR:-DATA}" O1_OUT="${O1_OUT:-RESULTS_O1}" O1_CLAIM=1
for mode in fast full; do
  python quick_o1_model2.py $mode 0 2 20 5 > "logs_o1_${mode}_0.txt" 2>&1 &
  python quick_o1_model2.py $mode 1 2 20 5 > "logs_o1_${mode}_1.txt" 2>&1 &
  wait
done
