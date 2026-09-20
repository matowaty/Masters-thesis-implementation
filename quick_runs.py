"""Seeded wrapper around the unmodified main.py experiments (time_horizon, attention).

Usage: python quick_runs.py <experiment> <seed>
Writes seed info into every RESULTS folder created by the run.
"""
import json, os, random, sys
import numpy as np
import torch

exp, seed = sys.argv[1], int(sys.argv[2])
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.set_num_threads(2)

import main  # noqa: E402  (unmodified pipeline)

before = set(os.listdir("RESULTS")) if os.path.isdir("RESULTS") else set()
main.setup_logging()
{"time_horizon": main.run_time_horizon_experiment,
 "attention": main.run_attention_comparison}[exp]()
after = set(os.listdir("RESULTS"))
for d in sorted(after - before):
    with open(os.path.join("RESULTS", d, "seed.json"), "w") as f:
        json.dump({"seed": seed, "experiment": exp, "wrapper": "quick_runs.py",
                   "torch": torch.__version__, "device": "cpu"}, f)
print("DONE", exp, seed, sorted(after - before))
