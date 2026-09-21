"""Test O1: does the classifier of Model 2 (and the size of the calibration set) explain the large fitness values in the early GA logs?

Background. The fitness of an individual of the second pipeline is the annualised Sharpe ratio of Model 1's trades on the
calibration set, for the bars that Model 2 approves, minus 0.05 per selected feature. Model 2 is trained on the same
calibration set (in-sample). The early runs of 16 June used GradientBoostingClassifier(200 trees, depth 4) (commit 87071b3);
from 18 June on the code uses HistGradientBoostingClassifier (commit 06b86b3), which stops early on its own held-out part of
the training data when the training set has more than 10 000 rows ("early_stopping='auto'"). The calibration set has about
16 600 rows in the full mode and about 3 300 in the fast mode (20% of the data), so the automatic early stopping is on in the
full mode and off in the fast mode.

Design. The same random individuals (chromosomes) are evaluated in both modes with the same random seeds. For each
individual Model 1 is trained once (5 epochs, as in the GA of the fast mode), and Model 2 is then fitted with four variants
on the same calibration features:
    GB        GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)       (code of 16 June)
    HGB_auto  HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.1)      (code from 18 June)
    HGB_off   the same with early_stopping=False
    HGB_on    the same with early_stopping=True
For every variant the fitness (as in ga_optimizer_v2.compute_sharpe_fitness) is computed in-sample (calibration set, as the
GA does) and, with the same formula and the same threshold, out-of-sample (test set), together with the precision of the
approved bars and the AUC of Model 2's confidence for "Model 1 was right".

Usage: python quick_o1_model2.py <fast|full> <part> <n_parts> [n_individuals] [epochs] [seed]
Writes RESULTS_O1/<mode>_part<part>.jsonl (one JSON line per individual and Model 2 variant).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

mode = sys.argv[1]
part, n_parts = int(sys.argv[2]), int(sys.argv[3])
N_IND = int(sys.argv[4]) if len(sys.argv) > 4 else 24
EPOCHS = int(sys.argv[5]) if len(sys.argv) > 5 else 5
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 20260920
assert mode in ("fast", "full")
FRAC = 0.2 if mode == "fast" else 1.0
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from confidence_model import extract_m2_features  # noqa: E402
from data_processor_v2 import DataProcessorV2  # noqa: E402
from feature_engineer_v2 import FeatureEngineerV2  # noqa: E402
from ga_optimizer_v2 import (CONF_THRESH_OPTIONS, DROPOUT_OPTIONS, HIDDEN_OPTIONS, LR_OPTIONS,  # noqa: E402
                             THRESH_MULT_OPTIONS, WINDOW_OPTIONS, ChromosomeV2)
from model_builder import ClassificationBiLSTMModel  # noqa: E402
from trainer_v2 import TrainerV2  # noqa: E402

DATA_DIR = os.environ.get("DATA_DIR", "DATA")
OUT = os.environ.get("O1_OUT", "RESULTS_O1")
os.makedirs(OUT, exist_ok=True)
BARS_PER_DAY = 13
PENALTY = 0.05

VARIANTS = {
    "GB": lambda: GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
    "HGB_auto": lambda: HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=42, learning_rate=0.1),
    "HGB_off": lambda: HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=42, learning_rate=0.1,
                                                      early_stopping=False),
    "HGB_on": lambda: HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=42, learning_rate=0.1,
                                                     early_stopping=True),
}


def draw_population(n: int, num_features: int, seed: int) -> list[list[int]]:
    """Random individuals as in GAOptimizerV2._create_ind (uniform genes, at least three features)."""
    rng = random.Random(seed)
    pop = []
    for _ in range(n):
        genes = [rng.randint(0, 1) for _ in range(num_features)]
        genes += [rng.randint(0, len(o) - 1) for o in (WINDOW_OPTIONS, HIDDEN_OPTIONS, DROPOUT_OPTIONS, LR_OPTIONS,
                                                       THRESH_MULT_OPTIONS, CONF_THRESH_OPTIONS)]
        if sum(genes[:num_features]) < 3:
            for idx in rng.sample(range(num_features), 3):
                genes[idx] = 1
        pop.append(genes)
    return pop


def fitness_terms(F: np.ndarray, n_feat: int, ret: np.ndarray, conf: np.ndarray, thr: float, min_trades: int = 10):
    """Sharpe of the approved bars as in compute_sharpe_fitness; returns (sharpe or None, n_approved, precision)."""
    probs = F[:, n_feat:n_feat + 3]
    pred = probs.argmax(1)
    direction = np.where(pred == 2, 1, np.where(pred == 0, -1, 0))
    approved = conf > thr
    n_app = int(approved.sum())
    if n_app < min_trades:
        return None, n_app
    pnl = direction[approved] * ret[approved]
    if pnl.std() < 1e-9:
        return None, n_app
    return float(pnl.mean() / pnl.std() * np.sqrt(252 * BARS_PER_DAY)), n_app


t0 = time.time()
dp = DataProcessorV2()
fe = FeatureEngineerV2()
stock_dfs, feature_cols = dp.load_and_engineer_all(DATA_DIR, fe, threshold_multiplier=0.5, resample_period="30min")
full_train, full_cal, full_test = dp.split_all_stocks(stock_dfs)
train_dfs = {t: d.iloc[:int(len(d) * FRAC)].copy() for t, d in full_train.items()}
cal_dfs = {t: d.iloc[:int(len(d) * FRAC)].copy() for t, d in full_cal.items()}
test_dfs = {t: d.copy() for t, d in full_test.items()}
print(f"[{mode}/{part}] data ready in {time.time() - t0:.0f}s; features {len(feature_cols)}", flush=True)

population = draw_population(N_IND, len(feature_cols), SEED)
device = torch.device("cpu")
out_path = os.path.join(OUT, f"{mode}_part{part}.jsonl")
done = set()
if os.path.exists(out_path):  # resume
    for line in open(out_path):
        done.add(json.loads(line)["ind"])
fh = open(out_path, "a")

for i, vec in enumerate(population):
    if os.environ.get("O1_CLAIM"):  # dynamic work sharing: the first worker that creates the claim folder evaluates the individual
        try:
            os.mkdir(os.path.join(OUT, f"claim_{mode}_{i}"))
        except FileExistsError:
            continue
    elif i % n_parts != part:
        continue
    if i in done:
        continue
    t1 = time.time()
    chrom = ChromosomeV2.from_vector(vec, len(feature_cols))
    active = [n for n, b in zip(feature_cols, chrom.feature_mask) if b == 1]
    mtr = {t: dp.compute_targets_and_labels(d.copy(), chrom.threshold_multiplier) for t, d in train_dfs.items()}
    mca = {t: dp.compute_targets_and_labels(d.copy(), chrom.threshold_multiplier) for t, d in cal_dfs.items()}
    mte = {t: dp.compute_targets_and_labels(d.copy(), chrom.threshold_multiplier) for t, d in test_dfs.items()}
    arr = dp.scale_and_window_multi(mtr, mca, mte, active, chrom.window_size)

    random.seed(SEED + i); np.random.seed(SEED + i); torch.manual_seed(SEED + i)
    model = ClassificationBiLSTMModel(len(active), chrom.hidden_units, 2, chrom.dropout)
    trainer = TrainerV2(model, device, chrom.learning_rate)
    train_loader, cal_loader = trainer.create_dataloaders(
        arr["X_train"], arr["y_train"], arr["ret_train"], arr["X_cal"], arr["y_cal"], arr["ret_cal"])
    trainer.train(train_loader, cal_loader, epochs=EPOCHS, verbose=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(arr["X_test"]), torch.LongTensor(arr["y_test"]),
                                           torch.FloatTensor(arr["ret_test"])), batch_size=512, shuffle=False)
    F_cal, y_cal = extract_m2_features(model, cal_loader, device)
    F_te, y_te = extract_m2_features(model, test_loader, device)
    ret_cal, ret_te = arr["ret_cal"], arr["ret_test"]
    n_feat = len(active)
    t_m1 = time.time() - t1

    for name, make in VARIANTS.items():
        t2 = time.time()
        m2 = make()
        m2.fit(F_cal, y_cal)
        conf_cal = m2.predict_proba(F_cal)[:, 1]
        conf_te = m2.predict_proba(F_te)[:, 1]
        sh_cal, n_cal_app = fitness_terms(F_cal, n_feat, ret_cal, conf_cal, chrom.confidence_threshold)
        sh_te, n_te_app = fitness_terms(F_te, n_feat, ret_te, conf_te, chrom.confidence_threshold)
        app_c, app_t = conf_cal > chrom.confidence_threshold, conf_te > chrom.confidence_threshold
        rec = {
            "mode": mode, "ind": i, "variant": name,
            "n_features": n_feat, "window": chrom.window_size, "hidden": chrom.hidden_units, "dropout": chrom.dropout,
            "lr": chrom.learning_rate, "mult": chrom.threshold_multiplier, "conf_thr": chrom.confidence_threshold,
            "n_train": int(len(arr["y_train"])), "n_cal": int(len(y_cal)), "n_test": int(len(y_te)),
            "m1_acc_cal": float(y_cal.mean()), "m1_acc_test": float(y_te.mean()),
            "n_iter": int(getattr(m2, "n_iter_", getattr(m2, "n_estimators_", -1))),
            "sharpe_cal": sh_cal, "fitness_cal": None if sh_cal is None else sh_cal - PENALTY * n_feat,
            "n_app_cal": n_cal_app, "prec_app_cal": float(y_cal[app_c].mean()) if app_c.any() else None,
            "sharpe_test": sh_te, "fitness_test": None if sh_te is None else sh_te - PENALTY * n_feat,
            "n_app_test": n_te_app, "prec_app_test": float(y_te[app_t].mean()) if app_t.any() else None,
            "auc_cal": float(roc_auc_score(y_cal, conf_cal)), "auc_test": float(roc_auc_score(y_te, conf_te)),
            "prec05_cal": float(y_cal[conf_cal > 0.5].mean()) if (conf_cal > 0.5).any() else None,
            "secs_m1": round(t_m1, 1), "secs_m2": round(time.time() - t2, 1),
        }
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
    print(f"[{mode}/{part}] ind {i}: {n_feat} feat, w{chrom.window_size} h{chrom.hidden_units}, cal {len(y_cal)}, "
          f"M1 {t_m1:.0f}s, total {time.time() - t1:.0f}s", flush=True)
print(f"[{mode}/{part}] done in {time.time() - t0:.0f}s", flush=True)
