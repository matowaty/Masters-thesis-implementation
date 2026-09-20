"""Quick CPU check of the one-bar window offset in the V2 pipeline (Model 1 only).

The V2 pipeline forms the window of bar i from the features of bars i-w .. i-1 and labels bar i with
the return C[i+1]/C[i]-1, so the model never sees bar i itself. The 'shift' variant uses bars i-w+1 .. i
(the last bar is the one whose close is the reference of the predicted return) with the same rows, labels,
split, scaler and seeds. Only Model 1 is trained; Model 2 and the GA are not involved.

Usage: python quick_v2_shift.py <base|shift> <seed,seed,...> [epochs] [patience]
Writes RESULTS_V2SHIFT/<variant>_seed<k>.json (metrics) and <variant>_seed<k>_test.npz (test-set probabilities, labels, returns, times, tickers)
"""
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

variant = sys.argv[1]
seeds = [int(s) for s in sys.argv[2].split(",")]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
PATIENCE = int(sys.argv[4]) if len(sys.argv) > 4 else 4
torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_processor_v2 import DataProcessorV2  # noqa: E402
from feature_engineer_v2 import FeatureEngineerV2  # noqa: E402
from model_builder import ClassificationBiLSTMModel  # noqa: E402
from trainer_v2 import TrainerV2  # noqa: E402

DATA_DIR = os.environ.get("DATA_DIR", "DATA")
OUT = "RESULTS_V2SHIFT"
os.makedirs(OUT, exist_ok=True)

# fixed configuration: all 28 features, window 12, 64 hidden units, dropout 0.2, lr 5e-4, m = 0.3
W, HIDDEN, DROPOUT, LR, MULT = 12, 64, 0.2, 5e-4, 0.3


def create_windows_shift(self, features, targets, fwd_returns, timestamps, tickers, window_size):
    X, y, ret, times, tks = [], [], [], [], []
    for i in range(window_size, features.shape[0]):  # same rows as the original
        X.append(features[i - window_size + 1: i + 1])  # last bar = bar i (original: i-w .. i-1)
        y.append(targets[i])
        ret.append(fwd_returns[i])
        times.append(timestamps[i])
        tks.append(tickers[i])
    return np.array(X), np.array(y), np.array(ret), np.array(times), np.array(tks)


if variant == "shift":
    DataProcessorV2.create_windows = create_windows_shift

t0 = time.time()
dp = DataProcessorV2()
fe = FeatureEngineerV2()
stock_dfs, feature_cols = dp.load_and_engineer_all(DATA_DIR, fe, threshold_multiplier=MULT, resample_period="30min")
train_dfs, cal_dfs, test_dfs = dp.split_all_stocks(stock_dfs)
data = dp.scale_and_window_multi(train_dfs, cal_dfs, test_dfs, feature_cols, W)
print(f"[{variant}] data ready in {time.time() - t0:.0f}s; features {len(feature_cols)}; "
      f"train {data['X_train'].shape} cal {data['X_cal'].shape} test {data['X_test'].shape}", flush=True)

from scipy.stats import binomtest  # noqa: E402
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, roc_auc_score)  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_analysis"))
from common import session_labels  # noqa: E402

for seed in seeds:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cpu")
    model = ClassificationBiLSTMModel(input_size=data["X_train"].shape[2], hidden_size=HIDDEN, num_layers=2, dropout=DROPOUT)
    trainer = TrainerV2(model=model, device=device, learning_rate=LR)
    train_loader, cal_loader = trainer.create_dataloaders(
        data["X_train"], data["y_train"], data["ret_train"],
        data["X_cal"], data["y_cal"], data["ret_cal"], batch_size=64)
    t1 = time.time()
    trainer.train(train_loader, cal_loader, epochs=EPOCHS, patience=PATIENCE, verbose=False)
    train_secs = time.time() - t1

    model.eval()
    probs = []
    with torch.no_grad():
        Xt = torch.FloatTensor(data["X_test"])
        for k in range(0, len(Xt), 2048):
            probs.append(torch.softmax(model(Xt[k:k + 2048]), dim=1).numpy())
    P = np.vstack(probs)
    y = data["y_test"].astype(int)
    r = data["ret_test"]
    pred = P.argmax(1)
    d = np.where(pred == 2, 1, np.where(pred == 0, -1, 0))
    call = (d != 0)
    nz = call & (r != 0)
    hits = int(((np.sign(r[nz]) == d[nz])).sum())
    n_nz = int(nz.sum())
    sess = session_labels(pd.DatetimeIndex(pd.to_datetime(data["time_test"])))
    reg = nz & (sess == "regular")
    res = {
        "variant": variant, "seed": seed, "epochs_max": EPOCHS, "patience": PATIENCE, "train_secs": round(train_secs),
        "n_test": int(len(y)), "class_share": [float((y == c).mean()) for c in range(3)],
        "pred_share": [float((pred == c).mean()) for c in range(3)],
        "accuracy": float((pred == y).mean()), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "kappa": float(cohen_kappa_score(y, pred)),
        "auc_magnitude": float(roc_auc_score((y == 1).astype(int), P[:, 1])),
        "auc_direction": float(roc_auc_score((y[y != 1] == 2).astype(int), (P[:, 2] - P[:, 0])[y != 1])),
        "calls": int(call.sum()), "calls_nonzero": n_nz, "sign_hits": hits,
        "sign_acc_nonzero": hits / max(n_nz, 1),
        "binom_p": float(binomtest(hits, n_nz, 0.5).pvalue) if n_nz else None,
        "sign_acc_regular": float((np.sign(r[reg]) == d[reg]).mean()) if reg.sum() else None,
        "n_regular": int(reg.sum()),
        "gross_bps_per_call": float((d[call] * r[call]).mean() * 1e4) if call.sum() else None,
    }
    np.savez_compressed(os.path.join(OUT, f"{variant}_seed{seed}_test.npz"), P=P.astype(np.float32), y=y.astype(np.int8), r=r,
                        time=np.array(pd.to_datetime(data["time_test"]).astype("datetime64[s]").astype(str)),
                        ticker=np.array(data["ticker_test"]).astype(str))
    with open(os.path.join(OUT, f"{variant}_seed{seed}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[{variant}] seed {seed}: sign acc {res['sign_acc_nonzero']:.3f} (n={n_nz}), reg {res['sign_acc_regular']}, "
          f"AUCmag {res['auc_magnitude']:.3f} AUCdir {res['auc_direction']:.3f} kappa {res['kappa']:.3f} "
          f"train {train_secs:.0f}s", flush=True)
