"""Patient-level evaluation shared by every model (MLP, CNN).

Window probabilities -> patient scores (src.datasets.aggregate_to_patient) -> hierarchical
decision rule, as in the Challenge: Present if p(Present) >= t_present, else Unknown if
p(Unknown) >= t_unknown, else Absent. The thresholds are tuned for the weighted accuracy.

Honest CV estimate (cv_report): for each fold k the thresholds are tuned on the
out-of-fold predictions of the other folds and applied to fold k.

Out-of-fold window probabilities are saved as results/<model>/oof.csv, indexed by the row of
data/processed/windows.csv, so models trained elsewhere (e.g. on a GPU server) can be
evaluated here.
"""
import json

import numpy as np
import pandas as pd

from src.data import CLASSES, ROOT
from src.datasets import aggregate_to_patient
from src.metrics import report, weighted_accuracy

RESULTS_DIR = ROOT / "results"
GRID = np.round(np.arange(0.05, 1.0, 0.025), 3)


def patient_scores(windows, probs, how="max"):
    """One row per patient: scores per class (columns CLASSES), true `murmur` and `fold`."""
    pat = aggregate_to_patient(windows, probs, how)
    first = windows.groupby("patient_id")[["murmur", "fold"]].first()
    return pat.join(first)


def predict_rule(pat, t_present, t_unknown):
    return np.where(pat["Present"] >= t_present, "Present",
                    np.where(pat["Unknown"] >= t_unknown, "Unknown", "Absent"))


def tune_thresholds(pat, grid=GRID):
    """Grid search of (t_present, t_unknown) maximising the weighted accuracy on `pat`
    (vectorised over t_unknown; ties go to the first pair in grid order)."""
    y = pat["murmur"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    w = np.array([5, 3, 1])[y]                          # weight of each patient's true class
    p, u = pat["Present"].to_numpy(), pat["Unknown"].to_numpy()
    best_wa, best_t = -1.0, None
    for tp in grid:
        is_p = p >= tp
        is_u = ~is_p[:, None] & (u[:, None] >= grid[None, :])   # (n_patients, len(grid))
        pred = np.where(is_p[:, None], 0, np.where(is_u, 1, 2))
        wa = (w[:, None] * (pred == y[:, None])).sum(0) / w.sum()
        k = int(np.argmax(wa))
        if wa[k] > best_wa:
            best_wa, best_t = float(wa[k]), (float(tp), float(grid[k]))
    return best_t, best_wa


def cv_report(pat):
    """Nested threshold tuning over the CV folds of `pat`.

    Returns (per-fold table, pooled predictions, pooled report). The final thresholds for the
    test / hidden set are tune_thresholds(pat) on all folds."""
    rows, pred = [], pd.Series(index=pat.index, dtype=object)
    for k in sorted(pat["fold"].unique()):
        val, other = pat[pat["fold"] == k], pat[pat["fold"] != k]
        t, _ = tune_thresholds(other)
        pred[val.index] = predict_rule(val, *t)
        rows.append({"fold": k, "t_present": t[0], "t_unknown": t[1],
                     "WA (t=0.5)": weighted_accuracy(val["murmur"], predict_rule(val, .5, .5)),
                     "WA (tuned)": weighted_accuracy(val["murmur"], pred[val.index])})
    return pd.DataFrame(rows).set_index("fold"), pred, report(pat["murmur"], pred)


def summary(table, rep):
    return {"wa_mean": float(table["WA (tuned)"].mean()), "wa_std": float(table["WA (tuned)"].std()),
            "wa_pooled": rep["weighted_accuracy"], "recall": {c: float(v) for c, v in rep["recall"].items()},
            "confusion": rep["confusion"].tolist()}


def confusion_df(rep):
    return pd.DataFrame(rep["confusion"], index=[f"pred {c}" for c in CLASSES],
                        columns=[f"true {c}" for c in CLASSES])


# ---------------------------------------------------------------- saved predictions

def result_dir(name):
    """Folder of a run: results/<name> for the final models (cnn, rnn) and new runs,
    results/experiments/<name> for the iterations and comparisons already moved there."""
    exp = RESULTS_DIR / "experiments" / name
    return exp if exp.is_dir() else RESULTS_DIR / name


def save_oof(name, window_idx, probs, extra=None):
    """Save out-of-fold window probabilities (+ optional JSON summary) to result_dir(name)."""
    out = result_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(probs, columns=CLASSES, index=pd.Index(window_idx, name="window_idx"))
    df.to_csv(out / "oof.csv", float_format="%.5f")
    if extra is not None:
        (out / "cv_summary.json").write_text(json.dumps(extra, indent=2))
    return out


def load_oof(name):
    """(window_idx, probs) saved by save_oof."""
    df = pd.read_csv(result_dir(name) / "oof.csv", index_col="window_idx")
    return df.index.to_numpy(), df[CLASSES].to_numpy(dtype=np.float32)
