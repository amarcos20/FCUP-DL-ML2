"""Patient-level murmur metrics, matching the PhysioNet Challenge 2022 definition.

Reference: https://github.com/physionetchallenges/evaluation-2022 (compute_weighted_accuracy)
"""
import numpy as np

CLASSES = ["Present", "Unknown", "Absent"]
# Weight of each *true* class: Present=5, Unknown=3, Absent=1
CLASS_WEIGHTS = np.array([5, 3, 1])


def confusion_matrix(y_true, y_pred, classes=CLASSES):
    """m[i, j] = number of patients predicted as classes[i] whose true class is classes[j]
    (rows = predicted, columns = true, as in the project statement)."""
    idx = {c: k for k, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[idx[p], idx[t]] += 1
    return m


def weighted_accuracy(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    w = np.tile(CLASS_WEIGHTS, (len(CLASSES), 1))  # weight by column (true class)
    denom = np.sum(w * m)
    return float(np.trace(w * m) / denom) if denom else float("nan")


def report(y_true, y_pred):
    """Weighted accuracy, confusion matrix, and per-class recall (additional metric)."""
    m = confusion_matrix(y_true, y_pred)
    col = m.sum(axis=0)
    recall = {c: (m[k, k] / col[k] if col[k] else float("nan")) for k, c in enumerate(CLASSES)}
    return {"weighted_accuracy": weighted_accuracy(y_true, y_pred), "confusion": m, "recall": recall}


if __name__ == "__main__":
    # Sanity checks
    y = ["Present"] * 10 + ["Unknown"] * 5 + ["Absent"] * 30
    assert weighted_accuracy(y, y) == 1.0
    always_absent = weighted_accuracy(y, ["Absent"] * len(y))
    assert abs(always_absent - 30 / (50 + 15 + 30)) < 1e-12
    print("ok; always-Absent WA =", round(always_absent, 3))
