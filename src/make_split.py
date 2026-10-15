"""Create the fixed patient-level split used by every model.

- 20% of patients -> held-out test set (fold = -1), never used for any decision.
- remaining 80% -> 5 stratified folds (fold = 0..4) for validation / model selection.

Stratified by Murmur label. Patients with an `Additional ID` are the same child seen in
both screening campaigns, so they are kept in the same group to avoid leakage.

Usage: python -m src.make_split
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data import SPLIT_FILE, load_patients

SEED = 42
N_FOLDS = 5
TEST_FRACTION = 0.2


def main():
    df = load_patients().reset_index()
    add = df["Additional ID"]
    df["group"] = np.where(add.notna(), np.minimum(df["Patient ID"], add.fillna(0).astype(int)), df["Patient ID"])

    # Test split: one fold of a 5-fold stratified group split = 20%
    outer = StratifiedGroupKFold(n_splits=round(1 / TEST_FRACTION), shuffle=True, random_state=SEED)
    dev_idx, test_idx = next(outer.split(df, df["Murmur"], df["group"]))
    df["fold"] = -1
    dev = df.iloc[dev_idx]
    inner = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, val_idx) in enumerate(inner.split(dev, dev["Murmur"], dev["group"])):
        df.loc[dev.index[val_idx], "fold"] = k

    out = df.rename(columns={"Patient ID": "patient_id", "Murmur": "murmur"})[["patient_id", "murmur", "group", "fold"]]
    SPLIT_FILE.parent.mkdir(exist_ok=True)
    out.to_csv(SPLIT_FILE, index=False)

    print(pd.crosstab(out["fold"], out["murmur"], margins=True))
    assert out.groupby("group")["fold"].nunique().max() == 1, "group split across folds"
    print(f"saved {SPLIT_FILE}")


if __name__ == "__main__":
    main()
