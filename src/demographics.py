"""Patient demographics as extra model inputs: wide branch of the CNN / RNN (--demo) and stacking (src.stacking).

Only fields present for every hidden-set patient are allowed (Age, Sex, Height, Weight, Pregnancy status,
Campaign). Selection in notebooks/01_eda.ipynb, section 10: age and height kept; weight redundant with height;
sex not associated with Murmur; pregnancy and campaign constant in the hidden set. Fields derived from the murmur
annotation (Murmur locations, Systolic/Diastolic murmur ..., Outcome) are never used: they leak the label.

Encoding (DemoEncoder, fitted on the training patients of each fold and saved with the model):
    age        one-hot over AGE_GROUPS (Neonate -> Infant; Young Adult / missing -> Other)
    height     z-score (training mean/std), missing -> training median, plus a 0/1 missing flag
    weight     same as height
    sex        1 = Male
    pregnancy  1 = True
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DATA_DIR, ROOT

AGE_GROUPS = ["Infant", "Child", "Adolescent", "Other"]
_AGE_MAP = {"Neonate": "Infant", "Infant": "Infant", "Child": "Child", "Adolescent": "Adolescent"}
ALL_FEATURES = ("age", "height", "weight", "sex", "pregnancy")
DEFAULT_FEATURES = ("age", "height")          # chosen in notebooks/01_eda.ipynb
TRAIN_TABLE = DATA_DIR / "training_data.csv"
_HIDDEN_DIRS = [ROOT / "data" / "hidden" / "CirCor_test_students", ROOT / "data" / "CirCor_test_students"]
HIDDEN_DIR = next((p for p in _HIDDEN_DIRS if p.is_dir()), _HIDDEN_DIRS[0])
HIDDEN_TABLE = HIDDEN_DIR / "test_data_participants.csv"


def read_table(path=TRAIN_TABLE):
    """One row per patient, indexed by the patient id as a string (works for training_data.csv and
    test_data_participants.csv)."""
    df = pd.read_csv(path)
    df.index = df["Patient ID"].astype(str)
    df.index.name = "patient_id"
    return df


class DemoEncoder:
    """Turns the demographic table into a float32 matrix. Statistics come from fit() (training patients)."""

    def __init__(self, features=DEFAULT_FEATURES):
        features = tuple(features)
        bad = [f for f in features if f not in ALL_FEATURES]
        if bad:
            raise ValueError(f"unknown demographic features {bad}; choose from {ALL_FEATURES}")
        self.features = features
        self.stats = {}

    # ------------------------------------------------------------------ fit / transform
    def fit(self, table):
        for f, col in (("height", "Height"), ("weight", "Weight")):
            if f in self.features:
                v = pd.to_numeric(table[col], errors="coerce").dropna()
                self.stats[f] = {"median": float(v.median()), "mean": float(v.mean()), "std": float(v.std() + 1e-6)}
        return self

    def transform(self, table):
        cols = []
        if "age" in self.features:
            g = table["Age"].map(_AGE_MAP).fillna("Other")
            cols += [(g == a).to_numpy(float) for a in AGE_GROUPS]
        for f, col in (("height", "Height"), ("weight", "Weight")):
            if f in self.features:
                s = self.stats[f]
                v = pd.to_numeric(table[col], errors="coerce")
                miss = v.isna().to_numpy(float)
                cols += [((v.fillna(s["median"]) - s["mean"]) / s["std"]).to_numpy(float), miss]
        if "sex" in self.features:
            cols.append((table["Sex"] == "Male").to_numpy(float))
        if "pregnancy" in self.features:
            cols.append(table["Pregnancy status"].astype(str).str.lower().eq("true").to_numpy(float))
        return np.stack(cols, axis=1).astype(np.float32)

    # ------------------------------------------------------------------ helpers
    @property
    def dim(self):
        return (4 * ("age" in self.features) + 2 * ("height" in self.features) + 2 * ("weight" in self.features)
                + ("sex" in self.features) + ("pregnancy" in self.features))

    def names(self):
        out = []
        if "age" in self.features:
            out += [f"age_{a}" for a in AGE_GROUPS]
        for f in ("height", "weight"):
            if f in self.features:
                out += [f"{f}_z", f"{f}_missing"]
        if "sex" in self.features:
            out.append("male")
        if "pregnancy" in self.features:
            out.append("pregnant")
        return out

    def state(self):
        return {"features": list(self.features), "stats": self.stats}

    @classmethod
    def from_state(cls, state):
        enc = cls(state["features"])
        enc.stats = state["stats"]
        return enc


def patient_matrix(encoder, table, patient_ids):
    """Encoded demographics for a list of patient ids (one row per id, in order)."""
    ids = pd.Index([str(p) for p in patient_ids])
    missing = ids.difference(table.index)
    if len(missing):
        raise KeyError(f"no demographics for patients {list(missing)[:10]}")
    return encoder.transform(table.loc[ids])


def fit_on_windows(features, table, windows):
    """DemoEncoder fitted on the patients of a set of windows (e.g. the training windows of a fold)."""
    ids = pd.unique(windows["patient_id"].astype(str))
    return DemoEncoder(features).fit(table.loc[ids])


def window_matrix(encoder, table, windows):
    """Encoded demographics repeated for every window (rows aligned with `windows`)."""
    pids = windows["patient_id"].astype(str)
    uniq = pd.unique(pids)
    per_patient = pd.DataFrame(patient_matrix(encoder, table, uniq), index=uniq)
    return per_patient.loc[pids.to_numpy()].to_numpy(np.float32)
