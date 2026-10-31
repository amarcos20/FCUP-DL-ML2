"""Loading CirCor metadata, recordings, segmentations and the fixed patient-level split."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "splits" / "split.csv"

# The dataset may live in data/circor/ (download_data.py) or in the folder created by
# unzipping the official PhysioNet zip into data/. Use whichever exists.
_CANDIDATES = [
    ROOT / "data" / "circor",
    ROOT / "data" / "the-circor-digiscope-phonocardiogram-dataset-1.0.3",
]
DATA_DIR = next((p for p in _CANDIDATES if (p / "training_data.csv").exists()), _CANDIDATES[0])
WAV_DIR = DATA_DIR / "training_data"

CLASSES = ["Present", "Unknown", "Absent"]
LOCATIONS = ["AV", "PV", "TV", "MV", "Phc"]
# Segmentation states in the .tsv files
STATES = {0: "unannotated", 1: "S1", 2: "systole", 3: "S2", 4: "diastole"}


def load_patients(data_dir=DATA_DIR):
    """One row per patient (training_data.csv), indexed by Patient ID."""
    df = pd.read_csv(Path(data_dir) / "training_data.csv")
    return df.set_index("Patient ID")


def load_recordings(data_dir=DATA_DIR):
    """One row per .wav file: patient_id, location (AV/PV/TV/MV/Phc), path, plus patient murmur label.

    Files are named {patient}_{location}.wav, or {patient}_{location}_{n}.wav when a
    location was recorded more than once.
    """
    wav_dir = Path(data_dir) / "training_data"
    rows = []
    for p in sorted(wav_dir.glob("*.wav")):
        parts = p.stem.split("_")
        rows.append({"patient_id": int(parts[0]), "location": parts[1], "recording": p.stem, "path": str(p)})
    rec = pd.DataFrame(rows)
    patients = load_patients(data_dir)
    rec["murmur"] = rec["patient_id"].map(patients["Murmur"])
    rec["rec_label"] = recording_labels(rec, patients)
    return rec


def recording_labels(rec, patients):
    """Recording-level label derived from the patient label and `Murmur locations`.

    - Absent / Unknown patients: every recording inherits the patient label.
    - Present patients: a recording is `Present` only if its location is listed in
      `Murmur locations`; otherwise the murmur was not heard there -> `Absent`.
    The patient label is still what we evaluate on; this cleaner label is only for training.
    """
    mur_locs = patients["Murmur locations"].fillna("").str.split("+").apply(set)
    out = []
    for pid, loc, lab in zip(rec["patient_id"], rec["location"], rec["murmur"]):
        if lab == "Present":
            out.append("Present" if loc in mur_locs[pid] else "Absent")
        else:
            out.append(lab)
    return out


def load_segmentation(recording, data_dir=DATA_DIR):
    """Segmentation of one recording: DataFrame with start, end (seconds) and state (0..4)."""
    path = Path(data_dir) / "training_data" / f"{recording}.tsv"
    return pd.read_csv(path, sep="\t", header=None, names=["start", "end", "state"])


def segmentation_stats(seg):
    """Summary of a segmentation: annotated span, heart rate and mean phase durations."""
    ann = seg[seg.state > 0]
    s1 = ann[ann.state == 1].start.to_numpy()
    rr = np.diff(s1)
    dur = (ann.end - ann.start).groupby(ann.state).mean()
    return {
        "ann_start": ann.start.min() if len(ann) else np.nan,
        "ann_end": ann.end.max() if len(ann) else np.nan,
        "n_cycles": len(s1),
        "heart_rate": 60 / np.median(rr) if len(rr) else np.nan,
        "rr_cv": rr.std() / rr.mean() if len(rr) > 1 else np.nan,
        **{f"dur_{STATES[k]}": dur.get(k, np.nan) for k in (1, 2, 3, 4)},
    }


def load_split(split_file=SPLIT_FILE):
    """patient_id -> fold (0..K-1 for train/val CV) or -1 for the held-out test set."""
    return pd.read_csv(split_file).set_index("patient_id")
