"""Loading CirCor metadata, recordings, and the fixed patient-level split."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "circor"
WAV_DIR = DATA_DIR / "training_data"
SPLIT_FILE = ROOT / "splits" / "split.csv"


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
    return rec


def load_split(split_file=SPLIT_FILE):
    """patient_id -> fold (0..K-1 for train/val CV) or -1 for the held-out test set."""
    return pd.read_csv(split_file).set_index("patient_id")
