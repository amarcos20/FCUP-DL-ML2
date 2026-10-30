"""Acoustic feature extraction from PCG audio recordings (librosa + soundfile).

Extracts features per recording (MFCCs, spectral features, ZCR, RMS)
and aggregates them to the patient level (mean and std across recordings).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.data import DATA_DIR, load_recordings, load_split

FEATURES_DIR = DATA_DIR.parent / "features"
REC_FEATURES_FILE = FEATURES_DIR / "recording_features.csv"
PATIENT_FEATURES_FILE = FEATURES_DIR / "patient_features.csv"


def extract_single_recording_features(file_path):
    """Extract summary statistics of acoustic features from one .wav file."""
    try:
        y, sr = sf.read(file_path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if len(y) == 0:
            return None

        # Normalization
        max_abs = np.max(np.abs(y))
        if max_abs > 0:
            y = y / max_abs

        # Feature extraction
        # 1. MFCCs (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # 2. Delta MFCC
        mfcc_delta = librosa.feature.delta(mfcc)
        # 3. Spectral centroid
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        # 4. Spectral rolloff
        sr_roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        # 5. Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        # 6. RMS Energy
        rms = librosa.feature.rms(y=y)

        feats = {
            "duration": len(y) / sr,
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "sc_mean": float(np.mean(sc)),
            "sc_std": float(np.std(sc)),
            "rolloff_mean": float(np.mean(sr_roll)),
            "rolloff_std": float(np.std(sr_roll)),
        }

        for i in range(13):
            feats[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
            feats[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))
            feats[f"mfcc_delta_{i}_mean"] = float(np.mean(mfcc_delta[i]))
            feats[f"mfcc_delta_{i}_std"] = float(np.std(mfcc_delta[i]))

        return feats
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def extract_features_worker(row_tuple):
    idx, row = row_tuple
    feats = extract_single_recording_features(row["path"])
    if feats is not None:
        feats["recording"] = row["recording"]
        feats["patient_id"] = row["patient_id"]
        feats["location"] = row["location"]
        return feats
    return None


def extract_all_recordings(n_workers=4, out_file=REC_FEATURES_FILE):
    rec_df = load_recordings()
    print(f"Extracting features for {len(rec_df)} recordings with {n_workers} workers...")
    
    tasks = list(rec_df.iterrows())
    results = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for res in tqdm(executor.map(extract_features_worker, tasks), total=len(tasks)):
            if res is not None:
                results.append(res)

    out_df = pd.DataFrame(results)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_file, index=False)
    print(f"Saved recording features to {out_file} (shape: {out_df.shape})")
    return out_df


def build_patient_features(rec_features_file=REC_FEATURES_FILE, out_file=PATIENT_FEATURES_FILE):
    """Aggregate recording-level features to patient-level (mean and std across locations)."""
    if not Path(rec_features_file).exists():
        raise FileNotFoundError(f"{rec_features_file} not found. Run extract_all_recordings first.")

    rec_df = pd.read_csv(rec_features_file)
    split_df = load_split().reset_index()

    feature_cols = [c for c in rec_df.columns if c not in ["recording", "patient_id", "location"]]

    # Aggregate by patient_id: mean and std across available auscultation locations
    agg_mean = rec_df.groupby("patient_id")[feature_cols].mean().add_suffix("_patient_mean")
    agg_std = rec_df.groupby("patient_id")[feature_cols].std().fillna(0).add_suffix("_patient_std")
    agg_count = rec_df.groupby("patient_id")["recording"].count().rename("num_recordings")

    patient_df = pd.concat([agg_mean, agg_std, agg_count], axis=1).reset_index()

    # Merge with split (fold, murmur, group)
    merged = pd.merge(split_df, patient_df, on="patient_id", how="inner")
    
    out_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_file, index=False)
    print(f"Saved patient-level features to {out_file} (shape: {merged.shape})")
    return merged


def get_patient_dataset():
    """Load patient-level features dataframe, extracting if necessary."""
    if not PATIENT_FEATURES_FILE.exists():
        if not REC_FEATURES_FILE.exists():
            extract_all_recordings()
        build_patient_features()
    return pd.read_csv(PATIENT_FEATURES_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    extract_all_recordings(n_workers=args.workers)
    build_patient_features()
