"""Window-level feature extraction for the MLP.

Each recording is band-pass filtered, z-score normalised, and cut into fixed-length
windows (default 5 s, 50% overlap; the last incomplete window is dropped). Every window
becomes one fixed-size vector: mean and std over time of frame-level features
(MFCC + deltas, log-mel bands, spectral descriptors).

Output: one row per window with metadata (patient_id, recording, location, t_start,
fold, murmur, rec_label) followed by the feature columns.

`rec_label` is the recording-level label: for Present patients, a recording at a location
not listed in `Murmur locations` gets "Absent" (the murmur was not heard there);
otherwise it is the patient label. `murmur` keeps the patient label.

Usage: python -m src.features [--win 5 --hop 2.5]
"""
import argparse

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt

from src.data import DATA_DIR, load_patients, load_recordings, load_split

FEAT_DIR = DATA_DIR.parent / "features"

SR = 4000
BAND = (25, 400)  # Hz; heart sounds and most murmurs
N_FFT = 512       # 128 ms -> 7.8 Hz resolution
HOP = 64          # 16 ms
N_MFCC = 20
N_MELS = 32

META_COLS = ["patient_id", "recording", "location", "fold", "murmur", "rec_label", "t_start"]


def preprocess(y, sr=SR):
    """Band-pass filter and z-score normalise a whole recording."""
    sos = butter(4, BAND, btype="bandpass", fs=sr, output="sos")
    y = sosfiltfilt(sos, y)
    return ((y - y.mean()) / (y.std() + 1e-8)).astype(np.float32)


def windows(y, sr=SR, win=5.0, hop=2.5):
    """Yield (t_start, segment) for full-length windows only."""
    n, h = int(win * sr), int(hop * sr)
    for s in range(0, len(y) - n + 1, h):
        yield s / sr, y[s:s + n]


def frame_features(y, sr=SR):
    """Frame-level features -> dict name -> array (n_coeffs, n_frames)."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    mel = librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=N_MELS, fmin=BAND[0], fmax=BAND[1])
    logmel = librosa.power_to_db(mel)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=N_MFCC)
    return {
        "mfcc": mfcc,
        "dmfcc": librosa.feature.delta(mfcc),
        "logmel": logmel,
        "centroid": librosa.feature.spectral_centroid(S=S, sr=sr),
        "bandwidth": librosa.feature.spectral_bandwidth(S=S, sr=sr),
        "rolloff": librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85),
        "flatness": librosa.feature.spectral_flatness(S=S),
        "zcr": librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP),
        "rms": librosa.feature.rms(S=S, frame_length=N_FFT),
    }


def window_vector(y, sr=SR):
    """Summarise frame-level features over time (mean, std) -> flat dict."""
    out = {}
    for name, F in frame_features(y, sr).items():
        for k, row in enumerate(F):
            suffix = f"{name}{k}" if len(F) > 1 else name
            out[f"{suffix}_mean"] = row.mean()
            out[f"{suffix}_std"] = row.std()
    return out


def recording_rows(path, win, hop):
    y, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"{path}: unexpected sample rate {sr}"
    y = preprocess(y, sr)
    return [{"t_start": t, **window_vector(seg, sr)} for t, seg in windows(y, sr, win, hop)]


def recording_labels(rec, patients):
    """Recording-level label using `Murmur locations` for Present patients."""
    mur_locs = patients["Murmur locations"].fillna("").str.split("+").apply(set)
    rec_label = rec["murmur"].copy()
    present = rec["murmur"] == "Present"
    heard = [loc in mur_locs[p] for p, loc in zip(rec.patient_id, rec.location)]
    rec_label[present & ~np.array(heard)] = "Absent"
    return rec_label


def build(win=5.0, hop=2.5, n_jobs=-1):
    rec = load_recordings()
    rec["rec_label"] = recording_labels(rec, load_patients())
    rec["fold"] = rec["patient_id"].map(load_split()["fold"])

    per_rec = Parallel(n_jobs=n_jobs)(delayed(recording_rows)(p, win, hop) for p in rec.path)
    frames = []
    for (_, r), rows in zip(rec.iterrows(), per_rec):
        df = pd.DataFrame(rows)
        for c in reversed(META_COLS[:-1]):  # t_start already present
            df.insert(0, c, r[c])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    feats = feature_columns(df)
    df[feats] = df[feats].astype(np.float32)
    return df


def feature_columns(df):
    return [c for c in df.columns if c not in META_COLS]


def feature_file(win=5.0, hop=2.5):
    return FEAT_DIR / f"windows_w{win:g}_h{hop:g}.pkl"


def load_features(win=5.0, hop=2.5):
    """Cached window table; returns (df, list of feature column names)."""
    df = pd.read_pickle(feature_file(win, hop))
    return df, feature_columns(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=float, default=5.0)
    ap.add_argument("--hop", type=float, default=2.5)
    args = ap.parse_args()

    df = build(args.win, args.hop)
    out = feature_file(args.win, args.hop)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)

    print(f"{len(df)} windows from {df.recording.nunique()} recordings, "
          f"{len(feature_columns(df))} features -> {out}")
    print(pd.crosstab(df["fold"], df["rec_label"], margins=True))


if __name__ == "__main__":
    main()
