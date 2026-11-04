"""Load the pre-processed data (src.prepare_data) and serve it to the MLP / CNN.

Typical use in a training notebook:

    from src.datasets import load_processed, fold_masks, mlp_arrays, CNNWindows
    d = load_processed()
    tr, va, te = fold_masks(d.windows, val_fold=0)          # boolean masks over windows

    # MLP
    Xtr, ytr, Xva, yva, scaler = mlp_arrays(d, tr, va)      # standardised with train stats

    # CNN (torch)
    band_stats = mel_band_stats(d, tr)                      # train-only mean/std per mel band
    ds_tr = CNNWindows(d, tr, band_stats, random_crop=True)  # augmentation: random time offset
    ds_va = CNNWindows(d, va, band_stats)

Labels: `label_col="rec_label"` (default) uses the recording-level label derived from
`Murmur locations`; `label_col="murmur"` uses the raw patient label for every window.
Evaluation is always at patient level (see aggregate_to_patient).
"""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import CLASSES, ROOT
from src.features import Standardizer

PROC_DIR = ROOT / "data" / "processed"
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}  # Present=0, Unknown=1, Absent=2


@dataclass
class Processed:
    windows: pd.DataFrame
    recordings: pd.DataFrame
    X_mlp: np.ndarray
    mel: np.ndarray
    config: dict

    def __post_init__(self):
        self.rec_idx = self.recordings.set_index("recording")


def load_processed(proc_dir=PROC_DIR, mmap=True):
    """mel.npy is memory-mapped by default (it is the largest file)."""
    return Processed(
        windows=pd.read_csv(proc_dir / "windows.csv"),
        recordings=pd.read_csv(proc_dir / "recordings.csv"),
        X_mlp=np.load(proc_dir / "X_mlp.npy"),
        mel=np.load(proc_dir / "mel.npy", mmap_mode="r" if mmap else None),
        config=json.loads((proc_dir / "config.json").read_text()),
    )


def fold_masks(df, val_fold):
    """Train = CV folds except val_fold; val = val_fold; test = fold -1 (never for decisions)."""
    f = df["fold"].to_numpy()
    return (f >= 0) & (f != val_fold), f == val_fold, f == -1


def encode(labels):
    return np.array([CLASS_TO_IDX[c] for c in labels], dtype=np.int64)


def class_weights(y, n_classes=len(CLASSES), power=1.0):
    """Inverse-frequency weights (normalised to mean 1) for a weighted cross-entropy.
    power < 1 softens them (0 = unweighted)."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = (counts.sum() / (n_classes * np.maximum(counts, 1))) ** power
    return (w / w.mean()).astype(np.float32)


# ---------------------------------------------------------------- MLP

def mlp_arrays(d, train_mask, *other_masks, label_col="rec_label"):
    """Standardised MLP inputs. Returns X_tr, y_tr, (X, y for each other mask)..., scaler."""
    scaler = Standardizer(axis=0).fit(d.X_mlp[train_mask])
    out = []
    for m in (train_mask, *other_masks):
        out += [scaler.transform(d.X_mlp[m]), encode(d.windows.loc[m, label_col])]
    return (*out, scaler)


# ---------------------------------------------------------------- CNN

def mel_band_stats(d, train_mask):
    """Mean/std per mel band over all frames of the *training* recordings."""
    recs = set(d.windows.loc[train_mask, "recording"])
    r = d.recordings[d.recordings.recording.isin(recs)]
    s1 = np.zeros(d.mel.shape[0])
    s2 = np.zeros(d.mel.shape[0])
    n = 0
    for o, k in zip(r.mel_offset, r.mel_frames):
        x = np.asarray(d.mel[:, o:o + k], dtype=np.float64)
        s1 += x.sum(1)
        s2 += (x ** 2).sum(1)
        n += k
    mean = s1 / n
    std = np.sqrt(s2 / n - mean ** 2) + 1e-6
    return mean.astype(np.float32)[:, None], std.astype(np.float32)[:, None]


def mel_window(d, row, n_frames, offset_frames=0):
    """Slice one (n_mels, n_frames) window from mel.npy. Past the end of the recording the
    window is padded with the minimum dB value (silence)."""
    rec = d.rec_idx.loc[row.recording]
    start = int(row.mel_start) + offset_frames
    end_rec = int(rec.mel_offset + rec.mel_frames)
    start = max(int(rec.mel_offset), min(start, end_rec - 1))
    x = np.asarray(d.mel[:, start:min(start + n_frames, end_rec)], dtype=np.float32)
    if x.shape[1] < n_frames:
        x = np.pad(x, ((0, 0), (0, n_frames - x.shape[1])), constant_values=float(x.min()))
    return x


def cnn_arrays(d, mask, band_stats, label_col="rec_label"):
    """All windows of a mask as one array (n, 1, n_mels, frames) -- for evaluation or small data."""
    n_frames = d.config["frames_per_window"]
    mean, std = band_stats
    rows = d.windows[mask]
    X = np.stack([(mel_window(d, r, n_frames) - mean) / std for r in rows.itertuples()])
    return X[:, None].astype(np.float32), encode(rows[label_col])


try:
    import torch
    from torch.utils.data import Dataset

    class CNNWindows(Dataset):
        """Torch dataset of log-mel windows (1, n_mels, frames), standardised per mel band.

        random_crop=True shifts each window by a random offset of up to +-half a window
        every epoch, so the network sees different crops of the same recording (cheap
        time-shift augmentation).
        """

        def __init__(self, d, mask, band_stats, label_col="rec_label", random_crop=False):
            self.d = d
            self.rows = d.windows[mask].reset_index(drop=True)
            self.y = encode(self.rows[label_col])
            self.mean, self.std = band_stats
            self.n_frames = d.config["frames_per_window"]
            self.random_crop = random_crop

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            off = np.random.randint(-self.n_frames // 2, self.n_frames // 2) if self.random_crop else 0
            x = (mel_window(self.d, self.rows.iloc[i], self.n_frames, off) - self.mean) / self.std
            return torch.from_numpy(x[None].astype(np.float32)), int(self.y[i])

except ImportError:  # torch not installed: numpy helpers above still work
    pass


# ---------------------------------------------------------------- patient-level aggregation

def aggregate_to_patient(windows, probs, how="mean"):
    """Window probabilities (n, 3) -> one probability vector per patient.

    how="mean": average over all windows of the patient.
    how="max_present": average windows -> recording, then the patient takes the recording
                       with the highest P(Present) (a murmur heard at one location is enough).
    how="max": average windows -> recording, then per class the maximum over recordings for
               Present and Unknown and the mean for Absent (scores for the hierarchical rule,
               not a distribution).
    Returns a DataFrame indexed by patient_id with columns CLASSES.
    """
    p = pd.DataFrame(probs, columns=CLASSES)
    p["patient_id"] = windows["patient_id"].to_numpy()
    p["recording"] = windows["recording"].to_numpy()
    if how == "mean":
        return p.groupby("patient_id")[CLASSES].mean()
    rec = p.groupby(["patient_id", "recording"])[CLASSES].mean().reset_index()
    if how == "max":
        return rec.groupby("patient_id").agg(Present=("Present", "max"), Unknown=("Unknown", "max"),
                                             Absent=("Absent", "mean"))[CLASSES]
    idx = rec.groupby("patient_id")["Present"].idxmax()
    return rec.loc[idx].set_index("patient_id")[CLASSES]
