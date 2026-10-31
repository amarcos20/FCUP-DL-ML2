"""Convolutional Neural Network (CNN 2D) on Log-Mel Spectrograms — High-Speed CPU.

Converts heart sound recordings into 2D Log-Mel Spectrograms (64 Mel frequency bands
focused on 25-400 Hz cardiac range). Slices recordings into 5-second windows (1 x 64 x 312),
trains a strided 2D ConvNet with BatchNorm, Dropout, and vectorized SpecAugment data augmentation,
and aggregates window predictions into patient-level diagnoses with threshold tuning.

Optimizations:
- Strided convolutions (16 -> 32 -> 64 -> 128 channels)
- Batch size 256 for rapid CPU vectorization (~15s per epoch)
- max_epochs 8, patience 3 (converges quickly on Log-Mel spectrograms)
- Unbuffered live stdout (flush=True) for instant real-time progress in Jupyter
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import DataLoader, TensorDataset
import torchaudio

from src.data import DATA_DIR, load_patients, load_recordings, load_split
from src.metrics import CLASSES, weighted_accuracy, report

FEAT_DIR = DATA_DIR.parent / "features"
CNN_CACHE_FILE = FEAT_DIR / "cnn_spectrogram_windows.pt"

SR = 4000
BAND = (25, 400)      # Hz; cardiac acoustic range
N_FFT = 512          # 128 ms window
HOP_LEN = 64         # 16 ms hop -> ~62.5 fps
N_MELS = 64          # 64 Mel frequency bins
WIN_SEC = 5.0        # 5 seconds per window
HOP_SEC = 2.5        # 50% overlap (2.5 seconds hop)
WIN_FRAMES = int(WIN_SEC * SR / HOP_LEN)  # ~312 frames

DEFAULT_CNN_CFG = dict(
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=256,
    max_epochs=8,
    patience=3,
    weight_power=1.0,
    dropout=0.3,
    spec_augment=True,
    seed=42,
)


# -----------------------------------------------------------------------------
# 1. Audio Preprocessing and Log-Mel Spectrogram Extraction
# -----------------------------------------------------------------------------

def bandpass_filter(y, sr=SR, band=BAND):
    """Band-pass filter (Butterworth order 4) and normalize."""
    sos = butter(4, band, btype="bandpass", fs=sr, output="sos")
    y_filt = sosfiltfilt(sos, y).astype(np.float32)
    return ((y_filt - y_filt.mean()) / (y_filt.std() + 1e-8)).astype(np.float32)


def get_mel_transform():
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        n_mels=N_MELS,
        f_min=BAND[0],
        f_max=BAND[1],
        power=2.0,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()
    return nn.Sequential(mel_transform, amp_to_db)


def compute_recording_spectrogram(path, mel_pipe):
    """Load, filter, and compute full Log-Mel spectrogram for a recording."""
    y, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"Unexpected sample rate {sr} in {path}"
    y_norm = bandpass_filter(y, sr=sr)
    t_y = torch.from_numpy(y_norm).unsqueeze(0)  # [1, T]
    with torch.no_grad():
        mel = mel_pipe(t_y)  # [1, 64, n_frames]
    return mel


def slice_spectrogram_windows(full_mel, win_sec=WIN_SEC, hop_sec=HOP_SEC):
    """Slice full spectrogram into fixed-length windows."""
    n_frames = full_mel.shape[-1]
    w_frames = int(win_sec * SR / HOP_LEN)
    h_frames = int(hop_sec * SR / HOP_LEN)

    windows = []
    for start in range(0, n_frames - w_frames + 1, h_frames):
        seg = full_mel[:, :, start : start + w_frames]
        # Standardize window
        seg_norm = (seg - seg.mean()) / (seg.std() + 1e-6)
        windows.append((start * HOP_LEN / SR, seg_norm))
    return windows


def build_cnn_dataset(cache_file=CNN_CACHE_FILE):
    """Extract Log-Mel spectrogram windows for all recordings and cache tensor table."""
    rec = load_recordings()
    patients = load_patients()
    split = load_split()

    # Medical recording-level labels
    mur_locs = patients["Murmur locations"].fillna("").str.split("+").apply(set)
    rec_labels = rec["murmur"].copy()
    present_mask = rec["murmur"] == "Present"
    heard = [loc in mur_locs[p] for p, loc in zip(rec.patient_id, rec.location)]
    rec_labels[present_mask & ~np.array(heard)] = "Absent"
    rec["rec_label"] = rec_labels
    rec["fold"] = rec["patient_id"].map(split["fold"])

    mel_pipe = get_mel_transform()

    print(f"Extracting Log-Mel Spectrogram windows from {len(rec)} recordings...", flush=True)
    spec_tensors = []
    meta_rows = []

    for idx, r in rec.iterrows():
        try:
            full_mel = compute_recording_spectrogram(r["path"], mel_pipe)
            wins = slice_spectrogram_windows(full_mel)
            for t_start, seg in wins:
                spec_tensors.append(seg)
                meta_rows.append({
                    "patient_id": r["patient_id"],
                    "recording": r["recording"],
                    "location": r["location"],
                    "fold": r["fold"],
                    "murmur": r["murmur"],
                    "rec_label": r["rec_label"],
                    "t_start": t_start,
                })
        except Exception as e:
            print(f"Error processing {r['path']}: {e}", flush=True)

    meta_df = pd.DataFrame(meta_rows)
    all_specs = torch.stack(spec_tensors)  # [N_windows, 1, 64, 312]

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta_df, "specs": all_specs}, cache_file)
    print(f"Saved {len(meta_df)} CNN windows to {cache_file} (tensor shape: {all_specs.shape})", flush=True)
    return meta_df, all_specs


def load_cnn_dataset(cache_file=CNN_CACHE_FILE):
    """Load cached spectrogram windows or build if missing."""
    if not cache_file.exists():
        return build_cnn_dataset(cache_file)
    data = torch.load(cache_file, weights_only=False)
    meta = data["meta"].copy()
    specs = data["specs"].clone().float()
    return meta, specs


# -----------------------------------------------------------------------------
# 2. Fast Vectorized SpecAugment
# -----------------------------------------------------------------------------

def apply_spec_augment(x, max_freq_mask=8, max_time_mask=24):
    """Fast in-place batch masking on 4D tensor [B, C, F, T]."""
    x = x.clone()
    b, c, f, t = x.shape
    f_len = int(torch.randint(2, max_freq_mask + 1, (1,)).item())
    f0 = int(torch.randint(0, f - f_len + 1, (1,)).item())
    x[:, :, f0 : f0 + f_len, :] = 0.0

    t_len = int(torch.randint(4, max_time_mask + 1, (1,)).item())
    t0 = int(torch.randint(0, t - t_len + 1, (1,)).item())
    x[:, :, :, t0 : t0 + t_len] = 0.0
    return x


# -----------------------------------------------------------------------------
# 3. 2D CNN Architecture (Strided ConvNet + BatchNorm + Global Pooling)
# -----------------------------------------------------------------------------

class HeartMurmurCNN(nn.Module):
    """CPU-optimized 2D Convolutional Neural Network for PCG Spectrograms."""
    def __init__(self, n_classes=len(CLASSES), dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: [B, 1, 64, 312] -> [B, 16, 32, 156]
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Block 2: [B, 16, 32, 156] -> [B, 32, 16, 78]
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 3: [B, 32, 16, 78] -> [B, 64, 8, 39]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 4: [B, 64, 8, 39] -> [B, 128, 4, 20]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Global average pooling: length-invariant [B, 128, 1, 1]
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        return self.classifier(feat)


# -----------------------------------------------------------------------------
# 4. Training, Validation and Cross-Validation
# -----------------------------------------------------------------------------

def encode_labels(labels):
    return np.array([CLASSES.index(l) for l in labels])


def compute_class_weights(y, power=1.0):
    counts = np.bincount(y, minlength=len(CLASSES))
    w = (counts.sum() / (len(CLASSES) * counts)) ** power
    return torch.tensor(w, dtype=torch.float32)


def predict_cnn_proba(model, specs, batch_size=256, device="cpu"):
    """Predict softmax probabilities in mini-batches."""
    model.eval()
    loader = DataLoader(TensorDataset(specs), batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            out = torch.softmax(model(batch), dim=1)
            probs.append(out.cpu().numpy())
    return np.concatenate(probs, axis=0)


def train_cnn(train_specs, train_labels, val_specs, val_labels, cfg, fold_idx=0):
    """Train CNN with live epoch progress logging and early stopping."""
    torch.manual_seed(cfg["seed"] + fold_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HeartMurmurCNN(dropout=cfg["dropout"]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=compute_class_weights(train_labels, cfg["weight_power"]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    train_ds = TensorDataset(train_specs, torch.tensor(train_labels, dtype=torch.long))
    val_ds = TensorDataset(val_specs, torch.tensor(val_labels, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    best_loss = float("inf")
    best_weights = None
    wait = 0
    history = []

    print(f"\n>>> [Fold {fold_idx}] Iniciando Treino ({len(train_ds)} janelas treino, {len(val_ds)} janelas validacao) <<<", flush=True)

    for epoch in range(cfg["max_epochs"]):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            if cfg["spec_augment"]:
                x_b = apply_spec_augment(x_b)
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = loss_fn(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_b)
        train_loss /= len(train_ds)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                logits = model(x_v)
                loss = loss_fn(logits, y_v)
                val_loss += loss.item() * len(x_v)
                correct += (logits.argmax(dim=1) == y_v).sum().item()
                total += len(y_v)
        val_loss /= len(val_ds)
        val_acc = correct / total

        is_best = val_loss < best_loss
        star = " [Melhor Loss]" if is_best else ""
        print(f"  Fold {fold_idx} | Epoca {epoch+1:02d}/{cfg['max_epochs']} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}{star}", flush=True)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

        if is_best:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"  Fold {fold_idx} | Early stopping ativado na epoca {epoch+1}", flush=True)
                break

    model.load_state_dict(best_weights)
    return model, pd.DataFrame(history)


def cross_validate_cnn(meta_df, specs, cfg=DEFAULT_CNN_CFG, folds=range(5)):
    """Run 5-Fold Patient-Level Cross Validation for the CNN."""
    dev_mask = (meta_df["fold"] >= 0).values
    dev_meta = meta_df[dev_mask].reset_index(drop=True)
    dev_specs = specs[dev_mask].clone()

    oof_probs = np.zeros((len(dev_meta), len(CLASSES)), dtype=np.float32)
    runs = []

    labels_encoded = encode_labels(dev_meta["rec_label"])
    folds_arr = dev_meta["fold"].values

    for k in folds:
        tr = folds_arr != k
        va = folds_arr == k

        tr_specs, tr_labels = dev_specs[tr], labels_encoded[tr]
        va_specs, va_labels = dev_specs[va], labels_encoded[va]

        model, hist = train_cnn(tr_specs, tr_labels, va_specs, va_labels, cfg, fold_idx=k)
        oof_probs[va] = predict_cnn_proba(model, va_specs)
        runs.append({"fold": k, "model": model, "history": hist})

        best = hist.loc[hist.val_loss.idxmin()]
        print(f"==> Fold {k} Concluido! Melhor Epoca: {int(best.epoch)+1} | Val Loss: {best.val_loss:.4f} | Val Acc: {best.val_acc:.4f}\n", flush=True)

    return dev_meta, oof_probs, runs


# -----------------------------------------------------------------------------
# 5. Patient-Level Aggregation and Threshold Tuning
# -----------------------------------------------------------------------------

def patient_scores(win_df, probs):
    """Aggregate window probabilities to patient-level scores."""
    d = win_df[["patient_id", "recording", "murmur"]].copy()
    d[CLASSES] = probs
    rec = d.groupby(["patient_id", "recording"]).agg({**{c: "mean" for c in CLASSES}, "murmur": "first"})
    pat = rec.groupby("patient_id").agg(
        p_present=("Present", "max"),
        p_unknown=("Unknown", "max"),
        p_absent=("Absent", "mean"),
        murmur=("murmur", "first"),
    )
    return pat


def predict_rule(pat, t_present, t_unknown):
    """Hierarchical decision rule for clinical prediction."""
    return np.where(
        pat.p_present >= t_present,
        "Present",
        np.where(pat.p_unknown >= t_unknown, "Unknown", "Absent"),
    )


def tune_thresholds(pat, grid=np.round(np.arange(0.05, 1.0, 0.025), 3)):
    """Grid search of (t_present, t_unknown) maximising the weighted accuracy on `pat`."""
    best_wa, best_t = -1.0, None
    for tp in grid:
        for tu in grid:
            wa = weighted_accuracy(pat.murmur, predict_rule(pat, tp, tu))
            if wa > best_wa:
                best_wa, best_t = wa, (tp, tu)
    return best_t, best_wa


def predict_ensemble_cnn(runs, specs, batch_size=256):
    """Average window probabilities across all 5 fold CNN models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_fold_probs = [predict_cnn_proba(r["model"].to(device), specs, batch_size=batch_size, device=device) for r in runs]
    return np.mean(all_fold_probs, axis=0)


if __name__ == "__main__":
    meta_df, specs = build_cnn_dataset()
    print("CNN Dataset successfully created!", flush=True)
