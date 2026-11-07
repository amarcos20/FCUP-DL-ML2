"""Predict the murmur label of new patients (e.g. the hidden data of the bonus) from their .wav files.

The recordings go through exactly the same pipeline as in training:
    load at 4 kHz, band-pass 25-1000 Hz, z-score (src.features.load_audio)
    -> log-mel 40 bands x 10 ms of the whole recording (src.features.logmel)
    -> 4 s windows with 2 s hop (src.features.window_starts), 401 frames each
    -> for each of the 5 fold models: per-band standardisation with that fold's training statistics,
       softmax probabilities per window; the 5 models are averaged (ensemble)
    -> mean over the windows of each recording, then per patient the maximum of P(Present) and
       P(Unknown) over its recordings (src.datasets.aggregate_to_patient, how="max")
    -> hierarchical rule with the final thresholds of the model (results/<model>/cv_summary.json):
       Present if P(Present) >= t1, else Unknown if P(Unknown) >= t2, else Absent.

File names must follow the CirCor convention <patient_id>_<location>[_<n>].wav (e.g. 50052_AV.wav,
50052_MV_1.wav): the patient id is the part before the first "_". Sub-folders are searched too.

Usage:
    python -m src.predict --wav-dir path/to/hidden_wavs                       # CNN -> predictions_cnn.csv
    python -m src.predict --wav-dir path/to/hidden_wavs --model rnn --out predictions_rnn.csv

Output: CSV with one row per patient and the columns patient_id,murmur (the format of the bonus).
--probs also saves the patient scores (P(Present), P(Unknown)) next to it, for checking.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.cnn import device
from src.data import CLASSES
from src.datasets import aggregate_to_patient
from src.evaluate import RESULTS_DIR, predict_rule
from src.features import AUDIO, MEL, frames_per_window, load_audio, logmel, window_starts


def find_wavs(wav_dir):
    files = sorted(Path(wav_dir).rglob("*.wav"))
    if not files:
        raise SystemExit(f"no .wav files found in {wav_dir}")
    rows = [{"path": p, "recording": p.stem, "patient_id": p.stem.split("_")[0]} for p in files]
    return pd.DataFrame(rows)


def recording_windows(path):
    """(n_windows, 40, 401) log-mel windows of one recording, cut as in training (src.prepare_data +
    src.datasets.mel_window): window start in frames = round(start_sample / hop); past the end of the
    recording the window is padded with its minimum dB value."""
    y = load_audio(path)
    mel = logmel(y)
    n = frames_per_window()
    out = []
    for s in window_starts(len(y)):
        f0 = min(int(round(s / MEL.hop_length)), mel.shape[1] - 1)
        x = mel[:, f0:f0 + n]
        if x.shape[1] < n:
            x = np.pad(x, ((0, 0), (0, n - x.shape[1])), constant_values=float(x.min()))
        out.append(x)
    return np.stack(out).astype(np.float16).astype(np.float32)   # mel.npy is stored as float16


def load_model_runs(model):
    """Fold models (each with its band statistics) and the final thresholds of results/<model>."""
    folder = RESULTS_DIR / model
    if model.startswith("rnn"):
        from src.rnn import load_runs
        runs, _ = load_runs(folder)
    else:
        from src.cnn import load_runs
        runs = load_runs(folder)
    if not runs:
        raise SystemExit(f"no fold models (fold*.pt) in {folder}")
    thresholds = json.loads((folder / "cv_summary.json").read_text())["thresholds"]
    return runs, thresholds


def window_probs(runs, X, batch=256):
    """Ensemble (mean over fold models) of the softmax probabilities of windows X (n, 40, 401)."""
    dev = device()
    out = []
    for r in runs:
        mean, std = r["band_stats"]
        Xs = torch.from_numpy(((X - mean) / std)[:, None].astype(np.float32))
        m = r["model"].to(dev).eval()
        with torch.no_grad():
            p = [torch.softmax(m(Xs[i:i + batch].to(dev)), 1).cpu().numpy() for i in range(0, len(Xs), batch)]
        out.append(np.concatenate(p))
    return np.mean(out, axis=0)


def predict(wav_dir, model="cnn"):
    """Returns (patient scores DataFrame, labels Series) for every patient in wav_dir."""
    recs = find_wavs(wav_dir)
    runs, (t1, t2) = load_model_runs(model)
    win_rows, probs = [], []
    for r in tqdm(recs.itertuples(), total=len(recs), desc="recordings"):
        X = recording_windows(r.path)
        probs.append(window_probs(runs, X))
        win_rows += [{"recording": r.recording, "patient_id": r.patient_id}] * len(X)
    pat = aggregate_to_patient(pd.DataFrame(win_rows), np.concatenate(probs), how="max")
    labels = pd.Series(predict_rule(pat, t1, t2), index=pat.index, name="murmur")
    return pat, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True, help="folder with the .wav files (searched recursively)")
    ap.add_argument("--model", default="cnn", help="folder under results/ with fold*.pt (cnn or rnn)")
    ap.add_argument("--out", help="output CSV (default predictions_<model>.csv)")
    ap.add_argument("--probs", action="store_true", help="also save the patient scores")
    a = ap.parse_args()
    out = Path(a.out or f"predictions_{a.model}.csv")
    print(f"model: results/{a.model} | device: {device()} | sample rate {AUDIO.sr} Hz", flush=True)

    pat, labels = predict(a.wav_dir, a.model)
    labels.rename_axis("patient_id").reset_index().to_csv(out, index=False)
    print(f"{len(labels)} patients -> {out}")
    print(labels.value_counts().reindex(CLASSES, fill_value=0).to_string())
    if a.probs:
        p = out.with_name(out.stem + "_scores.csv")
        pat[CLASSES].rename_axis("patient_id").to_csv(p, float_format="%.5f")
        print("scores ->", p)


if __name__ == "__main__":
    main()
