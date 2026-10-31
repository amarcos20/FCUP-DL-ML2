"""Run the full pre-processing pipeline and save model-ready arrays to data/processed/.

Usage:
    python -m src.prepare_data              # all recordings, all CPU cores
    python -m src.prepare_data --jobs 1     # single process (easier to debug)

Outputs (data/processed/):
    recordings.csv  one row per recording: patient, location, labels, fold, duration,
                    and where its log-mel lives in mel.npy (mel_offset, mel_frames)
    windows.csv     one row per 4 s window: recording, patient_id, start_s, labels, fold,
                    mel_start (frame index into mel.npy) -> used by both models
    X_mlp.npy       (n_windows, n_features) float32 -- MLP input, one row per window
    mel.npy         (n_mels, total_frames) float16  -- log-mel of every recording, concatenated
                    along time; CNN windows are sliced from it on the fly (src.datasets)
    config.json     every pre-processing parameter + feature names

Nothing here is fitted on the data (no normalisation statistics), so there is no leakage:
standardisation is fitted on the training folds at training time (src.features.Standardizer).
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data import DATA_DIR, ROOT, load_recordings, load_split
from src.features import (AUDIO, MEL, config_dict, frames_per_window, load_audio, logmel,
                          mlp_feature_names, mlp_features, split_windows)

OUT_DIR = ROOT / "data" / "processed"


def process_recording(path):
    """All features of one recording: duration, full log-mel, window starts, MLP features."""
    y = load_audio(path)
    mel = logmel(y)                                  # (n_mels, frames) of the whole recording
    wins, starts = split_windows(y)                  # (n_win, window_len)
    X = np.stack([mlp_features(w) for w in wins])    # (n_win, n_features)
    return len(y) / AUDIO.sr, mel, starts, X


def prepare_data(jobs=None, out_dir=OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {DATA_DIR}")
    rec = load_recordings()
    split = load_split()
    rec["fold"] = rec["patient_id"].map(split["fold"])
    print(f"{len(rec)} recordings from {rec.patient_id.nunique()} patients")

    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    if jobs == 1:
        results = [process_recording(p) for p in tqdm(rec.path, desc="features")]
    else:
        with ProcessPoolExecutor(jobs) as ex:
            results = list(tqdm(ex.map(process_recording, rec.path, chunksize=8), total=len(rec), desc="features"))

    # --- assemble
    mels, X_all, win_rows = [], [], []
    offset = 0
    rec["duration"], rec["mel_offset"], rec["mel_frames"], rec["n_windows"] = 0.0, 0, 0, 0
    for i, (dur, mel, starts, X) in enumerate(results):
        r = rec.iloc[i]
        rec.loc[rec.index[i], ["duration", "mel_offset", "mel_frames", "n_windows"]] = [dur, offset, mel.shape[1], len(starts)]
        for s in starts:
            win_rows.append({
                "recording": r.recording, "patient_id": r.patient_id, "location": r.location,
                "start_s": s, "mel_start": offset + int(round(s * AUDIO.sr / MEL.hop_length)),
                "murmur": r.murmur, "rec_label": r.rec_label, "fold": r.fold,
            })
        mels.append(mel)
        X_all.append(X)
        offset += mel.shape[1]

    mel = np.concatenate(mels, axis=1).astype(np.float16)
    X_mlp = np.concatenate(X_all).astype(np.float32)
    win = pd.DataFrame(win_rows)
    rec = rec.drop(columns=["path"])

    # --- integrity checks (as in the UrbanSound8K pipeline)
    fpw = frames_per_window()
    assert len(win) == len(X_mlp), "windows / MLP rows mismatch"
    assert np.isfinite(X_mlp).all(), "NaN/inf in X_mlp"
    assert np.isfinite(mel.astype(np.float32)).all(), "NaN/inf in mel"
    assert rec.fold.notna().all(), "recording without fold (patient missing from split.csv)"
    assert win.groupby("patient_id").fold.nunique().max() == 1, "patient split across folds"
    end = rec.set_index("recording")[["mel_offset", "mel_frames"]].sum(axis=1)
    assert (win.mel_start <= win.recording.map(end)).all()
    print("integrity checks passed")

    # --- save
    np.save(out_dir / "X_mlp.npy", X_mlp)
    np.save(out_dir / "mel.npy", mel)
    win.to_csv(out_dir / "windows.csv", index=False)
    rec.to_csv(out_dir / "recordings.csv", index=False)
    cfg = config_dict() | {"mlp_features": mlp_feature_names(), "frames_per_window": fpw}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    print("\n=== SUMMARY ===")
    print(f"MLP input  X_mlp: {X_mlp.shape}  (one vector per window)")
    print(f"CNN input  mel  : {mel.shape} -> windows of ({MEL.n_mels}, {fpw})  [{mel.nbytes / 1e6:.0f} MB]")
    print(f"windows: {len(win)}  |  per class (rec_label):\n{win.rec_label.value_counts().to_string()}")
    print(f"windows per fold:\n{win.groupby('fold').size().to_string()}")
    print(f"saved to {out_dir}")
    return rec, win, X_mlp, mel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    prepare_data(jobs=ap.parse_args().jobs)
