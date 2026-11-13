"""Raw waveforms for the HearHeart recipe (src.models.hearheart --mode recipe).

The winner of the PhysioNet Challenge 2022 (Lu et al., team HearHeart) takes a random 15 s crop of
each recording, adds noise to the raw signal and only then computes the mel spectrogram (128 bands,
25-2000 Hz). Our mel.npy (40 bands, <= 1 kHz, band-pass filtered and z-scored audio) cannot
reproduce that, so this script stores the original audio instead: every .wav (16-bit PCM, 4 kHz)
concatenated losslessly as int16, in the order of data/processed/recordings.csv.

Usage (after src.prepare_data):
    python -m src.models.prepare_hearheart

Outputs (data/processed/):
    waves_int16.npy   (total_samples,) int16 -- divide by 32768 to get the [-1, 1] float signal
    waves.csv         recording, wave_offset, n_samples (same row order as recordings.csv)
"""
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.data import WAV_DIR
from src.datasets import PROC_DIR

SR = 4000


def prepare_hearheart(proc_dir=PROC_DIR):
    rec = pd.read_csv(proc_dir / "recordings.csv")
    waves, rows, offset = [], [], 0
    for r in tqdm(rec.recording, desc="wav"):
        x, sr = sf.read(WAV_DIR / f"{r}.wav", dtype="int16")
        assert sr == SR and x.ndim == 1, (r, sr, x.shape)
        waves.append(x)
        rows.append({"recording": r, "wave_offset": offset, "n_samples": len(x)})
        offset += len(x)
    w = np.concatenate(waves)
    idx = pd.DataFrame(rows)
    assert (idx.recording.values == rec.recording.values).all()
    assert np.allclose(idx.n_samples / SR, rec.duration, atol=1e-3), "durations differ from recordings.csv"
    np.save(proc_dir / "waves_int16.npy", w)
    idx.to_csv(proc_dir / "waves.csv", index=False)
    print(f"{len(idx)} recordings, {len(w) / 1e6:.1f} M samples ({w.nbytes / 1e6:.0f} MB) -> {proc_dir}")


if __name__ == "__main__":
    prepare_hearheart()
