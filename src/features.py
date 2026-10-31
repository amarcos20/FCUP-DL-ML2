"""Audio pre-processing and feature extraction for the MLP and the CNN (librosa based).

Pipeline for every recording (same for train / validation / test / hidden data):

    load (native 4 kHz, mono) -> band-pass 25-1000 Hz -> amplitude normalisation (z-score)
      -> split into fixed-length windows (WINDOW_S seconds, hop HOP_S)
      -> MLP: one vector of summary statistics per window          (src.features.mlp_features)
      -> CNN: one log-mel spectrogram (N_MELS x frames) per window  (src.features.logmel)

Design choices (see notebooks/01_eda.ipynb for the evidence behind them):
- 4 kHz is the native sample rate of every file and >99% of the energy is below 1.5 kHz,
  so we do not resample (unlike UrbanSound8K at 22.05 kHz).
- Recordings last 5-65 s, so instead of a single crop/pad to a fixed length (as in the
  UrbanSound8K project, where clips were <= 4 s) we cut each recording into 4 s windows.
  4 s contains ~7 heart cycles at the median heart rate (103 bpm) and >= 3 at the lowest (46 bpm). Every window
  inherits the recording label; window predictions are aggregated back to recording and
  patient level at evaluation time.
- The .tsv segmentations are NOT used as model input, because hidden-set recordings may
  not come with them. They are only used in the EDA.
- Dataset-level normalisation (feature standardisation for the MLP, per-mel-band
  mean/std for the CNN) is fitted on the training folds only -> see `Standardizer`.
"""
from dataclasses import asdict, dataclass

import librosa
import numpy as np
import scipy.signal as ss


@dataclass(frozen=True)
class AudioConfig:
    sr: int = 4000            # native sample rate of CirCor
    f_low: float = 25.0       # band-pass lower cut-off (Hz): removes DC / body-movement drift
    f_high: float = 1000.0    # band-pass upper cut-off (Hz): heart sounds + murmurs live below
    window_s: float = 4.0     # window length (s)
    hop_s: float = 2.0        # hop between windows (s) -> 50 % overlap
    min_last_s: float = 2.0   # keep a final partial window only if it has >= this many seconds (zero-padded)

    @property
    def window_len(self):
        return int(self.sr * self.window_s)

    @property
    def hop_len(self):
        return int(self.sr * self.hop_s)


@dataclass(frozen=True)
class MelConfig:
    n_fft: int = 256          # 64 ms analysis window at 4 kHz
    hop_length: int = 40      # 10 ms -> 100 frames per second, 401 frames per 4 s window
    n_mels: int = 40
    fmin: float = 20.0
    fmax: float = 1000.0


@dataclass(frozen=True)
class MFCCConfig:
    n_mfcc: int = 20
    n_fft: int = 256
    hop_length: int = 40
    n_mels: int = 40
    fmin: float = 20.0
    fmax: float = 1000.0


AUDIO = AudioConfig()
MEL = MelConfig()
MFCC = MFCCConfig()

# Frequency bands (Hz) used for the band-energy features of the MLP
BANDS = [(25, 100), (100, 200), (200, 400), (400, 700), (700, 1000)]


def config_dict():
    return {"audio": asdict(AUDIO), "mel": asdict(MEL), "mfcc": asdict(MFCC), "bands": BANDS}


# ---------------------------------------------------------------- signal level

_SOS = None


def _bandpass_sos(cfg=AUDIO):
    global _SOS
    if _SOS is None:
        _SOS = ss.butter(4, [cfg.f_low, cfg.f_high], btype="bandpass", fs=cfg.sr, output="sos")
    return _SOS


def load_audio(path, cfg=AUDIO):
    """Load at the native rate, band-pass filter (zero-phase) and z-score normalise."""
    y, _ = librosa.load(path, sr=cfg.sr, mono=True)
    y = ss.sosfiltfilt(_bandpass_sos(cfg), y)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return y.astype(np.float32)


def window_starts(n_samples, cfg=AUDIO):
    """Start indices (samples) of the fixed-length windows covering a signal."""
    L, H = cfg.window_len, cfg.hop_len
    if n_samples <= L:
        return [0]
    starts = list(range(0, n_samples - L + 1, H))
    last_end = starts[-1] + L
    if n_samples - last_end >= int(cfg.min_last_s * cfg.sr):
        starts.append(starts[-1] + H)  # partial window, zero-padded in split_windows
    return starts


def split_windows(y, cfg=AUDIO):
    """(n_windows, window_len) array of windows; the last one is zero-padded if needed."""
    L = cfg.window_len
    out = []
    for s in window_starts(len(y), cfg):
        w = y[s:s + L]
        if len(w) < L:
            w = np.pad(w, (0, L - len(w)))
        out.append(w)
    return np.stack(out), np.array(window_starts(len(y), cfg)) / cfg.sr


# ---------------------------------------------------------------- CNN input

def logmel(y, cfg=MEL, sr=AUDIO.sr):
    """Log-mel spectrogram in dB (n_mels x frames). No per-sample min-max scaling:
    absolute levels carry information (e.g. noisy 'Unknown' recordings), and dataset
    standardisation is done later with training statistics."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax, power=2.0,
    )
    return librosa.power_to_db(S, ref=1.0, top_db=None).astype(np.float32)


def frames_per_window(cfg=MEL, acfg=AUDIO):
    return 1 + acfg.window_len // cfg.hop_length


# ---------------------------------------------------------------- MLP input

def mlp_feature_names():
    names = []
    names += [f"mfcc{i}_mean" for i in range(MFCC.n_mfcc)]
    names += [f"mfcc{i}_std" for i in range(MFCC.n_mfcc)]
    names += [f"dmfcc{i}_std" for i in range(MFCC.n_mfcc)]
    names += [f"ddmfcc{i}_std" for i in range(MFCC.n_mfcc)]
    for f in ["centroid", "bandwidth", "rolloff", "flatness", "zcr", "rms"]:
        names += [f"{f}_mean", f"{f}_std"]
    names += [f"band_{lo}_{hi}" for lo, hi in BANDS]
    names += ["env_autocorr_peak", "env_autocorr_lag"]
    return names


def mlp_features(w, sr=AUDIO.sr):
    """Fixed-size summary vector of one window (see mlp_feature_names for the layout).

    - MFCC mean/std + std of deltas and delta-deltas (as in the UrbanSound8K MLP, but with
      std of the deltas: their mean over time is ~0 and carries no information).
    - Spectral shape statistics (centroid, bandwidth, roll-off, flatness, ZCR, RMS):
      murmurs add broadband high-frequency energy between S1 and S2.
    - Relative energy in 5 frequency bands (log ratio).
    - Periodicity of the energy envelope (autocorrelation peak in the 0.3-1.5 s lag range,
      i.e. 40-200 bpm): clean heart sounds are strongly periodic, noisy / 'Unknown'
      recordings are not.
    """
    c = MFCC
    m = librosa.feature.mfcc(y=w, sr=sr, n_mfcc=c.n_mfcc, n_fft=c.n_fft, hop_length=c.hop_length,
                             n_mels=c.n_mels, fmin=c.fmin, fmax=c.fmax)
    d1 = librosa.feature.delta(m, order=1)
    d2 = librosa.feature.delta(m, order=2)
    feats = [m.mean(1), m.std(1), d1.std(1), d2.std(1)]

    S = np.abs(librosa.stft(w, n_fft=c.n_fft, hop_length=c.hop_length))
    spec = [
        librosa.feature.spectral_centroid(S=S, sr=sr),
        librosa.feature.spectral_bandwidth(S=S, sr=sr),
        librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85),
        librosa.feature.spectral_flatness(S=S),
        librosa.feature.zero_crossing_rate(w, frame_length=c.n_fft, hop_length=c.hop_length),
        librosa.feature.rms(S=S, frame_length=c.n_fft),
    ]
    for s in spec:
        feats.append(np.array([s.mean(), s.std()]))

    freqs = librosa.fft_frequencies(sr=sr, n_fft=c.n_fft)
    P = (S ** 2).sum(1)
    tot = P[(freqs >= BANDS[0][0]) & (freqs < BANDS[-1][1])].sum() + 1e-10
    feats.append(np.array([np.log(P[(freqs >= lo) & (freqs < hi)].sum() / tot + 1e-10) for lo, hi in BANDS]))

    feats.append(np.array(envelope_periodicity(w, sr)))
    return np.concatenate(feats).astype(np.float32)


def envelope_periodicity(w, sr=AUDIO.sr, fs_env=50):
    """Peak of the normalised autocorrelation of the (downsampled) energy envelope within
    lags 0.3-1.5 s, and the lag (s) at which it occurs (~ one heart cycle)."""
    env = np.abs(ss.hilbert(w))
    env = ss.resample_poly(env, fs_env, sr)
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    ac = ac / (ac[0] + 1e-10)
    lo, hi = int(0.3 * fs_env), min(int(1.5 * fs_env), len(ac) - 1)
    k = lo + int(np.argmax(ac[lo:hi]))
    return float(ac[k]), k / fs_env


# ---------------------------------------------------------------- normalisation

class Standardizer:
    """z-score with statistics fitted on training data only (avoids leakage).

    axis=0 for MLP feature matrices (n, d); for CNN spectrograms (n, n_mels, frames)
    use axis=(0, 2) -> one mean/std per mel band.
    """

    def __init__(self, axis=0):
        self.axis = axis

    def fit(self, X):
        self.mean_ = X.mean(axis=self.axis, keepdims=True).astype(np.float32)
        self.std_ = X.std(axis=self.axis, keepdims=True).astype(np.float32) + 1e-6
        return self

    def transform(self, X):
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
