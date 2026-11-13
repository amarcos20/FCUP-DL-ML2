"""CNN of the PhysioNet Challenge 2022 winner (Lu et al., team HearHeart, literature/02_*), re-implemented
from the official source code (physionet.org/files/challenge-2022/1.0.0/sources/HearHeart_Murmur.zip).

Network (base_model.AudioClassifier): 4 conv 3x3 blocks with stride 1 (32-32-32-64 maps), BatchNorm, ReLU,
two max-pooling layers, dropout 0.1, global average pooling; a second "wide" branch (linear 20) whose
output is concatenated before the final linear layer. ~38 k parameters, a third of our CNN.

Two modes, to separate the two effects (both evaluated with our folds and src.evaluate):

  --mode arch    (B: effect of the architecture)
      their network on OUR input and training: log-mel 40 x 401 windows of 4 s, per-band standardisation,
      our augmentation, class weights, Adam 1e-3 / wd 1e-4, early stopping on validation loss
      (src.models.cnn.cross_validate_cnn with model_fn). No wide branch (our windows have no hand features).

  --mode recipe  (A: their whole recipe)
      * input: random 15 s crop of the raw recording (zero-padded if shorter), 128 mel bands 25-2000 Hz,
        window 50 ms / hop 25 ms, power in dB clipped at 80 dB below the maximum (torchaudio defaults),
        no further normalisation;
      * augmentation: white noise at 15 dB SNR on the raw signal (p = 0.5), SpecAugment with 2 frequency
        and 2 time masks of up to 10 % (p = 0.5);
      * wide branch: 6 hand-crafted features of the signal (zero-crossing rate, spectral centroid and
        bandwidth, mean and std). Their age / sex / pregnancy one-hot inputs are LEFT OUT: our models only
        use the audio;
      * training set: one sample per recording; recordings of Present patients where the murmur was not
        heard are excluded (as in their get_corr_recordings);
      * training: cross-entropy with weights 5/3/1 (the metric weights), Adam 1e-3 / wd 1e-3, batch 24,
        lr / 10 at epochs 30, 50, 80, up to 100 epochs, early stopping (patience 20) on the
        recording-level weighted accuracy of the validation fold;
      * prediction: each recording is cut into 15 s patches with 7.5 s hop and the patch probabilities
        are averaged.
      The mel spectrogram is computed on the GPU in torch (same filters as torchaudio: HTK mel scale, no
      filter normalisation, periodic Hann window, centred reflect-padded STFT).

Patient-level evaluation is ours (max over recordings + two tuned thresholds, nested CV), so the numbers
are comparable with the other models; the recording probabilities are copied to every window of the
recording so that oof.csv has the same format as src.models.cnn / src.models.rnn. Their own patient rule (Present if
any recording is Present, Absent if all are Absent, else Unknown) is reported as an extra metric.

Usage (needs data/processed/waves_int16.npy from src.models.prepare_hearheart for --mode recipe):
    python -m src.models.hearheart --mode arch     # -> results/experiments/hearheart_arch
    python -m src.models.hearheart --mode recipe   # -> results/experiments/hearheart
"""
import argparse
import copy
import math
import random

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader, Dataset

from src.models.cnn import DEFAULT_CNN_CFG, cross_validate_cnn, device
from src.models.cnn import save_runs as save_cnn_runs
from src.data import CLASSES
from src.datasets import CNNWindows, PROC_DIR, encode, load_processed
from src.evaluate import RESULTS_DIR, result_dir, cv_report, patient_scores, save_oof, summary, tune_thresholds
from src.metrics import weighted_accuracy

SR = 4000
N_HAND = 6

RECIPE_CFG = dict(
    mode="recipe", wide=True, noise=True, spec_augment=True,
    seg_s=15.0, patch_hop_s=7.5, n_mels=128, fmin=25, fmax=2000, n_fft=512, win_ms=50, hop_ms=25, top_db=80,
    snr_db=15, p_noise=0.5, p_specaug=0.5, max_mask_pct=0.1, n_masks=2,
    class_weights=[5.0, 3.0, 1.0], lr=1e-3, weight_decay=1e-3, batch_size=24,
    max_epochs=100, patience=20, milestones=[30, 50, 80],
    num_workers=4, seed=42,
)
ARCH_CFG = DEFAULT_CNN_CFG | dict(mode="arch", max_epochs=50, patience=8)  # as the CNN run in results/cnn


# -----------------------------------------------------------------------------
# Network (base_model.AudioClassifier)
# -----------------------------------------------------------------------------

def _conv(c_in, c_out, padding):
    conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=padding)
    init.kaiming_normal_(conv.weight, a=0.1)
    conv.bias.data.zero_()
    return conv


class HearHeartCNN(nn.Module):
    """[B, 1, F, T] -> conv(32, pad 2) BN ReLU maxpool -> conv(32) BN ReLU maxpool dropout 0.1
    -> conv(32) BN ReLU dropout 0.1 -> conv(64) BN ReLU -> global average pooling (64)
    [-> concat with linear(n_wide -> 20)] -> linear -> 3 logits."""

    def __init__(self, n_wide=N_HAND, n_out=len(CLASSES)):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(1, 32, 2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            _conv(32, 32, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.1),
            _conv(32, 32, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout(0.1),
            _conv(32, 64, 1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.wide = nn.Linear(n_wide, 20) if n_wide else None
        self.lin = nn.Linear(64 + (20 if n_wide else 0), n_out)

    def forward(self, x, wide=None):
        x = torch.flatten(self.ap(self.conv(x)), 1)
        if self.wide is not None:
            x = torch.cat([x, self.wide(wide)], dim=1)
        return self.lin(x)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# -----------------------------------------------------------------------------
# Recipe front end: log-mel on the GPU, SpecAugment, hand features
# -----------------------------------------------------------------------------

class LogMel(nn.Module):
    """torchaudio MelSpectrogram(power 2) + AmplitudeToDB(top_db) for a batch of waveforms [B, N]."""

    def __init__(self, cfg):
        super().__init__()
        self.n_fft, self.top_db = cfg["n_fft"], cfg["top_db"]
        self.win, self.hop = int(cfg["win_ms"] * SR / 1000), int(cfg["hop_ms"] * SR / 1000)
        fb = librosa.filters.mel(sr=SR, n_fft=self.n_fft, n_mels=cfg["n_mels"], fmin=cfg["fmin"], fmax=cfg["fmax"],
                                 htk=True, norm=None)
        self.register_buffer("fb", torch.tensor(fb, dtype=torch.float32))
        self.register_buffer("window", torch.hann_window(self.win))

    def forward(self, w):
        spec = torch.stft(w, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win, window=self.window,
                          center=True, pad_mode="reflect", return_complex=True).abs() ** 2
        db = 10 * torch.log10(torch.matmul(self.fb, spec).clamp_min(1e-10))
        db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - self.top_db)
        return db[:, None]                                    # [B, 1, n_mels, frames]


def spec_augment_hh(x, cfg):
    """Their spectro_augment: per sample with prob p, n frequency masks filled with the spectrogram mean
    and n time masks filled with 0, each up to max_mask_pct of the axis (torchaudio masking)."""
    x = x.clone()
    _, _, f, t = x.shape
    for b in range(x.shape[0]):
        if random.random() >= cfg["p_specaug"]:
            continue
        mean = x[b].mean()
        for size, axis, fill in [(f, 2, mean), (t, 3, 0.0)]:
            for _ in range(cfg["n_masks"]):
                v = random.random() * cfg["max_mask_pct"] * size
                start, width = int(random.random() * (size - v)), int(v)
                if axis == 2:
                    x[b, :, start:start + width, :] = fill
                else:
                    x[b, :, :, start:start + width] = fill
    return x


def hand_features(sig):
    """Their hand_fea: zero-crossing rate (1 s frames) and spectral centroid / bandwidth (50 ms frames),
    mean and std; centroid and bandwidth divided by 2000 Hz."""
    zcr = librosa.feature.zero_crossing_rate(sig, frame_length=4000, hop_length=2000)[0]
    cen = librosa.feature.spectral_centroid(y=sig, sr=SR, n_fft=200, hop_length=100)[0] / 2000
    bw = librosa.feature.spectral_bandwidth(y=sig, sr=SR, n_fft=200, hop_length=100)[0] / 2000
    return np.nan_to_num(np.array([zcr.mean(), zcr.std(), cen.mean(), cen.std(), bw.mean(), bw.std()],
                                  dtype=np.float32))


# -----------------------------------------------------------------------------
# Recipe data
# -----------------------------------------------------------------------------

def load_waves(proc_dir=PROC_DIR, mmap=True):
    """(int16 waveform array, DataFrame recording -> wave_offset, n_samples)."""
    w = np.load(proc_dir / "waves_int16.npy", mmap_mode="r" if mmap else None)
    return w, pd.read_csv(proc_dir / "waves.csv").set_index("recording")


def get_wave(waves, widx, recording):
    o, n = widx.loc[recording, ["wave_offset", "n_samples"]]
    return np.asarray(waves[o:o + n], dtype=np.float32) / 32768.0


def training_recordings(recordings):
    """Recordings used for training (theirs): all recordings of Unknown/Absent patients, and of Present
    patients only those at a murmur location."""
    return recordings[~((recordings.murmur == "Present") & (recordings.rec_label != "Present"))]


class Crops(Dataset):
    """One random 15 s crop per recording per epoch (+ noise), with its hand features."""

    def __init__(self, waves, widx, recs, cfg):
        self.waves, self.widx, self.cfg = waves, widx, cfg
        self.recs = recs.recording.to_numpy()
        self.y = encode(recs.rec_label)
        self.seg = int(cfg["seg_s"] * SR)

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        c, x = self.cfg, get_wave(self.waves, self.widx, self.recs[i])
        orig, n, L = x, len(x), self.seg
        if c["noise"] and np.random.rand() < c["p_noise"]:
            p_noise = np.mean(x ** 2) / 10 ** (c["snr_db"] / 10)
            x = x + np.random.randn(n).astype(np.float32) * np.sqrt(p_noise)
        if n > L:
            s = np.random.randint(0, n - L + 1)
            x = x[s:s + L]
        elif n < L:
            b = np.random.randint(0, L - n + 1)
            x = np.pad(x, (b, L - n - b))
        fea = hand_features(orig if n < L else x)           # short recordings: features of the whole signal
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(fea), int(self.y[i])


def _seed_worker(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_patches(waves, widx, recordings, cfg):
    """Deterministic 15 s patches (hop 7.5 s) of every recording, zero-padded at the end; short recordings
    give one centred patch. Returns waves [P, L], hand features [P, 6] and the recording of each patch."""
    L, H = int(cfg["seg_s"] * SR), int(cfg["patch_hop_s"] * SR)
    X, F, owner = [], [], []
    for r in recordings:
        x = get_wave(waves, widx, r)
        n = len(x)
        if n <= L:
            b = (L - n) // 2
            segs = [(np.pad(x, (b, L - n - b)), x)]
        else:
            segs = []
            for i in range(math.ceil((n - L) / H) + 1):
                seg = x[i * H:i * H + L]
                segs.append((np.pad(seg, (0, L - len(seg))), seg))
        for padded, raw in segs:
            X.append(padded)
            F.append(hand_features(raw))
            owner.append(r)
    return np.stack(X).astype(np.float32), np.stack(F), np.array(owner)


def predict_patches(model, frontend, X, F, cfg, batch=64):
    dev = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            spec = frontend(torch.from_numpy(X[i:i + batch]).to(dev))
            wide = torch.from_numpy(F[i:i + batch]).to(dev) if cfg["wide"] else None
            out.append(torch.softmax(model(spec, wide), 1).cpu().numpy())
    return np.concatenate(out)


def recording_probs(patch_probs, owner):
    """Mean patch probability per recording (DataFrame indexed by recording, columns CLASSES)."""
    df = pd.DataFrame(patch_probs, columns=CLASSES)
    df["recording"] = owner
    return df.groupby("recording", sort=False)[CLASSES].mean()


# -----------------------------------------------------------------------------
# Recipe training
# -----------------------------------------------------------------------------

def train_recipe(ds_tr, val, cfg, fold_idx=0):
    """val = (X, F, owner, labels of the early-stopping recordings: Series recording -> class)."""
    torch.manual_seed(cfg["seed"] + fold_idx)
    np.random.seed(cfg["seed"] + fold_idx)
    random.seed(cfg["seed"] + fold_idx)
    dev = device()
    model = HearHeartCNN(N_HAND if cfg["wide"] else 0).to(dev)
    frontend = LogMel(cfg).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cfg["class_weights"], device=dev))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, cfg["milestones"], gamma=0.1)
    loader = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
                        worker_init_fn=_seed_worker, persistent_workers=cfg["num_workers"] > 0)
    Xv, Fv, owner, es_labels = val
    es_mask = np.isin(owner, es_labels.index)
    y_patch = torch.from_numpy(encode(es_labels.loc[owner[es_mask]])).to(dev)

    best_wa, best_weights, wait, history = -1.0, None, 0, []
    print(f"\n>>> [Fold {fold_idx} | recipe] {len(ds_tr)} train recordings, {len(es_labels)} val recordings "
          f"({len(Xv)} patches), {n_params(model) / 1e3:.0f} k parameters <<<", flush=True)
    for epoch in range(cfg["max_epochs"]):
        model.train()
        tot, seen = 0.0, 0
        for x, f, y in loader:
            x, f, y = x.to(dev), f.to(dev), y.to(dev)
            spec = frontend(x)
            if cfg["spec_augment"]:
                spec = spec_augment_hh(spec, cfg)
            opt.zero_grad()
            loss = loss_fn(model(spec, f if cfg["wide"] else None), y)
            loss.backward()
            opt.step()
            tot, seen = tot + loss.item() * len(y), seen + len(y)

        pp = predict_patches(model, frontend, Xv, Fv, cfg)
        with torch.no_grad():
            val_loss = loss_fn(torch.log(torch.from_numpy(pp[es_mask]).to(dev).clamp_min(1e-8)), y_patch).item()
        rp = recording_probs(pp, owner).loc[es_labels.index]
        pred = np.array(CLASSES)[rp.to_numpy().argmax(1)]
        val_wa = weighted_accuracy(es_labels.to_numpy(), pred)
        val_acc = float((pred == es_labels.to_numpy()).mean())

        is_best = val_wa > best_wa
        print(f"  Fold {fold_idx} recipe | epoch {epoch + 1:03d}/{cfg['max_epochs']} | lr {sched.get_last_lr()[0]:.0e} "
              f"| train loss {tot / seen:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f} "
              f"| val WA (recordings) {val_wa:.4f}{' *' if is_best else ''}", flush=True)
        history.append({"epoch": epoch, "train_loss": tot / seen, "val_loss": val_loss, "val_acc": val_acc,
                        "val_wa_rec": val_wa})
        if is_best:
            best_wa, best_weights, wait = val_wa, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"  Fold {fold_idx} recipe | early stopping at epoch {epoch + 1}", flush=True)
                break
        sched.step()

    model.load_state_dict(best_weights)
    hist = pd.DataFrame(history)
    hist.attrs["best_epoch"] = int(hist.val_wa_rec.idxmax())
    return model, frontend, hist


def cross_validate_recipe(d, cfg=RECIPE_CFG, folds=range(5), mmap=True):
    """One model per CV fold (test patients, fold -1, never used). Returns the validation windows of
    `folds`, their out-of-fold probabilities (recording probability copied to its windows), the
    recording-level probabilities and the runs."""
    waves, widx = load_waves(mmap=mmap)
    rec = d.recordings
    rec_oof, runs = [], []
    for k in folds:
        tr = training_recordings(rec[(rec.fold >= 0) & (rec.fold != k)])
        va = rec[rec.fold == k]
        es = training_recordings(va).set_index("recording").rec_label   # early-stopping recordings (theirs)
        X, F, owner = make_patches(waves, widx, va.recording, cfg)
        model, frontend, hist = train_recipe(Crops(waves, widx, tr, cfg), (X, F, owner, es), cfg, fold_idx=k)
        rp = recording_probs(predict_patches(model, frontend, X, F, cfg), owner)
        rec_oof.append(rp)
        runs.append({"fold": k, "model": model, "history": hist})
        b = hist.attrs["best_epoch"]
        print(f"==> Fold {k}: best epoch {b + 1} | val WA (recordings) {hist.val_wa_rec[b]:.4f}", flush=True)
    rec_oof = pd.concat(rec_oof)
    dev_mask = d.windows["fold"].isin(folds).to_numpy()
    dev = d.windows[dev_mask]
    return dev, rec_oof.loc[dev.recording].to_numpy(dtype=np.float32), rec_oof, runs


def hearheart_rule(rec_oof, recordings):
    """Their patient rule on recording argmax: Absent if all recordings are Absent, Present if any is
    Present, otherwise Unknown. Returns per-fold and pooled weighted accuracy."""
    r = recordings.set_index("recording").loc[rec_oof.index]
    pred_rec = pd.Series(np.array(CLASSES)[rec_oof[CLASSES].to_numpy().argmax(1)], index=rec_oof.index)
    g = pd.DataFrame({"pid": r.patient_id, "fold": r.fold, "true": r.murmur, "pred": pred_rec})

    def rule(p):
        return "Absent" if (p == "Absent").all() else "Present" if (p == "Present").any() else "Unknown"

    pat = g.groupby("pid").agg(fold=("fold", "first"), true=("true", "first"), pred=("pred", rule))
    per_fold = {int(k): weighted_accuracy(x.true, x.pred) for k, x in pat.groupby("fold")}
    return per_fold, weighted_accuracy(pat.true, pat.pred)


# -----------------------------------------------------------------------------
# Save / load / ensemble prediction
# -----------------------------------------------------------------------------

def save_recipe_runs(runs, cfg, out_dir):
    for r in runs:
        torch.save({"cfg": cfg, "state_dict": r["model"].state_dict()}, out_dir / f"fold{r['fold']}.pt")
        r["history"].to_csv(out_dir / f"history_fold{r['fold']}.csv", index=False)


def load_runs(out_dir=result_dir("hearheart")):
    """Returns (runs, cfg); works for both modes."""
    runs, cfg = [], None
    for p in sorted(out_dir.glob("fold*.pt")):
        ck = torch.load(p, map_location=device(), weights_only=False)
        cfg = ck["cfg"]
        model = HearHeartCNN(N_HAND if cfg["mode"] == "recipe" and cfg["wide"] else 0)
        model.load_state_dict(ck["state_dict"])
        runs.append({"fold": int(p.stem[4:]), "model": model.to(device()).eval(), "band_stats": ck.get("band_stats")})
    return runs, cfg


def predict_ensemble_hearheart(runs, d, mask, cfg):
    """Window probabilities for d.windows[mask], averaged over the fold models."""
    win = d.windows[mask]
    if cfg["mode"] == "arch":
        from src.models.cnn import predict_cnn_proba
        return np.mean([predict_cnn_proba(r["model"], CNNWindows(d, mask, r["band_stats"]), cfg | {"num_workers": 0})
                        for r in runs], axis=0)
    waves, widx = load_waves()
    X, F, owner = make_patches(waves, widx, win.recording.unique(), cfg)
    frontend = LogMel(cfg).to(device())
    rp = np.mean([recording_probs(predict_patches(r["model"], frontend, X, F, cfg), owner).to_numpy()
                  for r in runs], axis=0)
    rp = pd.DataFrame(rp, index=pd.unique(owner), columns=CLASSES)
    return rp.loc[win.recording].to_numpy(dtype=np.float32)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["recipe", "arch"], default="recipe")
    ap.add_argument("--epochs", type=int, help="default 100 (recipe) / 50 (arch)")
    ap.add_argument("--patience", type=int, help="default 20 (recipe) / 8 (arch)")
    ap.add_argument("--batch-size", type=int, help="default 24 (recipe) / 128 (arch)")
    ap.add_argument("--no-wide", action="store_true", help="recipe without the hand-feature branch")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--no-spec-augment", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--in-memory", action="store_true", help="load mel.npy / waves into RAM instead of mmap")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--name", help="output folder under results/ (default hearheart / hearheart_arch)")
    a = ap.parse_args()

    cfg = dict(RECIPE_CFG if a.mode == "recipe" else ARCH_CFG)
    for key, val in [("max_epochs", a.epochs), ("patience", a.patience), ("batch_size", a.batch_size)]:
        if val is not None:
            cfg[key] = val
    cfg["num_workers"] = a.workers
    if a.mode == "recipe":
        cfg |= dict(wide=not a.no_wide, noise=not a.no_noise, spec_augment=not a.no_spec_augment)
    else:
        cfg["spec_augment"] = not a.no_spec_augment
    name = a.name or ("hearheart" if a.mode == "recipe" else "hearheart_arch")
    print(cfg, "| device:", device(), flush=True)

    d = load_processed(mmap=not a.in_memory)
    extra = {}
    if a.mode == "arch":
        dev, oof, runs = cross_validate_cnn(d, cfg, a.folds, model_fn=lambda c: HearHeartCNN(n_wide=0))
    else:
        dev, oof, rec_oof, runs = cross_validate_recipe(d, cfg, a.folds, mmap=not a.in_memory)
        per_fold, pooled = hearheart_rule(rec_oof, d.recordings)
        extra = {"hearheart_rule": {"wa_per_fold": per_fold, "wa_mean": float(np.mean(list(per_fold.values()))),
                                    "wa_pooled": pooled}}
        print(f"their patient rule: WA per fold {np.round(list(per_fold.values()), 3)} | pooled {pooled:.3f}")

    pat = patient_scores(dev, oof)
    table, _, rep = cv_report(pat)
    t_final, _ = tune_thresholds(pat)
    print(table.round(3).to_string())
    print(f"WA (tuned, nested): {table['WA (tuned)'].mean():.3f} ± {table['WA (tuned)'].std():.3f}")

    out = save_oof(name, dev.index, oof, summary(table, rep) | {"cfg": cfg, "thresholds": t_final} | extra)
    if a.mode == "arch":
        save_cnn_runs(runs, cfg, out)
    else:
        save_recipe_runs(runs, cfg, out)
        rec_oof.to_csv(out / "rec_oof.csv", float_format="%.5f")
    print("saved to", out)


if __name__ == "__main__":
    main()
