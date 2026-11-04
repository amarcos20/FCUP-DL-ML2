"""2D CNN on the log-mel windows of src.prepare_data (40 mel bands x 401 frames = 4 s at 10 ms).

Inputs come from src.datasets: windows are sliced on the fly from mel.npy (CNNWindows),
standardised per mel band with statistics of the training folds only (mel_band_stats), and
randomly shifted in time during training (random_crop). No per-window normalisation, so the
absolute level in dB (informative for noisy 'Unknown' recordings, 01_eda §8) is kept.
Targets are the recording-level labels (`rec_label`); one prediction per patient via src.evaluate.

Usage (trains the 5 CV folds, saves results/cnn/{oof.csv, cv_summary.json, fold*.pt}):
    python -m src.cnn [--epochs 30 --patience 5 --batch-size 128 --workers 4 ...]
"""
import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import CLASSES
from src.datasets import CNNWindows, class_weights, fold_masks, load_processed, mel_band_stats
from src.evaluate import RESULTS_DIR, cv_report, patient_scores, save_oof, summary, tune_thresholds

DEFAULT_CNN_CFG = dict(
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=128,
    max_epochs=30,
    patience=5,
    weight_power=1.0,
    dropout=0.3,
    spec_augment=True,
    random_crop=True,
    label_col="rec_label",
    num_workers=4,
    seed=42,
)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# SpecAugment (inputs are standardised per mel band, so 0 = band mean)
# -----------------------------------------------------------------------------

def apply_spec_augment(x, max_freq_mask=8, max_time_mask=40):
    """One frequency mask (<= 8 of 40 bands) and one time mask (<= 0.4 s) per batch, [B, C, F, T]."""
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
# 2D CNN (strided ConvNet + BatchNorm + global pooling)
# -----------------------------------------------------------------------------

class HeartMurmurCNN(nn.Module):
    """Four strided conv blocks (16 -> 32 -> 64 -> 128 maps), global average pooling, MLP head."""
    def __init__(self, n_classes=len(CLASSES), dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: [B, 1, 40, 401] -> [B, 16, 20, 201]
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Block 2: -> [B, 32, 10, 101]
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 3: -> [B, 64, 5, 51]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 4: -> [B, 128, 3, 26]
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
# Training, validation and cross-validation
# -----------------------------------------------------------------------------

def loader(ds, cfg, shuffle):
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, drop_last=shuffle,
                      num_workers=cfg["num_workers"], persistent_workers=cfg["num_workers"] > 0)


def predict_cnn_proba(model, ds, cfg):
    """Softmax probabilities for every window of a CNNWindows dataset (in order)."""
    dev = next(model.parameters()).device
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in loader(ds, cfg, shuffle=False):
            probs.append(torch.softmax(model(x.to(dev)), dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def train_cnn(ds_tr, ds_va, cfg, fold_idx=0):
    """Weighted cross-entropy + Adam, early stopping on validation loss."""
    torch.manual_seed(cfg["seed"] + fold_idx)
    np.random.seed(cfg["seed"] + fold_idx)
    dev = device()

    model = HeartMurmurCNN(dropout=cfg["dropout"]).to(dev)
    w = torch.from_numpy(class_weights(ds_tr.y, power=cfg["weight_power"])).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    train_loader, val_loader = loader(ds_tr, cfg, shuffle=True), loader(ds_va, cfg, shuffle=False)

    best_loss, best_weights, wait, history = float("inf"), None, 0, []
    print(f"\n>>> [Fold {fold_idx}] {len(ds_tr)} train windows, {len(ds_va)} val windows <<<", flush=True)

    for epoch in range(cfg["max_epochs"]):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            if cfg["spec_augment"]:
                x_b = apply_spec_augment(x_b)
            x_b, y_b = x_b.to(dev), y_b.to(dev)
            optimizer.zero_grad()
            loss = loss_fn(model(x_b), y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_b)
        train_loss /= len(train_loader) * cfg["batch_size"]

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(dev), y_v.to(dev)
                logits = model(x_v)
                val_loss += loss_fn(logits, y_v).item() * len(x_v)
                correct += (logits.argmax(dim=1) == y_v).sum().item()
                total += len(y_v)
        val_loss /= total
        val_acc = correct / total

        is_best = val_loss < best_loss
        star = " *" if is_best else ""
        print(f"  Fold {fold_idx} | epoch {epoch+1:02d}/{cfg['max_epochs']} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val acc {val_acc:.4f}{star}", flush=True)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

        if is_best:
            best_loss, best_weights, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"  Fold {fold_idx} | early stopping at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_weights)
    return model, pd.DataFrame(history)


def cross_validate_cnn(d, cfg=DEFAULT_CNN_CFG, folds=range(5)):
    """One model per CV fold (test patients, fold -1, are never used).

    Returns the validation windows of `folds` (rows of d.windows, original index kept),
    their out-of-fold probabilities, and per fold a dict with model, band_stats and history."""
    dev_mask = d.windows["fold"].isin(folds).to_numpy()
    oof = np.full((len(d.windows), len(CLASSES)), np.nan, dtype=np.float32)
    runs = []
    for k in folds:
        tr, va, _ = fold_masks(d.windows, val_fold=k)
        band_stats = mel_band_stats(d, tr)
        ds_tr = CNNWindows(d, tr, band_stats, label_col=cfg["label_col"], random_crop=cfg["random_crop"])
        ds_va = CNNWindows(d, va, band_stats, label_col=cfg["label_col"])
        model, hist = train_cnn(ds_tr, ds_va, cfg, fold_idx=k)
        oof[va] = predict_cnn_proba(model, ds_va, cfg)
        runs.append({"fold": k, "model": model, "band_stats": band_stats, "history": hist})
        best = hist.loc[hist.val_loss.idxmin()]
        print(f"==> Fold {k}: best epoch {int(best.epoch)+1} | val loss {best.val_loss:.4f} | "
              f"val acc {best.val_acc:.4f}\n", flush=True)
    return d.windows[dev_mask], oof[dev_mask], runs


def predict_ensemble_cnn(runs, d, mask, cfg=DEFAULT_CNN_CFG):
    """Average window probabilities of all fold models (each with its own band statistics)."""
    return np.mean([predict_cnn_proba(r["model"].to(device()), CNNWindows(d, mask, r["band_stats"]), cfg)
                    for r in runs], axis=0)


# -----------------------------------------------------------------------------
# Save / load fold models
# -----------------------------------------------------------------------------

def save_runs(runs, cfg, out_dir):
    for r in runs:
        torch.save({"cfg": cfg, "state_dict": r["model"].state_dict(), "band_stats": r["band_stats"]},
                   out_dir / f"fold{r['fold']}.pt")
        r["history"].to_csv(out_dir / f"history_fold{r['fold']}.csv", index=False)


def load_runs(out_dir=RESULTS_DIR / "cnn"):
    runs = []
    for p in sorted(out_dir.glob("fold*.pt")):
        ck = torch.load(p, map_location=device(), weights_only=False)
        model = HeartMurmurCNN(dropout=ck["cfg"]["dropout"]).to(device())
        model.load_state_dict(ck["state_dict"])
        runs.append({"fold": int(p.stem[4:]), "model": model.eval(), "band_stats": ck["band_stats"]})
    return runs


def main():
    ap = argparse.ArgumentParser()
    c = DEFAULT_CNN_CFG
    ap.add_argument("--epochs", type=int, default=c["max_epochs"])
    ap.add_argument("--patience", type=int, default=c["patience"])
    ap.add_argument("--lr", type=float, default=c["lr"])
    ap.add_argument("--batch-size", type=int, default=c["batch_size"])
    ap.add_argument("--dropout", type=float, default=c["dropout"])
    ap.add_argument("--weight-power", type=float, default=c["weight_power"])
    ap.add_argument("--no-spec-augment", action="store_true")
    ap.add_argument("--no-random-crop", action="store_true")
    ap.add_argument("--workers", type=int, default=c["num_workers"])
    ap.add_argument("--in-memory", action="store_true", help="load mel.npy into RAM (~0.6 GB) instead of mmap")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--name", default="cnn", help="output folder under results/")
    a = ap.parse_args()
    cfg = c | dict(max_epochs=a.epochs, patience=a.patience, lr=a.lr, batch_size=a.batch_size, dropout=a.dropout,
                   weight_power=a.weight_power, spec_augment=not a.no_spec_augment,
                   random_crop=not a.no_random_crop, num_workers=a.workers)
    print(cfg, "| device:", device(), flush=True)

    d = load_processed(mmap=not a.in_memory)
    dev, oof, runs = cross_validate_cnn(d, cfg, a.folds)
    pat = patient_scores(dev, oof)
    table, _, rep = cv_report(pat)
    t_final, _ = tune_thresholds(pat)
    print(table.round(3).to_string())
    print(f"WA (tuned, nested): {table['WA (tuned)'].mean():.3f} ± {table['WA (tuned)'].std():.3f}")

    out = save_oof(a.name, dev.index, oof, summary(table, rep) | {"cfg": cfg, "thresholds": t_final})
    save_runs(runs, cfg, out)
    print("saved to", out)


if __name__ == "__main__":
    main()
