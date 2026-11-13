"""MLP on the window-level feature vectors of src.prepare_data (X_mlp.npy, 99 features per 4 s window).

Inputs come from src.datasets (mlp_arrays: z-score fitted on the training folds only); targets
are the recording-level labels (`rec_label`). Window predictions are turned into one
prediction per patient by src.evaluate.

Usage (trains the 5 CV folds, saves results/experiments/mlp/{oof.csv, cv_summary.json, fold*.pt}):
    python -m src.mlp [--epochs 100 --patience 10 --lr 1e-3 ...]
"""
import argparse
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import CLASSES
from src.datasets import class_weights, fold_masks, load_processed, mlp_arrays
from src.evaluate import RESULTS_DIR, result_dir, cv_report, patient_scores, save_oof, summary, tune_thresholds
from src.features import Standardizer

DEFAULT_CFG = dict(
    hidden=(256, 128), dropout=0.3,           # architecture
    lr=1e-3, weight_decay=1e-4,               # Adam
    batch_size=256, max_epochs=100, patience=10,
    weight_power=1.0,                         # class weights = (inverse frequency) ** power
    label_col="rec_label", seed=42,
)


class MLP(nn.Module):
    """Linear -> BatchNorm -> ReLU -> Dropout per hidden layer, linear output (logits)."""

    def __init__(self, n_in, hidden=(256, 128), dropout=0.3, n_out=len(CLASSES)):
        super().__init__()
        layers = []
        for h in hidden:
            layers += [nn.Linear(n_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            n_in = h
        layers.append(nn.Linear(n_in, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_proba(model, X, batch_size=4096):
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = [torch.softmax(model(torch.from_numpy(X[i:i + batch_size]).to(dev)), 1).cpu()
               for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


def train_mlp(X_tr, y_tr, X_va, y_va, cfg):
    """Train with weighted cross-entropy and Adam; early stopping on validation loss.
    Returns the best model and the per-epoch history."""
    torch.manual_seed(cfg["seed"])
    dev = device()
    model = MLP(X_tr.shape[1], cfg["hidden"], cfg["dropout"]).to(dev)
    w = torch.from_numpy(class_weights(y_tr, power=cfg["weight_power"])).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    Xt, yt = torch.from_numpy(X_tr).to(dev), torch.from_numpy(y_tr).to(dev)
    Xv, yv = torch.from_numpy(X_va).to(dev), torch.from_numpy(y_va).to(dev)
    best, best_loss, wait, history = None, np.inf, 0, []
    for epoch in range(cfg["max_epochs"]):
        model.train()
        perm, total = torch.randperm(len(Xt), device=dev), 0.0
        for i in range(0, len(Xt), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            if len(idx) < 2:  # BatchNorm needs > 1 sample
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            logits = model(Xv)
            val_loss = loss_fn(logits, yv).item()
            val_acc = (logits.argmax(1) == yv).float().mean().item()
        history.append({"epoch": epoch, "train_loss": total / len(Xt), "val_loss": val_loss, "val_acc": val_acc})

        if val_loss < best_loss:
            best, best_loss, wait = copy.deepcopy(model.state_dict()), val_loss, 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best)
    return model, pd.DataFrame(history)


def cross_validate(d, cfg=DEFAULT_CFG, folds=range(5)):
    """Train one model per CV fold (test patients, fold -1, are never used).

    Returns the validation windows of `folds` (rows of d.windows, original index kept),
    their out-of-fold probabilities, and per fold a dict with model, scaler and history."""
    dev_mask = d.windows["fold"].isin(folds).to_numpy()
    oof = np.full((len(d.windows), len(CLASSES)), np.nan, dtype=np.float32)
    runs = []
    for k in folds:
        tr, va, _ = fold_masks(d.windows, val_fold=k)
        X_tr, y_tr, X_va, y_va, scaler = mlp_arrays(d, tr, va, label_col=cfg["label_col"])
        model, hist = train_mlp(X_tr, y_tr, X_va, y_va, cfg)
        oof[va] = predict_proba(model, X_va)
        runs.append({"fold": k, "model": model, "scaler": scaler, "history": hist})
        best = hist.loc[hist.val_loss.idxmin()]
        print(f"fold {k}: best epoch {int(best.epoch)}, val loss {best.val_loss:.3f}, window acc {best.val_acc:.3f}",
              flush=True)
    return d.windows[dev_mask], oof[dev_mask], runs


def predict_ensemble(runs, X_raw):
    """Average window probabilities of all fold models (each with its own scaler).
    X_raw: un-normalised rows of d.X_mlp."""
    return np.mean([predict_proba(r["model"], r["scaler"].transform(X_raw)) for r in runs], 0)


# ---------------------------------------------------------------- save / load fold models

def save_runs(runs, cfg, out_dir):
    for r in runs:
        torch.save({"cfg": cfg, "state_dict": r["model"].state_dict(), "n_in": r["model"].net[0].in_features,
                    "scaler_mean": r["scaler"].mean_, "scaler_std": r["scaler"].std_},
                   out_dir / f"fold{r['fold']}.pt")
        r["history"].to_csv(out_dir / f"history_fold{r['fold']}.csv", index=False)


def load_runs(out_dir=result_dir("mlp")):
    runs = []
    for p in sorted(out_dir.glob("fold*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        model = MLP(ck["n_in"], ck["cfg"]["hidden"], ck["cfg"]["dropout"])
        model.load_state_dict(ck["state_dict"])
        scaler = Standardizer()
        scaler.mean_, scaler.std_ = ck["scaler_mean"], ck["scaler_std"]
        runs.append({"fold": int(p.stem[4:]), "model": model.eval(), "scaler": scaler})
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=DEFAULT_CFG["max_epochs"])
    ap.add_argument("--patience", type=int, default=DEFAULT_CFG["patience"])
    ap.add_argument("--lr", type=float, default=DEFAULT_CFG["lr"])
    ap.add_argument("--batch-size", type=int, default=DEFAULT_CFG["batch_size"])
    ap.add_argument("--dropout", type=float, default=DEFAULT_CFG["dropout"])
    ap.add_argument("--weight-power", type=float, default=DEFAULT_CFG["weight_power"])
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--name", default="mlp", help="output folder under results/")
    a = ap.parse_args()
    cfg = DEFAULT_CFG | dict(max_epochs=a.epochs, patience=a.patience, lr=a.lr, batch_size=a.batch_size,
                             dropout=a.dropout, weight_power=a.weight_power)
    print(cfg, "| device:", device(), flush=True)

    d = load_processed()
    dev, oof, runs = cross_validate(d, cfg, a.folds)
    pat = patient_scores(dev, oof)
    table, _, rep = cv_report(pat)
    t_final, _ = tune_thresholds(pat)
    print(table.round(3).to_string())
    print(f"WA (tuned, nested): {table['WA (tuned)'].mean():.3f} ± {table['WA (tuned)'].std():.3f}")

    out = save_oof(a.name, dev.index, oof,
                   summary(table, rep) | {"cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
                                          "thresholds": t_final})
    save_runs(runs, cfg, out)
    print("saved to", out)


if __name__ == "__main__":
    main()
