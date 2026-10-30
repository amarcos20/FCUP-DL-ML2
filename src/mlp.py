"""MLP on window-level features: model, training with early stopping, and
aggregation of window predictions into one prediction per patient.

Aggregation: window probabilities are averaged per recording; a patient gets the
maximum over recordings of P(Present) and P(Unknown) (a murmur may be audible at a
single location). Decision rule, as in the Challenge's hierarchical approach:
Present if p_present >= t_present, else Unknown if p_unknown >= t_unknown, else Absent.
Thresholds are chosen on validation data to maximise the weighted accuracy.
"""
import copy

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.metrics import CLASSES, weighted_accuracy

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


def encode(labels):
    return np.array([CLASSES.index(l) for l in labels])


def class_weights(y, power=1.0):
    counts = np.bincount(y, minlength=len(CLASSES))
    w = (counts.sum() / (len(CLASSES) * counts)) ** power
    return torch.tensor(w, dtype=torch.float32)


def predict_proba(model, X, batch_size=4096):
    model.eval()
    with torch.no_grad():
        out = [torch.softmax(model(torch.from_numpy(X[i:i + batch_size])), 1) for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


def train_mlp(X_tr, y_tr, X_va, y_va, cfg):
    """Train with weighted cross-entropy and Adam; early stopping on validation loss.
    Returns the best model and the per-epoch history."""
    torch.manual_seed(cfg["seed"])
    model = MLP(X_tr.shape[1], cfg["hidden"], cfg["dropout"])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_tr, cfg["weight_power"]))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    Xv, yv = torch.from_numpy(X_va), torch.from_numpy(y_va)
    best, best_loss, wait, history = None, np.inf, 0, []
    for epoch in range(cfg["max_epochs"]):
        model.train()
        perm, total = torch.randperm(len(Xt)), 0.0
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


def cross_validate(df, feats, cfg=DEFAULT_CFG, folds=range(5)):
    """Train one model per CV fold (test patients, fold -1, are never used).
    Returns out-of-fold window probabilities and, per fold, (model, scaler, history)."""
    dev = df[df["fold"] >= 0]
    oof = np.zeros((len(dev), len(CLASSES)), dtype=np.float32)
    runs = []
    for k in folds:
        tr, va = (dev["fold"] != k).values, (dev["fold"] == k).values
        scaler = StandardScaler().fit(dev.loc[tr, feats])
        X_tr = scaler.transform(dev.loc[tr, feats]).astype(np.float32)
        X_va = scaler.transform(dev.loc[va, feats]).astype(np.float32)
        y_tr, y_va = encode(dev.loc[tr, cfg["label_col"]]), encode(dev.loc[va, cfg["label_col"]])
        model, hist = train_mlp(X_tr, y_tr, X_va, y_va, cfg)
        oof[va] = predict_proba(model, X_va)
        runs.append({"fold": k, "model": model, "scaler": scaler, "history": hist})
        best = hist.loc[hist.val_loss.idxmin()]
        print(f"fold {k}: best epoch {int(best.epoch)}, val loss {best.val_loss:.3f}, window acc {best.val_acc:.3f}")
    return dev, oof, runs


def predict_ensemble(runs, df, feats):
    """Average window probabilities of all fold models (each with its own scaler)."""
    return np.mean([predict_proba(r["model"], r["scaler"].transform(df[feats]).astype(np.float32)) for r in runs], 0)


def patient_scores(win_df, probs):
    """Window probabilities -> one row per patient: p_present, p_unknown, p_absent, murmur (true label)."""
    d = win_df[["patient_id", "recording", "murmur"]].copy()
    d[CLASSES] = probs
    rec = d.groupby(["patient_id", "recording"]).agg({**{c: "mean" for c in CLASSES}, "murmur": "first"})
    pat = rec.groupby("patient_id").agg(p_present=("Present", "max"), p_unknown=("Unknown", "max"),
                                        p_absent=("Absent", "mean"), murmur=("murmur", "first"))
    return pat


def predict_rule(pat, t_present, t_unknown):
    return np.where(pat.p_present >= t_present, "Present",
                    np.where(pat.p_unknown >= t_unknown, "Unknown", "Absent"))


def tune_thresholds(pat, grid=np.round(np.arange(0.05, 1.0, 0.025), 3)):
    """Grid search of (t_present, t_unknown) maximising the weighted accuracy on `pat`."""
    best_wa, best_t = -1.0, None
    for tp in grid:
        for tu in grid:
            wa = weighted_accuracy(pat.murmur, predict_rule(pat, tp, tu))
            if wa > best_wa:
                best_wa, best_t = wa, (tp, tu)
    return best_t, best_wa
