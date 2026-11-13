"""RNN on the same log-mel windows as src.cnn: each 4 s window is read as a sequence of 401 frames
(10 ms each), every frame a vector of 40 mel bands.

Why an RNN: the heart cycle is a sequence (S1 -> systole -> S2 -> diastole) and a murmur is defined by
*when* it happens (01_eda: almost all murmurs are systolic, between S1 and S2). A recurrent network reads
the frames in order and keeps a memory of what came before; a bidirectional one also sees what comes after,
so at every frame it knows where it is in the cycle.

Everything except the network is the CNN pipeline (src.cnn.cross_validate_cnn with model_fn): per-band
standardisation with training statistics, SpecAugment + random time shift, recording-level labels, class-
weighted cross-entropy, Adam, early stopping on validation loss, patient aggregation and nested threshold
tuning (src.evaluate). So CNN vs RNN compares only "local time-frequency patterns" vs "sequence model".

Network: [B, 1, 40, 401] -> sequence [B, 401, 40] -> GRU/LSTM (bidirectional, stacked, dropout between
layers) -> one vector per window by pooling the outputs over time -> dropout -> linear (3 logits).
  --pool mean       average of the outputs of all frames
  --pool attention  learned weighted average: a small layer scores every frame and the softmax of the scores
                    gives the weights, so the network can focus on the frames with a murmur (systole)

Usage (5 CV folds, saves results/<name>/{oof.csv, cv_summary.json, history_fold*.csv, fold*.pt}):
    python -m src.rnn                                   # iteration 1: BiGRU, mean pooling -> results/rnn
    python -m src.rnn --pool attention --name rnn_att   # iteration 2
    python -m src.rnn --cell lstm --name rnn_lstm
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from src.cnn import DEFAULT_CNN_CFG, cross_validate_cnn, device, predict_cnn_proba
from src.cnn import save_runs
from src.data import CLASSES
from src.datasets import CNNWindows, load_processed
from src.evaluate import RESULTS_DIR, result_dir, cv_report, patient_scores, save_oof, summary, tune_thresholds

DEFAULT_RNN_CFG = DEFAULT_CNN_CFG | dict(
    cell="gru", hidden=64, layers=2, bidirectional=True, pool="mean",
    rnn_dropout=0.3,          # between stacked recurrent layers
    dropout=0.3,              # before the output layer (as in the CNN)
    max_epochs=50, patience=8,  # as the CNN run in results/cnn
)


class HeartMurmurRNN(nn.Module):
    def __init__(self, n_mels=40, cell="gru", hidden=64, layers=2, bidirectional=True, pool="mean",
                 rnn_dropout=0.3, dropout=0.3, n_out=len(CLASSES)):
        super().__init__()
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn(n_mels, hidden, num_layers=layers, batch_first=True, bidirectional=bidirectional,
                       dropout=rnn_dropout if layers > 1 else 0.0)
        d = hidden * (2 if bidirectional else 1)
        self.pool = pool
        self.att = nn.Linear(d, 1) if pool == "attention" else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_out))

    def forward(self, x, return_weights=False):
        h, _ = self.rnn(x[:, 0].transpose(1, 2))           # [B, 1, F, T] -> [B, T, F] -> [B, T, d]
        if self.pool == "attention":
            w = torch.softmax(self.att(h), dim=1)           # [B, T, 1], sums to 1 over time
            z = (w * h).sum(1)
        else:
            w, z = None, h.mean(1)
        out = self.head(z)
        return (out, w) if return_weights else out


def build_model(cfg):
    return HeartMurmurRNN(cell=cfg["cell"], hidden=cfg["hidden"], layers=cfg["layers"],
                          bidirectional=cfg["bidirectional"], pool=cfg["pool"],
                          rnn_dropout=cfg["rnn_dropout"], dropout=cfg["dropout"])


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def load_runs(out_dir=result_dir("rnn")):
    """Returns (runs, cfg)."""
    runs, cfg = [], None
    for p in sorted(out_dir.glob("fold*.pt")):
        ck = torch.load(p, map_location=device(), weights_only=False)
        cfg = ck["cfg"]
        model = build_model(cfg)
        model.load_state_dict(ck["state_dict"])
        runs.append({"fold": int(p.stem[4:]), "model": model.to(device()).eval(), "band_stats": ck["band_stats"]})
    return runs, cfg


def predict_ensemble_rnn(runs, d, mask, cfg):
    """Average window probabilities of all fold models (each with its own band statistics)."""
    return np.mean([predict_cnn_proba(r["model"], CNNWindows(d, mask, r["band_stats"]), cfg) for r in runs], axis=0)


def main():
    ap = argparse.ArgumentParser()
    c = DEFAULT_RNN_CFG
    ap.add_argument("--cell", choices=["gru", "lstm"], default=c["cell"])
    ap.add_argument("--hidden", type=int, default=c["hidden"])
    ap.add_argument("--layers", type=int, default=c["layers"])
    ap.add_argument("--unidirectional", action="store_true")
    ap.add_argument("--pool", choices=["mean", "attention"], default=c["pool"])
    ap.add_argument("--rnn-dropout", type=float, default=c["rnn_dropout"])
    ap.add_argument("--dropout", type=float, default=c["dropout"])
    ap.add_argument("--epochs", type=int, default=c["max_epochs"])
    ap.add_argument("--patience", type=int, default=c["patience"])
    ap.add_argument("--lr", type=float, default=c["lr"])
    ap.add_argument("--batch-size", type=int, default=c["batch_size"])
    ap.add_argument("--no-spec-augment", action="store_true")
    ap.add_argument("--no-random-crop", action="store_true")
    ap.add_argument("--monitor", choices=["loss", "wa"], default=c["monitor"], help="early-stopping metric")
    ap.add_argument("--scheduler", choices=["none", "plateau"], default=c["scheduler"])
    ap.add_argument("--plateau-patience", type=int, default=c["plateau_patience"])
    ap.add_argument("--workers", type=int, default=c["num_workers"])
    ap.add_argument("--in-memory", action="store_true", help="load mel.npy into RAM (~0.6 GB) instead of mmap")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--name", default="rnn", help="output folder under results/")
    a = ap.parse_args()
    cfg = c | dict(cell=a.cell, hidden=a.hidden, layers=a.layers, bidirectional=not a.unidirectional, pool=a.pool,
                   rnn_dropout=a.rnn_dropout, dropout=a.dropout, max_epochs=a.epochs, patience=a.patience, lr=a.lr,
                   batch_size=a.batch_size, spec_augment=not a.no_spec_augment, random_crop=not a.no_random_crop,
                   num_workers=a.workers, monitor=a.monitor, scheduler=a.scheduler,
                   plateau_patience=a.plateau_patience)
    print(cfg, "| device:", device(), "| parameters:", n_params(build_model(cfg)), flush=True)

    d = load_processed(mmap=not a.in_memory)
    dev, oof, runs = cross_validate_cnn(d, cfg, a.folds, model_fn=build_model)
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
