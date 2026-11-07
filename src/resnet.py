"""ResNet50 on the same log-mel windows as src.cnn, after Walker et al. 2022 ("Dual Bayesian ResNet",
4th in the PhysioNet Challenge 2022, literature/04_*). Only the network changes: input (40 mel x 401
frames = 4 s), per-band standardisation on the training folds, augmentation, recording-level labels,
patient aggregation and threshold tuning are exactly those of the CNN.

Two tasks, to separate the two ideas of the paper:

  --task multiclass  (step 1: effect of the architecture)
      one ResNet50 with 3 outputs, weighted cross-entropy -- same training as src.cnn.
  --task dual        (step 2: effect of splitting the problem, the paper's model)
      two Bayesian ResNet50: Present vs rest and Unknown vs rest. "Bayesian" = Monte Carlo dropout:
      dropout inside every residual block, kept on at prediction time; the prediction is the mean
      of `mc_samples` stochastic forward passes (30 in the paper).
      The window scores are [P(Present), P(Unknown), (1-P(Present))(1-P(Unknown))], so the
      hierarchical rule of src.evaluate (Present -> Unknown -> Absent) uses each binary model
      directly, as in the paper.

Differences from the paper (kept on purpose so that the comparison with our CNN is fair):
  * our input (40 mel bands, 20-1000 Hz) instead of 64 bands up to 2 kHz;
  * one input channel: with --pretrained the ImageNet conv1 filters are summed over RGB, which is
    the same as repeating the spectrogram 3 times (what the paper does) but cheaper;
  * class-weighted cross-entropy for both binary models (the paper uses a balanced sampler for
    the Unknown model) and early stopping on the validation loss (paper: balanced accuracy);
  * validation during training is deterministic (dropout off); MC dropout is used for the
    out-of-fold predictions only.
  The paper starts from ImageNet weights with lr 3e-5 (--pretrained --lr 3e-5); the default here is
  from scratch with the CNN's optimiser settings.

Usage (5 CV folds, saves results/<name>/{oof.csv, cv_summary.json, history_*.csv, fold*.pt}):
    python -m src.resnet --task multiclass --name resnet50
    python -m src.resnet --task dual --name dual_resnet50
    python -m src.resnet --task dual --pretrained --lr 3e-5 --name dual_resnet50_imagenet
"""
import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cnn import apply_spec_augment, device, loader
from src.data import CLASSES
from src.datasets import CNNWindows, class_weights, fold_masks, load_processed, mel_band_stats
from src.evaluate import RESULTS_DIR, cv_report, patient_scores, save_oof, summary, tune_thresholds

IMAGENET_URL = "https://download.pytorch.org/models/resnet50-0676ba61.pth"  # torchvision IMAGENET1K_V1

DEFAULT_RESNET_CFG = dict(
    task="multiclass",
    pretrained=False,
    block_dropout=0.0,      # dropout inside the residual blocks (MC dropout); dual default 0.3
    head_dropout=0.3,       # dropout before the final linear layer (as in our CNN)
    mc_samples=1,           # stochastic passes at prediction time; dual default 30
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=128,
    max_epochs=50,
    patience=8,
    weight_power=1.0,
    spec_augment=True,
    random_crop=True,
    label_col="rec_label",
    num_workers=4,
    seed=42,
)
DUAL_DEFAULTS = dict(block_dropout=0.3, mc_samples=30)  # Walker et al.: dropout 0.3, 30 passes
BINARY_TARGETS = {"present": CLASSES.index("Present"), "unknown": CLASSES.index("Unknown")}


# -----------------------------------------------------------------------------
# ResNet50 with optional dropout in the blocks (module names follow torchvision, so the
# ImageNet state_dict loads directly)
# -----------------------------------------------------------------------------

class MCDropout(nn.Module):
    """Dropout that can stay active in eval mode (model.mc = True) for Monte Carlo sampling."""

    def __init__(self, p):
        super().__init__()
        self.p, self.mc = p, False

    def forward(self, x):
        return F.dropout(x, self.p, training=self.training or self.mc) if self.p > 0 else x


class Bottleneck(nn.Module):
    """1x1 -> 3x3 (stride) -> 1x1 convolutions with a skip connection; dropout after each ReLU
    (placement of Walker et al.)."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.drop = MCDropout(p)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.drop(self.relu(self.bn1(self.conv1(x))))
        out = self.drop(self.relu(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        return self.drop(self.relu(out + identity))


class ResNet50(nn.Module):
    """ResNet50 (3-4-6-3 bottleneck blocks, 64 -> 2048 maps) for 1-channel spectrograms.

    [B, 1, 40, 401] -> stem (conv 7x7 /2, maxpool /2) -> [B, 64, 10, 101]
    -> layer1 [B, 256, 10, 101] -> layer2 [B, 512, 5, 51] -> layer3 [B, 1024, 3, 26]
    -> layer4 [B, 2048, 2, 13] -> global average pooling -> dropout -> linear (n_out logits)."""

    def __init__(self, n_out=len(CLASSES), block_dropout=0.0, head_dropout=0.3):
        super().__init__()
        self.inplanes, self.p = 64, block_dropout
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.stem_drop = MCDropout(block_dropout)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head_drop = MCDropout(head_dropout)
        self.fc = nn.Linear(2048, n_out)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        for m in self.modules():  # zero-init the last BN of each block: every block starts as identity
            if isinstance(m, Bottleneck):
                nn.init.zeros_(m.bn3.weight)

    def _make_layer(self, planes, blocks, stride=1):
        down = None
        if stride != 1 or self.inplanes != planes * 4:
            down = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, 1, stride=stride, bias=False),
                                 nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, down, self.p)]
        self.inplanes = planes * 4
        layers += [Bottleneck(self.inplanes, planes, p=self.p) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def set_mc(self, on):
        for m in self.modules():
            if isinstance(m, MCDropout):
                m.mc = on

    def forward(self, x):
        x = self.stem_drop(self.maxpool(self.relu(self.bn1(self.conv1(x)))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.head_drop(torch.flatten(self.avgpool(x), 1)))


def load_imagenet(model):
    """ImageNet weights (torchvision ResNet50 V1), conv1 summed over RGB, fc left random."""
    sd = torch.hub.load_state_dict_from_url(IMAGENET_URL, map_location="cpu", progress=False)
    sd["conv1.weight"] = sd["conv1.weight"].sum(1, keepdim=True)
    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert set(missing) == {"fc.weight", "fc.bias"} and not unexpected, (missing, unexpected)
    return model


def build_model(cfg, n_out):
    model = ResNet50(n_out, cfg["block_dropout"], cfg["head_dropout"])
    return load_imagenet(model) if cfg["pretrained"] else model


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# -----------------------------------------------------------------------------
# Training and prediction (same loop as src.cnn; `target` turns it into a binary task)
# -----------------------------------------------------------------------------

def to_target(y, target):
    """3-class indices -> binary {0: rest, 1: target class} when target is not None."""
    return y if target is None else (y == target).long()


def predict_resnet_proba(model, ds, cfg, mc_samples=1):
    """Softmax probabilities for every window of a CNNWindows dataset (in order), averaged over
    `mc_samples` passes with dropout on (mc_samples=1: deterministic)."""
    dev = next(model.parameters()).device
    model.eval()
    model.set_mc(mc_samples > 1)
    probs = []
    with torch.no_grad():
        for x, _ in loader(ds, cfg, shuffle=False):
            x = x.to(dev)
            p = torch.stack([torch.softmax(model(x), 1) for _ in range(max(mc_samples, 1))]).mean(0)
            probs.append(p.cpu().numpy())
    model.set_mc(False)
    return np.concatenate(probs, axis=0)


def train_resnet(ds_tr, ds_va, cfg, fold_idx=0, target=None, tag="multiclass"):
    """Weighted cross-entropy + Adam, early stopping on validation loss (as src.cnn.train_cnn)."""
    torch.manual_seed(cfg["seed"] + fold_idx)
    np.random.seed(cfg["seed"] + fold_idx)
    dev = device()

    n_out = len(CLASSES) if target is None else 2
    model = build_model(cfg, n_out).to(dev)
    y_tr = ds_tr.y if target is None else (ds_tr.y == target).astype(np.int64)
    w = torch.from_numpy(class_weights(y_tr, n_classes=n_out, power=cfg["weight_power"])).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    train_loader, val_loader = loader(ds_tr, cfg, shuffle=True), loader(ds_va, cfg, shuffle=False)

    best_loss, best_weights, wait, history = float("inf"), None, 0, []
    print(f"\n>>> [Fold {fold_idx} | {tag}] {len(ds_tr)} train windows, {len(ds_va)} val windows, "
          f"{n_params(model) / 1e6:.1f} M parameters, class weights {w.cpu().numpy().round(2)} <<<", flush=True)

    for epoch in range(cfg["max_epochs"]):
        model.train()
        train_loss, n_seen = 0.0, 0
        for x_b, y_b in train_loader:
            if cfg["spec_augment"]:
                x_b = apply_spec_augment(x_b)
            x_b, y_b = x_b.to(dev), to_target(y_b, target).to(dev)
            optimizer.zero_grad()
            loss = loss_fn(model(x_b), y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_b)
            n_seen += len(x_b)
        train_loss /= n_seen

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(dev), to_target(y_v, target).to(dev)
                logits = model(x_v)
                val_loss += loss_fn(logits, y_v).item() * len(x_v)
                correct += (logits.argmax(dim=1) == y_v).sum().item()
                total += len(y_v)
        val_loss /= total
        val_acc = correct / total

        is_best = val_loss < best_loss
        print(f"  Fold {fold_idx} {tag} | epoch {epoch + 1:02d}/{cfg['max_epochs']} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val acc {val_acc:.4f}{' *' if is_best else ''}", flush=True)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

        if is_best:
            best_loss, best_weights, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"  Fold {fold_idx} {tag} | early stopping at epoch {epoch + 1}", flush=True)
                break

    model.load_state_dict(best_weights)
    return model, pd.DataFrame(history)


def heads(cfg):
    """(tag, target class index) of the networks trained per fold."""
    return [("multiclass", None)] if cfg["task"] == "multiclass" else list(BINARY_TARGETS.items())


def window_scores(models, ds, cfg):
    """Window scores (n, 3) in CLASSES order from the fold's network(s)."""
    if cfg["task"] == "multiclass":
        return predict_resnet_proba(models["multiclass"], ds, cfg, cfg["mc_samples"])
    p = predict_resnet_proba(models["present"], ds, cfg, cfg["mc_samples"])[:, 1]
    u = predict_resnet_proba(models["unknown"], ds, cfg, cfg["mc_samples"])[:, 1]
    return np.stack([p, u, (1 - p) * (1 - u)], axis=1).astype(np.float32)


def cross_validate_resnet(d, cfg=DEFAULT_RESNET_CFG, folds=range(5)):
    """One model (multiclass) or two models (dual) per CV fold; test patients (fold -1) never used.

    Returns the validation windows of `folds`, their out-of-fold scores and per fold a dict with
    models, band_stats and histories."""
    dev_mask = d.windows["fold"].isin(folds).to_numpy()
    oof = np.full((len(d.windows), len(CLASSES)), np.nan, dtype=np.float32)
    runs = []
    for k in folds:
        tr, va, _ = fold_masks(d.windows, val_fold=k)
        band_stats = mel_band_stats(d, tr)
        ds_tr = CNNWindows(d, tr, band_stats, label_col=cfg["label_col"], random_crop=cfg["random_crop"])
        ds_va = CNNWindows(d, va, band_stats, label_col=cfg["label_col"])
        models, hists = {}, {}
        for tag, target in heads(cfg):
            models[tag], hists[tag] = train_resnet(ds_tr, ds_va, cfg, fold_idx=k, target=target, tag=tag)
            best = hists[tag].loc[hists[tag].val_loss.idxmin()]
            print(f"==> Fold {k} {tag}: best epoch {int(best.epoch) + 1} | val loss {best.val_loss:.4f} | "
                  f"val acc {best.val_acc:.4f}", flush=True)
        oof[va] = window_scores(models, ds_va, cfg)
        runs.append({"fold": k, "models": models, "band_stats": band_stats, "history": hists})
    return d.windows[dev_mask], oof[dev_mask], runs


def predict_ensemble_resnet(runs, d, mask, cfg):
    """Average window scores of all fold models (each with its own band statistics)."""
    out = []
    for r in runs:
        models = {t: m.to(device()) for t, m in r["models"].items()}
        out.append(window_scores(models, CNNWindows(d, mask, r["band_stats"]), cfg))
    return np.mean(out, axis=0)


# -----------------------------------------------------------------------------
# Save / load fold models
# -----------------------------------------------------------------------------

def save_runs(runs, cfg, out_dir):
    for r in runs:
        for tag, model in r["models"].items():
            suffix = "" if tag == "multiclass" else f"_{tag}"
            torch.save({"cfg": cfg, "tag": tag, "state_dict": model.state_dict(), "band_stats": r["band_stats"]},
                       out_dir / f"fold{r['fold']}{suffix}.pt")
            r["history"][tag].to_csv(out_dir / f"history_fold{r['fold']}{suffix}.csv", index=False)


def load_runs(out_dir=RESULTS_DIR / "resnet50"):
    """Returns (runs, cfg)."""
    runs, cfg = {}, None
    for p in sorted(out_dir.glob("fold*.pt")):
        ck = torch.load(p, map_location=device(), weights_only=False)
        cfg, tag = ck["cfg"], ck["tag"]
        model = ResNet50(len(CLASSES) if tag == "multiclass" else 2, cfg["block_dropout"], cfg["head_dropout"])
        model.load_state_dict(ck["state_dict"])
        k = int(p.stem[4:].split("_")[0])
        r = runs.setdefault(k, {"fold": k, "models": {}, "band_stats": ck["band_stats"]})
        r["models"][tag] = model.to(device()).eval()
    return [runs[k] for k in sorted(runs)], cfg


def main():
    ap = argparse.ArgumentParser()
    c = DEFAULT_RESNET_CFG
    ap.add_argument("--task", choices=["multiclass", "dual"], default=c["task"])
    ap.add_argument("--pretrained", action="store_true", help="start from ImageNet weights")
    ap.add_argument("--block-dropout", type=float, help="default 0 (multiclass) / 0.3 (dual)")
    ap.add_argument("--head-dropout", type=float, default=c["head_dropout"])
    ap.add_argument("--mc-samples", type=int, help="default 1 (multiclass) / 30 (dual)")
    ap.add_argument("--epochs", type=int, default=c["max_epochs"])
    ap.add_argument("--patience", type=int, default=c["patience"])
    ap.add_argument("--lr", type=float, default=c["lr"])
    ap.add_argument("--batch-size", type=int, default=c["batch_size"])
    ap.add_argument("--weight-power", type=float, default=c["weight_power"])
    ap.add_argument("--no-spec-augment", action="store_true")
    ap.add_argument("--no-random-crop", action="store_true")
    ap.add_argument("--workers", type=int, default=c["num_workers"])
    ap.add_argument("--in-memory", action="store_true", help="load mel.npy into RAM (~0.6 GB) instead of mmap")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--name", help="output folder under results/ (default: resnet50 / dual_resnet50)")
    a = ap.parse_args()

    cfg = c | (DUAL_DEFAULTS if a.task == "dual" else {}) | dict(
        task=a.task, pretrained=a.pretrained, head_dropout=a.head_dropout, max_epochs=a.epochs,
        patience=a.patience, lr=a.lr, batch_size=a.batch_size, weight_power=a.weight_power,
        spec_augment=not a.no_spec_augment, random_crop=not a.no_random_crop, num_workers=a.workers)
    if a.block_dropout is not None:
        cfg["block_dropout"] = a.block_dropout
    if a.mc_samples is not None:
        cfg["mc_samples"] = a.mc_samples
    name = a.name or ("resnet50" if a.task == "multiclass" else "dual_resnet50")
    print(cfg, "| device:", device(), flush=True)

    d = load_processed(mmap=not a.in_memory)
    dev, oof, runs = cross_validate_resnet(d, cfg, a.folds)
    pat = patient_scores(dev, oof)
    table, _, rep = cv_report(pat)
    t_final, _ = tune_thresholds(pat)
    print(table.round(3).to_string())
    print(f"WA (tuned, nested): {table['WA (tuned)'].mean():.3f} ± {table['WA (tuned)'].std():.3f}")

    out = save_oof(name, dev.index, oof, summary(table, rep) | {"cfg": cfg, "thresholds": t_final})
    save_runs(runs, cfg, out)
    print("saved to", out)


if __name__ == "__main__":
    main()
