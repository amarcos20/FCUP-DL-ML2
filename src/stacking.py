"""Stacking (late fusion): averaged out-of-fold scores of an audio model + demographics -> logistic regression.

The CNN / RNN are not retrained. For each CV patient the averaged oof.csv of `--runs` gives P(Present) and
P(Unknown) (max over recordings). A multinomial logistic regression (class_weight="balanced", L2 with C) takes
their logits (standardised) plus the encoded demographics; the usual two-threshold rule gives the class.

Evaluation is nested on the 5 CV folds (the test fold is not used): for each outer fold the encoder and the
regression are fitted on the other 4 folds, and the thresholds are tuned on an inner CV over those 4 folds.

Variants: base (the audio model alone, src.evaluate.cv_report), meta_audio (regression on the audio scores only),
meta_age, meta_age_height. The printed adoption rule (+0.02 mean WA and better in >= 4/5 folds) compares each
demographic variant with meta_audio; the comparison with base is in notebooks/07_final_experiments.ipynb.

    python -m src.stacking --runs rnn rnn_s1 rnn_s2 rnn_s3 --name stack_rnn

Outputs in results/<name>/: stacking_summary.json, stack_<variant>.json (meta-model refitted on all CV patients,
with its thresholds; for src.predict --stack) and meta_oof_<variant>.csv.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src import demographics
from src.data import CLASSES
from src.datasets import PROC_DIR
from src.evaluate import RESULTS_DIR, cv_report, load_oof, patient_scores, predict_rule, tune_thresholds
from src.metrics import report, weighted_accuracy

VARIANTS = {
    "base": None,
    "meta_audio": [],
    "meta_age": ["age"],
    "meta_age_height": ["age", "height"],
    "meta_age_height_weight": ["age", "height", "weight"],
    "meta_all": ["age", "height", "weight", "sex"],
}
DEFAULT_VARIANTS = ["base", "meta_audio", "meta_age", "meta_age_height"]
EPS = 1e-4


# ------------------------------------------------------------------------------------------ inputs

def audio_patient_scores(runs):
    """Patient scores of the averaged out-of-fold predictions of `runs` (all must share the same windows)."""
    windows = pd.read_csv(PROC_DIR / "windows.csv")
    idx0, probs = None, []
    for n in runs:
        idx, p = load_oof(n)
        if idx0 is None:
            idx0 = idx
        elif not np.array_equal(idx, idx0):
            raise SystemExit(f"{n}: out-of-fold windows differ from {runs[0]}")
        probs.append(p)
    return patient_scores(windows.loc[idx0], np.mean(probs, axis=0))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------------------------ meta-model

def fit_meta(pat, features, table, C=1.0):
    """Fit encoder + logistic regression on the patients of `pat`. Returns a JSON-serialisable dict."""
    a = np.column_stack([_logit(pat["Present"]), _logit(pat["Unknown"])])
    mu, sd = a.mean(0), a.std(0) + 1e-6
    X = (a - mu) / sd
    enc = None
    if features:
        enc = demographics.DemoEncoder(features).fit(table.loc[[str(p) for p in pat.index]])
        X = np.hstack([X, demographics.patient_matrix(enc, table, pat.index)])
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(X, pat["murmur"].to_numpy())
    names = ["logit_P(Present)", "logit_P(Unknown)"] + (enc.names() if enc else [])
    return {"features": list(features), "encoder": enc.state() if enc else None,
            "audio_mean": mu.tolist(), "audio_std": sd.tolist(), "input_names": names,
            "classes": [str(c) for c in lr.classes_], "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(),
            "C": C}


def apply_stack(m, pat, table):
    """Meta-model probabilities (columns CLASSES) for patient scores `pat` (+ demographics from `table`)."""
    a = np.column_stack([_logit(pat["Present"]), _logit(pat["Unknown"])])
    X = (a - np.array(m["audio_mean"])) / np.array(m["audio_std"])
    if m["encoder"]:
        enc = demographics.DemoEncoder.from_state(m["encoder"])
        X = np.hstack([X, demographics.patient_matrix(enc, table, pat.index)])
    z = X @ np.array(m["coef"]).T + np.array(m["intercept"])
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return pd.DataFrame(p, index=pat.index, columns=m["classes"])[CLASSES]


def oof_meta(pat, features, table, C):
    """Out-of-fold meta probabilities over the folds present in `pat` (fit on the others, predict one)."""
    out = []
    for k in sorted(pat["fold"].unique()):
        m = fit_meta(pat[pat.fold != k], features, table, C)
        out.append(apply_stack(m, pat[pat.fold == k], table))
    return pd.concat(out).loc[pat.index]


def nested_eval(pat, features, table, C):
    """Per-fold table, pooled predictions, pooled report and outer out-of-fold meta scores (see module doc)."""
    rows, pred, outer = [], pd.Series(index=pat.index, dtype=object), []
    for k in sorted(pat["fold"].unique()):
        tr, va = pat[pat.fold != k], pat[pat.fold == k]
        pv = apply_stack(fit_meta(tr, features, table, C), va, table)
        inner = oof_meta(tr, features, table, C).join(tr[["murmur", "fold"]])
        t, _ = tune_thresholds(inner)
        pred[va.index] = predict_rule(pv, *t)
        outer.append(pv)
        rows.append({"fold": k, "t_present": t[0], "t_unknown": t[1],
                     "WA (tuned)": weighted_accuracy(va["murmur"], pred[va.index])})
    return pd.DataFrame(rows).set_index("fold"), pred, report(pat["murmur"], pred), pd.concat(outer).loc[pat.index]


def aucs(scores, y):
    return {c: float(roc_auc_score(y == c, scores[c])) for c in ("Present", "Unknown")}


def adopt(fold_wa, ref_wa, min_gain=0.02, min_folds=4):
    gain = float(np.mean(fold_wa) - np.mean(ref_wa))
    better = int(np.sum(np.asarray(fold_wa) > np.asarray(ref_wa)))
    return {"gain": gain, "folds_better": better, "adopted": bool(gain >= min_gain and better >= min_folds)}


# ------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="runs whose oof.csv are averaged (e.g. the 4 seeds)")
    ap.add_argument("--name", required=True, help="output folder results/<name>/")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 strength of the logistic regression")
    a = ap.parse_args()

    table = demographics.read_table()
    pat = audio_patient_scores(a.runs)
    y = pat["murmur"]
    print(f"{len(pat)} CV patients | runs {a.runs}", flush=True)

    out_dir = RESULTS_DIR / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    res, fold_tables = {}, {}
    for v in a.variants:
        feats = VARIANTS[v]
        if feats is None:
            table_v, _, rep = cv_report(pat)
            scores = pat[CLASSES]
        else:
            table_v, _, rep, scores = nested_eval(pat, feats, table, a.C)
            scores.rename_axis("patient_id").to_csv(out_dir / f"meta_oof_{v}.csv", float_format="%.5f")
            final = fit_meta(pat, feats, table, a.C)
            t_final, _ = tune_thresholds(oof_meta(pat, feats, table, a.C).join(pat[["murmur", "fold"]]))
            final |= {"variant": v, "base_runs": a.runs, "thresholds": list(t_final)}
            (out_dir / f"stack_{v}.json").write_text(json.dumps(final, indent=2))
        fold_tables[v] = table_v["WA (tuned)"]
        res[v] = {"features": feats, "wa_mean": float(table_v["WA (tuned)"].mean()),
                  "wa_std": float(table_v["WA (tuned)"].std()), "wa_per_fold": table_v["WA (tuned)"].round(4).tolist(),
                  "wa_pooled": rep["weighted_accuracy"], "recall": {c: float(r) for c, r in rep["recall"].items()},
                  "confusion": rep["confusion"].tolist(), "auc": aucs(scores, y)}
        print(f"{v:24s} WA {res[v]['wa_mean']:.3f} ± {res[v]['wa_std']:.3f} | folds {res[v]['wa_per_fold']} | "
              f"recall P/U/A {[round(x, 2) for x in res[v]['recall'].values()]} | AUC P/U "
              f"{res[v]['auc']['Present']:.3f}/{res[v]['auc']['Unknown']:.3f}", flush=True)

    if "meta_audio" in fold_tables:
        print("\nAdoption rule vs meta_audio (mean WA +0.02 and better in >= 4/5 folds):")
        for v in a.variants:
            if VARIANTS[v]:
                res[v]["vs_meta_audio"] = adopt(fold_tables[v], fold_tables["meta_audio"])
                print(f"  {v:24s} {res[v]['vs_meta_audio']}")
    (out_dir / "stacking_summary.json").write_text(json.dumps({"runs": a.runs, "C": a.C, "variants": res}, indent=2))
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
