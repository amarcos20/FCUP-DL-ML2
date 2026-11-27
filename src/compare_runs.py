"""Compare runs and groups of seeds, with every number recomputed from results/<run>/oof.csv (not taken from
cv_summary.json).

Groups are separated by spaces, runs of a group by commas. The first group is the reference.

    python -m src.compare_runs cnn,cnn_s1,cnn_s2,cnn_s3 cnn_noaug_s42,cnn_noaug_s1,cnn_noaug_s2,cnn_noaug_s3
    python -m src.compare_runs rnn,rnn_s1,rnn_s2,rnn_s3 rnn_demo_s42,rnn_demo_s1,rnn_demo_s2,rnn_demo_s3

Per run: WA mean ± sd over the 5 folds (nested thresholds), pooled WA, recall per class, AUC Present/Unknown.
Per group: mean ± sd of the WA over its seeds, the WA of the group ensemble (mean of its oof.csv), and, against
the reference, the adoption rule of the project (mean WA +0.02 and better in >= 4/5 folds, with the per-fold WA
averaged over the seeds). --json saves everything.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.datasets import PROC_DIR
from src.evaluate import cv_report, load_oof, patient_scores, result_dir


def evaluate(windows, idx, probs):
    pat = patient_scores(windows.loc[idx], probs)
    table, _, rep = cv_report(pat)
    y = pat["murmur"]
    return {"wa_folds": table["WA (tuned)"].to_numpy(), "wa_mean": float(table["WA (tuned)"].mean()),
            "wa_std": float(table["WA (tuned)"].std()), "wa_pooled": rep["weighted_accuracy"],
            "recall": {c: float(v) for c, v in rep["recall"].items()},
            "auc_present": float(roc_auc_score(y == "Present", pat["Present"])),
            "auc_unknown": float(roc_auc_score(y == "Unknown", pat["Unknown"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+", help="comma-separated runs per group; the first group is the reference")
    ap.add_argument("--json", help="save the comparison to this file")
    a = ap.parse_args()
    windows = pd.read_csv(PROC_DIR / "windows.csv")

    out, ref_folds, ref_mean = [], None, None
    for g in a.groups:
        names = [n for n in g.split(",") if n]
        runs, probs, idx0 = [], [], None
        print(f"\n=== group {names}")
        for n in names:
            idx, p = load_oof(n)
            if idx0 is None:
                idx0 = idx
            elif not np.array_equal(idx, idx0):
                raise SystemExit(f"{n}: out-of-fold windows differ from {names[0]}")
            r = evaluate(windows, idx, p)
            js = json.loads((result_dir(n) / "cv_summary.json").read_text()).get("wa_mean", np.nan)
            flag = "" if abs(js - r["wa_mean"]) < 1e-3 else f"   <-- cv_summary.json says {js:.4f}"
            print(f"  {n:22s} WA {r['wa_mean']:.3f} ± {r['wa_std']:.3f} | pooled {r['wa_pooled']:.3f} | recall P/U/A "
                  f"{[round(v, 2) for v in r['recall'].values()]} | AUC P {r['auc_present']:.3f} U {r['auc_unknown']:.3f}{flag}")
            runs.append({"run": n} | {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()})
            probs.append(p)
        seeds = np.array([r["wa_mean"] for r in runs])
        folds = np.mean([r["wa_folds"] for r in runs], axis=0)
        ens = evaluate(windows, idx0, np.mean(probs, axis=0))
        line = {"group": names, "runs": runs, "seed_mean": float(seeds.mean()),
                "seed_std": float(seeds.std(ddof=1)) if len(seeds) > 1 else 0.0,
                "folds_mean_over_seeds": folds.tolist(), "ensemble_wa": ens["wa_mean"], "ensemble_wa_std": ens["wa_std"]}
        print(f"  -> over seeds: WA {line['seed_mean']:.3f} ± {line['seed_std']:.3f} | ensemble of the group: "
              f"WA {ens['wa_mean']:.3f} ± {ens['wa_std']:.3f}")
        if ref_folds is None:
            ref_folds, ref_mean = folds, line["seed_mean"]
        else:
            gain, better = line["seed_mean"] - ref_mean, int(np.sum(folds > ref_folds))
            line["vs_reference"] = {"gain": float(gain), "folds_better": better,
                                    "adopted": bool(gain >= 0.02 and better >= 4)}
            print(f"  -> vs reference: {gain:+.3f} WA, better in {better}/5 folds -> "
                  f"{'ADOPT' if line['vs_reference']['adopted'] else 'not adopted'} (rule: +0.02 and >= 4/5 folds)")
        out.append(line)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
        print("\nsaved", a.json)


if __name__ == "__main__":
    main()
