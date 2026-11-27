"""Predict the murmur label of new patients (e.g. the hidden set) from their .wav files.

Same pipeline as in training: load at 4 kHz, band-pass 25-1000 Hz, z-score -> log-mel (40 bands, 10 ms) ->
4 s windows with 2 s hop -> each fold model standardises with its own training statistics (and encodes the
demographics, for --demo models) -> softmax per window, averaged over fold models and over runs -> mean per
recording, max per patient -> Present if P(Present) >= t1, else Unknown if P(Unknown) >= t2, else Absent.
Thresholds: those of the run (one run) or tuned on the averaged out-of-fold predictions (several runs).
--stack applies a meta-model of src.stacking with its own thresholds.

File names follow the CirCor convention <patient_id>_<location>[_<n>].wav; sub-folders are searched too.

    python -m src.predict --wav-dir path/to/wavs --models rnn rnn_s1 rnn_s2 rnn_s3

Hidden set (submission in the format of the template PLX_GYY_test_results.csv, columns ID,Murmur):
    D=data/hidden/CirCor_test_students
    python -m src.predict --wav-dir $D/test_data --models rnn_final_s42 rnn_final_s1 rnn_final_s2 rnn_final_s3 \\
        --demographics $D/test_data_participants.csv --template $D/PLX_GYY_test_results.csv \\
        --out submission/PL3_G08_test_results.csv --probs
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import demographics
from src.models.cnn import device
from src.data import CLASSES
from src.datasets import PROC_DIR, aggregate_to_patient
from src.evaluate import load_oof, patient_scores, predict_rule, result_dir, tune_thresholds
from src.features import AUDIO, MEL, frames_per_window, load_audio, logmel, window_starts


def find_wavs(wav_dir):
    files = sorted(Path(wav_dir).rglob("*.wav"))
    if not files:
        raise SystemExit(f"no .wav files found in {wav_dir}")
    rows = [{"path": p, "recording": p.stem, "patient_id": p.stem.split("_")[0]} for p in files]
    return pd.DataFrame(rows)


def recording_windows(path):
    """(n_windows, 40, 401) log-mel windows of one recording, cut as in training (src.prepare_data +
    src.datasets.mel_window): window start in frames = round(start_sample / hop); past the end of the
    recording the window is padded with its minimum dB value."""
    y = load_audio(path)
    mel = logmel(y)
    n = frames_per_window()
    out = []
    for s in window_starts(len(y)):
        f0 = min(int(round(s / MEL.hop_length)), mel.shape[1] - 1)
        x = mel[:, f0:f0 + n]
        if x.shape[1] < n:
            x = np.pad(x, ((0, 0), (0, n - x.shape[1])), constant_values=float(x.min()))
        out.append(x)
    return np.stack(out).astype(np.float16).astype(np.float32)   # mel.npy is stored as float16


# ------------------------------------------------------------------------------------------ models

def load_model_runs(name):
    """Fold models of results/<name> (CNN or RNN, detected from the saved config) and its cv_summary."""
    folder = result_dir(name)
    pts = sorted(folder.glob("fold*.pt"))
    if not pts:
        raise SystemExit(f"no fold models (fold*.pt) in {folder}")
    cfg = torch.load(pts[0], map_location="cpu", weights_only=False)["cfg"]
    if "mode" in cfg:
        raise SystemExit(f"{name} is a HearHeart run: not supported here (use the CNN or the RNN)")
    if "cell" in cfg:
        from src.models.rnn import load_runs
        runs, _ = load_runs(folder)
    else:
        from src.models.cnn import load_runs
        runs = load_runs(folder)
    summary = json.loads((folder / "cv_summary.json").read_text())
    return runs, summary


def ensemble_thresholds(names, summaries):
    """Final thresholds: those of the run itself (one run), or tuned on the averaged out-of-fold predictions."""
    if len(names) == 1:
        return tuple(summaries[0]["thresholds"])
    windows = pd.read_csv(PROC_DIR / "windows.csv")
    idx0, probs = None, []
    for n in names:
        idx, p = load_oof(n)
        if idx0 is None:
            idx0 = idx
        elif not np.array_equal(idx, idx0):
            raise SystemExit(f"{n}: out-of-fold windows differ from {names[0]}: cannot tune ensemble thresholds")
        probs.append(p)
    t, wa = tune_thresholds(patient_scores(windows.loc[idx0], np.mean(probs, axis=0)))
    print(f"ensemble thresholds tuned on the averaged OOF of {len(names)} runs: {t} (WA on all CV folds {wa:.3f}, "
          f"optimistic: not nested)", flush=True)
    return t


def window_probs(runs, X, demo_row=None, batch=256):
    """Mean over the fold models of the softmax probabilities of windows X (n, 40, 401).
    demo_row: the patient's row of the demographic table (needed by models trained with --demo)."""
    dev = device()
    out = []
    for r in runs:
        mean, std = r["band_stats"]
        Xs = torch.from_numpy(((X - mean) / std)[:, None].astype(np.float32))
        Z = None
        if r.get("demo_encoder"):
            if demo_row is None:
                raise SystemExit("a model was trained with --demo: pass --demographics")
            enc = demographics.DemoEncoder.from_state(r["demo_encoder"])
            Z = torch.from_numpy(np.repeat(enc.transform(demo_row), len(X), axis=0))
        m = r["model"].to(dev).eval()
        with torch.no_grad():
            p = []
            for i in range(0, len(Xs), batch):
                x = Xs[i:i + batch].to(dev)
                logits = m(x) if Z is None else m(x, Z[i:i + batch].to(dev))
                p.append(torch.softmax(logits, 1).cpu().numpy())
        out.append(np.concatenate(p))
    return np.mean(out, axis=0)


def predict(wav_dir, names=("cnn",), table=None, stack=None):
    """Returns (patient scores DataFrame, labels Series, info dict) for every patient in wav_dir."""
    recs = find_wavs(wav_dir)
    loaded = [load_model_runs(n) for n in names]
    needs_demo = bool(stack and stack.get("encoder")) or any(r.get("demo_encoder") for runs, _ in loaded for r in runs)
    if needs_demo and table is None:
        raise SystemExit("these models / the stack use demographics: pass --demographics <participants csv>")
    if needs_demo:
        missing = sorted(set(recs.patient_id) - set(table.index))
        if missing:
            raise SystemExit(f"no demographics for patients {missing[:10]}")

    win_rows, probs = [], []
    for r in tqdm(recs.itertuples(), total=len(recs), desc="recordings"):
        X = recording_windows(r.path)
        demo_row = table.loc[[r.patient_id]] if needs_demo else None
        probs.append(np.mean([window_probs(runs, X, demo_row) for runs, _ in loaded], axis=0))
        win_rows += [{"recording": r.recording, "patient_id": r.patient_id}] * len(X)
    pat = aggregate_to_patient(pd.DataFrame(win_rows), np.concatenate(probs), how="max")

    if stack is None:
        t1, t2 = ensemble_thresholds(list(names), [s for _, s in loaded])
        scores = pat
    else:
        from src.stacking import apply_stack
        if list(stack["base_runs"]) != list(names):
            print(f"WARNING: the stack was fitted on {stack['base_runs']}, predicting with {list(names)}", flush=True)
        scores = apply_stack(stack, pat, table)
        t1, t2 = stack["thresholds"]
    labels = pd.Series(predict_rule(scores, t1, t2), index=scores.index, name="murmur")
    info = {"runs": list(names), "thresholds": [t1, t2], "stack": stack.get("variant") if stack else None}
    return scores, labels, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True, help="folder with the .wav files (searched recursively)")
    ap.add_argument("--models", "--model", dest="models", nargs="+", default=["cnn"],
                    help="one or more runs in results/ (CNN or RNN fold models); several runs are averaged")
    ap.add_argument("--demographics", help="patient table with Age, Sex, Height, Weight, ... "
                    "(e.g. data/hidden/CirCor_test_students/test_data_participants.csv); needed with --demo models / --stack")
    ap.add_argument("--stack", help="stacked logistic regression saved by src.stacking (stack_<variant>.json)")
    ap.add_argument("--out", help="output CSV (default predictions_<runs>.csv)")
    ap.add_argument("--probs", action="store_true", help="also save the patient scores")
    ap.add_argument("--template", help="submission template (columns ID,Murmur): output in its format and order")
    a = ap.parse_args()
    out = Path(a.out or f"predictions_{'+'.join(a.models)}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"runs: {a.models} | device: {device()} | sample rate {AUDIO.sr} Hz", flush=True)

    table = demographics.read_table(a.demographics) if a.demographics else None
    stack = json.loads(Path(a.stack).read_text()) if a.stack else None
    scores, labels, info = predict(a.wav_dir, a.models, table, stack)
    if a.template:
        sub = pd.read_csv(a.template, dtype={"ID": str})[["ID"]]
        sub["Murmur"] = sub["ID"].map(labels)
        missing = sub.loc[sub["Murmur"].isna(), "ID"].tolist()
        if missing:
            raise SystemExit(f"{len(missing)} patients of the template have no recordings: {missing}")
        sub.to_csv(out, index=False)
    else:
        labels.rename_axis("patient_id").reset_index().to_csv(out, index=False)
    print(f"{len(labels)} patients -> {out} | thresholds {info['thresholds']}")
    print(labels.value_counts().reindex(CLASSES, fill_value=0).to_string())
    out.with_name(out.stem + "_info.json").write_text(json.dumps(info, indent=2))
    if a.probs:
        p = out.with_name(out.stem + "_scores.csv")
        scores[CLASSES].rename_axis("patient_id").to_csv(p, float_format="%.5f")
        print("scores ->", p)


if __name__ == "__main__":
    main()
