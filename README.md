# ML II 2026/27 — Heart murmur detection (CirCor DigiScope)

Patient-level classification of `Murmur` (Present / Unknown / Absent) from phonocardiograms,
evaluated with the PhysioNet Challenge 2022 weighted accuracy.

## Setup

```bash
conda create -n ml2 python=3.12 -y && conda activate ml2
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python download_data.py       # ~590 MB -> data/circor/ (from PhysioNet's open S3 mirror)
python -m src.make_split      # -> splits/split.csv (already committed; only rerun to regenerate)
```

## Layout

| Path | Contents |
|---|---|
| `download_data.py` | Dataset downloader (resumable) |
| `src/data.py` | Load patient table, list of recordings, split |
| `src/metrics.py` | Weighted accuracy + patient-level confusion matrix (rows = predicted, cols = true) |
| `src/make_split.py` | Fixed patient-level split |
| `splits/split.csv` | `patient_id, murmur, group, fold` — fold `-1` = test, `0..4` = CV folds |
| `notebooks/` | EDA and experiments |
| `results/` | Metrics, figures, hidden-set predictions |
| `reports/` | One report per notebook/script (what, why, findings, decisions) — in Portuguese |

## Evaluation protocol

- Split by patient (children seen in both campaigns, via `Additional ID`, are kept together).
- 20% of patients held out as **test** — used only for the final reported numbers.
- 5 stratified folds on the remaining 80% for model selection, early stopping and thresholds.
