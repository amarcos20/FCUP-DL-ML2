# ML II 2026/27 — Heart murmur detection (CirCor DigiScope)

Patient-level classification of `Murmur` (Present / Unknown / Absent) from phonocardiograms,
evaluated with the PhysioNet Challenge 2022 weighted accuracy (WA: Present ×5, Unknown ×3, Absent ×1).

**The two classifiers of the project** (same input, training and evaluation):

| Model | Input | WA (5-fold CV) | Code |
|---|---|---|---|
| **CNN 2D** | log-mel 40 × 401 per 4 s window | 0.767 ± 0.021 | `src/cnn.py` |
| **RNN** (bidirectional GRU) | the same window read as a sequence of 401 frames | 0.761 ± 0.037 | `src/rnn.py` |

The MLP (`src/mlp.py`, WA 0.648) was the first approach. HearHeart (`src/hearheart.py`, winner of the
Challenge) is a re-implementation from the literature used only for comparison. Everything is summarised in **`reports/report.pdf`**.

## Setup

```bash
conda create -n ml2 python=3.12 -y && conda activate ml2
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python download_data.py       # ~590 MB -> data/circor/ (from PhysioNet's open S3 mirror)
python -m src.make_split      # -> splits/split.csv (already committed; only rerun to regenerate)
python -m src.prepare_data    # -> data/processed/ (MLP features + log-mels, ~1-3 min with all cores)
```

The dataset can be either in `data/circor/` or in `data/the-circor-digiscope-phonocardiogram-dataset-1.0.3/`
(official zip extracted into `data/`); `src/data.py` finds whichever exists.

## Training (5 CV folds; the test fold is never touched)

```bash
python -m src.cnn --epochs 50 --patience 8 --name cnn   # ~10 min on one GPU  -> results/cnn/
python -m src.rnn --name rnn                            # ~10 min on one GPU  -> results/rnn/
python -m src.mlp                                       # ~20 s on CPU        -> results/mlp/
```

Iterations (all rejected, see the report): `--scheduler plateau`, `--monitor wa` (CNN and RNN),
`--pool attention` (RNN). On Deucalion: `scripts/deucalion_cnn.sbatch` and `scripts/deucalion_rnn.sbatch`
(arguments are passed to the module), results brought back with `bash scripts/fetch_deucalion.sh <name>`.

Each run writes `oof.csv` (out-of-fold window probabilities, indexed by the row of `windows.csv`),
`cv_summary.json` (config, nested-CV WA, confusion matrix, final thresholds), `history_fold*.csv`
and `fold*.pt` (fold models, not committed).

## Predictions for new patients (hidden data of the bonus)

```bash
python -m src.predict --wav-dir path/to/hidden_wavs --model cnn --out predictions.csv
```

Files must be named `<patient_id>_<location>.wav` (CirCor convention). The recordings go through the
same pre-processing as in training, the 5 fold models are averaged, recordings are aggregated per patient
(maximum over locations) and the final thresholds of `results/<model>/cv_summary.json` are applied.
Output: `patient_id,murmur`, one row per patient. Needs the `fold*.pt` of the chosen model.

## Report

```bash
conda install -c conda-forge tectonic   # LaTeX engine (once); latexmk -xelatex also works
python reports/make_report.py           # -> reports/report.pdf
```

A short report in Portuguese with only the essentials (data, pre-processing, the two models, iterations,
results, comparison with the literature, conclusions, likely questions); every number is recomputed
from `results/`.

## Layout

| Path | Contents |
|---|---|
| `src/data.py` | Patient table, recordings (+ recording-level label from `Murmur locations`), segmentations, split |
| `src/features.py` | Pre-processing config + feature extraction (band-pass, 4 s windows, MLP features, log-mel) |
| `src/prepare_data.py` | Runs the pre-processing over all recordings -> `data/processed/` |
| `src/datasets.py` | Processed data, fold masks, train-only normalisation, torch `CNNWindows`, patient aggregation |
| `src/evaluate.py` / `src/metrics.py` | Patient scores, hierarchical rule, nested threshold tuning / weighted accuracy |
| `src/cnn.py`, `src/rnn.py` | **The two classifiers**: models, training (shared loop in `cnn.py`), 5-fold CV |
| `src/predict.py` | Predictions for new `.wav` files (bonus CSV) |
| `src/mlp.py` | First approach (MLP on 99 features per window) |
| `src/hearheart.py`, `src/prepare_hearheart.py` | Literature model (comparison only) |
| `notebooks/` | `01_eda`, `02_preprocessing`, `03_mlp`, `04_cnn`, `05_hearheart` (Challenge winner vs our CNN), `06_cnn_rnn` (the two classifiers, iterations, comparison with all models, test cell) |
| `results/<model>/` | `oof.csv`, `cv_summary.json`, `history_fold*.csv` of every run |
| `reports/` | `make_report.py` → `report.pdf` |
| `scripts/` | SLURM jobs for Deucalion and `fetch_deucalion.sh` |
| `splits/split.csv` | `patient_id, murmur, group, fold` — fold `-1` = test, `0..4` = CV folds |
| `literature/` | Paper of the re-implemented model + `README.md` with the reviewed literature |

## Evaluation protocol

- Split by patient (children seen in both campaigns, via `Additional ID`, are kept together), stratified.
- 20% of patients held out as **test** — evaluated once, at the end (`RUN_TEST` cell of `06_cnn_rnn.ipynb`).
- 5 folds on the remaining 80% for early stopping, thresholds and model choice; the thresholds of each
  fold are tuned on the other 4 folds (nested), so the CV WA is not optimistic.
