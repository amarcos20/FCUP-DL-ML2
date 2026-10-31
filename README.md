# ML II 2026/27 — Heart murmur detection (CirCor DigiScope)

Patient-level classification of `Murmur` (Present / Unknown / Absent) from phonocardiograms,
evaluated with the PhysioNet Challenge 2022 weighted accuracy.

## Setup

```bash
conda create -n ml2 python=3.12 -y && conda activate ml2
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python download_data.py       # ~590 MB -> data/circor/ (from PhysioNet's open S3 mirror)
python -m src.make_split      # -> splits/split.csv (already committed; only rerun to regenerate)
python -m src.prepare_data    # -> data/processed/ (MLP features + log-mels, ~1-3 min with all cores)
```

The dataset can be either in `data/circor/` (from `download_data.py`) or in
`data/the-circor-digiscope-phonocardiogram-dataset-1.0.3/` (official zip extracted into `data/`);
`src/data.py` finds whichever exists.

## Layout

| Path | Contents |
|---|---|
| `download_data.py` | Dataset downloader (resumable) |
| `src/data.py` | Load patient table, recordings (+ recording-level label from `Murmur locations`), segmentations, split |
| `src/features.py` | Pre-processing config + feature extraction (band-pass, 4 s windows, MLP features, log-mel) |
| `src/prepare_data.py` | Runs the pre-processing over all recordings -> `data/processed/` |
| `src/datasets.py` | Load processed data, fold masks, train-only normalisation, torch `CNNWindows`, patient aggregation |
| `src/metrics.py` | Weighted accuracy + patient-level confusion matrix (rows = predicted, cols = true) |
| `src/make_split.py` | Fixed patient-level split |
| `splits/split.csv` | `patient_id, murmur, group, fold` — fold `-1` = test, `0..4` = CV folds |
| `notebooks/` | `01_eda.ipynb` exploration (+ conclusions for pre-processing), `02_preprocessing.ipynb` pre-processing for MLP/CNN |
| `results/` | Metrics, figures, hidden-set predictions |

## Evaluation protocol

- Split by patient (children seen in both campaigns, via `Additional ID`, are kept together).
- 20% of patients held out as **test** — used only for the final reported numbers.
- 5 stratified folds on the remaining 80% for model selection, early stopping and thresholds.

## Pre-processing (summary, details in `notebooks/02_preprocessing.ipynb`)

- Native 4 kHz, band-pass 25-1000 Hz, z-score per recording.
- Each recording is cut into 4 s windows (hop 2 s); every window inherits the recording label
  (`rec_label`: Present only at the locations listed in `Murmur locations`).
- MLP input: 99 features per window (MFCC stats, spectral descriptors, band energies, envelope periodicity).
- CNN input: log-mel 40 x 401 per window (20-1000 Hz, 10 ms hop), sliced on the fly from `mel.npy`.
- Normalisation statistics are fitted on the training folds only (`src.datasets`).
- Window predictions -> recording -> patient with `aggregate_to_patient`.
