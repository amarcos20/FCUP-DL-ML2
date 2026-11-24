# ML II 2026/27 — Heart murmur detection (CirCor DigiScope)

Machine Learning II (CC3043), DCC/FCUP · group **PL3 G08**: Afonso Marcos (202404088), Pedro Afonso (202404125),
Rafael Costa (202300278).

Patient-level classification of `Murmur` (Present / Unknown / Absent) from phonocardiograms, evaluated with the
PhysioNet Challenge 2022 weighted accuracy (WA: each Present counts 5, Unknown 3, Absent 1). Always predicting
Present gives WA 0.50, the baseline to beat.

## Results

The two classifiers of the project get the same input (log-mel spectrogram of 4 s windows) and the same training
and evaluation.

| Model | Code | WA, 5-fold CV (753 patients) | WA, test fold (189 patients, evaluated once) |
|---|---|---|---|
| **CNN 2D** (4 strided conv blocks, 106 k parameters) | `src/models/cnn.py` | **0.767 ± 0.021** | 0.751 |
| **RNN** (2-layer bidirectional GRU, 116 k parameters) | `src/models/rnn.py` | **0.761 ± 0.037** | 0.789 |

Patient-level confusion matrices, per-class recall and final thresholds are in `results/{cnn,rnn}/cv_summary.json`
and in `reports/report.pdf`; the test-fold matrices are in the last cell of `notebooks/06_cnn_rnn.ipynb`.
The CNN was chosen by cross-validation, before the test fold was evaluated, and is the model of the bonus submission.

For comparison (same folds and evaluation, in `results/experiments/`): the MLP we started with, 0.648 ± 0.026;
HearHeart, winner of the Challenge, re-implemented from its official code, 0.750 ± 0.030.

## What is in this repository

| Included | Not included (gitignored) |
|---|---|
| code (`src/`), notebooks with their outputs, SLURM scripts | the CirCor dataset (`data/`): download it with `download_data.py` |
| `splits/split.csv` (the patient-level split used everywhere) | pre-processed arrays (`data/processed/`, rebuilt by `src.prepare_data`) |
| per run: `oof.csv`, `cv_summary.json`, `history_fold*.csv` | **trained models (`fold*.pt`)**: retrain them to predict (see below) |
| EDA figures, `reports/report.pdf` and the script that builds it | the hidden test set of the bonus (`data/hidden/`) |
| `submission/PL3_G08_test_results.csv` (bonus predictions) | training logs, local notes, papers we did not implement |

The cross-validation numbers (WA, confusion matrices, thresholds, comparisons) are recomputed from the committed
`oof.csv` files, so they can be checked without the models. Only the test-fold cell and the attention analysis of
the report load the fold models; their outputs are saved in the notebook and in `reports/report.pdf`.

## Setup

```bash
conda create -n ml2 python=3.12 -y && conda activate ml2
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu   # or the CUDA wheels
python download_data.py       # ~590 MB -> data/circor/ (PhysioNet's open S3 mirror)
python -m src.prepare_data    # -> data/processed/ (log-mels + MLP features, ~1-3 min with all cores)
```

`data/circor/` or the official zip extracted into `data/` both work (`src/data.py` finds either).
`python -m src.make_split` regenerates `splits/split.csv`, which is already committed.

## Training (5 CV folds; the test fold is never used)

```bash
python -m src.models.cnn      # ~10 min on one GPU -> results/cnn/  (defaults = the final model)
python -m src.models.rnn      # ~10 min on one GPU -> results/rnn/
python -m src.models.mlp      # ~20 s on CPU      -> results/experiments/mlp/
```

Each run writes `oof.csv` (out-of-fold window probabilities, indexed by the row of `data/processed/windows.csv`),
`cv_summary.json` (config, nested-CV WA, confusion matrix, final thresholds), `history_fold*.csv` and the five
fold models `fold*.pt`. The rejected iterations are reproduced with flags and a new name, e.g.
`python -m src.models.cnn --scheduler plateau --name cnn_plateau` (`--monitor wa`, and `--pool attention` for the
RNN); see `results/experiments/README.md`. On Deucalion: `scripts/deucalion_cnn.sbatch` /
`scripts/deucalion_rnn.sbatch` (arguments are passed to the module), results fetched with
`bash scripts/fetch_deucalion.sh <name>`.

## Predictions for new patients (bonus)

Needs the `fold*.pt` of the model, i.e. train it first (above).

```bash
python -m src.predict --wav-dir path/to/wavs --models cnn --out predictions.csv   # columns patient_id,murmur
python -m src.predict --wav-dir path/to/wavs --models rnn rnn_s1 rnn_s2 rnn_s3     # ensemble of runs
```

Files must be named `<patient_id>_<location>.wav` (CirCor convention). The recordings go through the same
pre-processing as in training; the 5 fold models are averaged; windows are averaged per recording and each patient
takes the maximum over its recordings; the final thresholds of `results/<model>/cv_summary.json` give the label.

Hidden test set of the bonus (100 patients, campaign CC2015, no labels), extracted to
`data/hidden/CirCor_test_students/`. Its README asks for the template `PLX_GYY_test_results.csv` (columns
`ID,Murmur`, all 100 rows), which takes precedence over the `patient_id,murmur` example of the project PDF:

```bash
D=data/hidden/CirCor_test_students
python -m src.predict --wav-dir $D/test_data --template $D/PLX_GYY_test_results.csv \
    --out submission/PL3_G08_test_results.csv --probs
```

`--template` keeps the template's row order and stops if a patient is missing; `--probs` also writes the patient
scores (`P(Present)`, `P(Unknown)`) to `submission/PL3_G08_test_results_scores.csv`, for checking only.

## Final experiments (November 2026)

Four questions, all on the same 5 CV folds and the same 4 seeds (42, 1, 2, 3) as the reference runs
`cnn, cnn_s1..s3` / `rnn, rnn_s1..s3`; every number is recomputed from `oof.csv` by `src/compare_runs.py`.

| Question | How | Runs |
|---|---|---|
| Does data augmentation help? | `--no-spec-augment --no-random-crop` | `{cnn,rnn}_noaug_s{42,1,2,3}` |
| Demographics from the start? | wide branch, `--demo age height` (`src/demographics.py`) | `{cnn,rnn}_demo_s{42,1,2,3}` |
| Demographics on top of the model we have? | stacking, `src/stacking.py` (no retraining) | `stack_{cnn,rnn}/` |
| Final model of the hidden set | best configuration retrained on all labelled patients, `--include-test` | `{cnn,rnn}_final_s{42,1,2,3}` |

The demographic fields were chosen in `notebooks/01_eda.ipynb` §10 (available and non-constant in the hidden set,
not derived from the label, associated with `Murmur`, not redundant): **age group + height**. A demographic option
is adopted only with the rule of the project: mean WA +0.02 over the same model without it, better in ≥ 4/5 folds.

On Deucalion (login node, from the repo; `ACC` is the GPU account used with `sbatch -A`):

```bash
export PROJ=/projects/F202500007HPCVLABUPORTO/amarcos.up ACC=<account_g>
cd $PROJ/FCUP-DL-ML2 && git pull
# 1. quick check of the new options (1 fold, 2 epochs, dev partition)
sbatch -A $ACC -p dev-a100-40 -t 00:30:00 scripts/deucalion_cnn.sbatch --epochs 2 --folds 0 --demo age height --name smoke_cnn_demo
sbatch -A $ACC -p dev-a100-40 -t 00:30:00 scripts/deucalion_rnn.sbatch --epochs 2 --folds 0 --demo age height --include-test --name smoke_rnn_final
# 2. all experiments: 16 trainings + 2 stacking jobs + 2 comparisons (after the trainings)
bash scripts/submit_experiments.sh            # or one block: noaug | demo | stack
# 3. read logs/compare-cnn-*.out, logs/compare-rnn-*.out, logs/stack-*.out (or fetch results/ and logs/)
# 4. final model + hidden-set prediction with the chosen configuration, e.g.
bash scripts/submit_final.sh rnn                                  # audio only
bash scripts/submit_final.sh rnn --demo age height                # demographics from the start
STACK=meta_age_height bash scripts/submit_final.sh rnn            # + stacking refitted on the final runs
```

`scripts/submit_final.sh` trains 4 seeds with `--include-test` (the 189 test patients join the training set of
every fold; validation, early stopping and thresholds stay on the CV folds), then `src.predict` averages the
20 fold models, applies the thresholds tuned on their averaged out-of-fold predictions and writes
`submission/PL3_G08_test_results.csv` (+ `_scores.csv`, `_info.json`). The hidden set must be in
`data/hidden/CirCor_test_students/` (`data/CirCor_test_students/` also works, or set `HIDDEN=...`). These runs cannot be
evaluated on the test fold any more; the test-fold results of the report stay those of `cnn` / `rnn`.

## Evaluation protocol

- Split by patient, stratified by class; the 70 children recorded in both campaigns (`Additional ID`) stay together.
- 20% of the patients (189) form the **test fold**, evaluated once at the end with the frozen models
  (git tag `modelo-final`, `RUN_TEST` cell of `notebooks/06_cnn_rnn.ipynb`).
- The other 753 patients form 5 CV folds; we report the mean ± std of the WA over the folds.
- Normalisation statistics come from the training folds only; the decision thresholds of each fold are tuned on the
  other 4 folds (nested). Early stopping does use the validation fold, so the CV WA is slightly optimistic; the test
  fold shows the effect is small.
- From windows to patients: mean over the windows of a recording, maximum over the recordings of a patient, then
  Present if P(Present) ≥ t1, else Unknown if P(Unknown) ≥ t2, else Absent.

## Report

```bash
conda install -c conda-forge tectonic   # LaTeX engine (once); latexmk -xelatex also works
python reports/make_report.py           # -> reports/report.pdf
```

A short report in Portuguese (data, pre-processing, the two models, iterations, results, comparison with the
literature, conclusions, likely questions); every number is recomputed from `results/`.

## Layout

```
src/
├── data.py, make_split.py         patients, recordings, labels; patient-level split -> splits/split.csv
├── features.py, prepare_data.py   pre-processing (band-pass, 4 s windows, log-mel, MLP features) -> data/processed/
├── datasets.py                    fold masks, train-only normalisation, torch dataset, patient aggregation
├── evaluate.py, metrics.py        patient scores, decision rule, nested thresholds, weighted accuracy, result_dir
├── predict.py                     predictions for new .wav files (bonus CSV)
└── models/
    ├── cnn.py, rnn.py             THE TWO FINAL CLASSIFIERS (training loop shared in cnn.py, 5-fold CV)
    ├── mlp.py                     first approach (experiment)
    └── hearheart.py, prepare_hearheart.py   Challenge 2022 winner, re-implemented (comparison only)
```

| Path | Contents |
|---|---|
| `notebooks/` | `01_eda`, `02_preprocessing`, `03_mlp`, `04_cnn`, `05_hearheart` (Challenge winner vs our CNN), `06_cnn_rnn` (the two classifiers, iterations, comparison of all models, test cell) |
| `results/cnn/`, `results/rnn/` | **final models**: `oof.csv`, `cv_summary.json`, `history_fold*.csv` |
| `results/experiments/` | MLP, rejected iterations and HearHeart (see its README) |
| `results/eda/`, `results/preprocessing/` | figures of notebooks 01 and 02 |
| `submission/` | predictions on the hidden data, as submitted |
| `reports/` | `make_report.py` → `report.pdf` |
| `scripts/` | SLURM jobs for Deucalion and `fetch_deucalion.sh` |
| `splits/split.csv` | `patient_id, murmur, group, fold`: fold `-1` = test, `0..4` = CV folds |
| `literature/` | dataset and Challenge papers, the re-implemented HearHeart paper, `README.md` with the reviewed literature |
| `AC2_Project_20262027.pdf` | project statement |
