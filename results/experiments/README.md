# Experiments (not the final models)

The two final classifiers are in `results/cnn/` and `results/rnn/`. Everything here was trained with the
same folds and evaluated with the same protocol (`src.evaluate`), and is only used for comparison in
`notebooks/06_cnn_rnn.ipynb` and `reports/report.pdf`.

| Folder | What | WA (5-fold CV, CNN / RNN) | Outcome |
|---|---|---|---|
| `mlp` | First approach: MLP on 99 summary features per 4 s window | 0.648 | replaced by the RNN |
| `cnn_plateau`, `rnn_plateau` | + ReduceLROnPlateau | 0.756 / 0.766 | rejected |
| `cnn_wa`, `rnn_wa` | + early stopping on validation WA instead of loss | 0.752 / 0.753 | rejected |
| `cnn_plateau_wa`, `rnn_plateau_wa` | both changes | 0.773 / 0.750 | rejected |
| `rnn_att` | RNN with attention pooling instead of mean | 0.762 | rejected |
| `hearheart_arch` | Challenge 2022 winner's network with our input and training | 0.715 | literature comparison |
| `hearheart` | Challenge 2022 winner's full recipe, re-implemented | 0.750 | literature comparison |

Final models for reference: CNN 0.767, RNN 0.761. **Adoption rule, fixed before seeing the results**
(`MIN_GAIN` in `reports/make_report.py`): a change is kept only if it raises the mean WA by more than 0.02
(about one fold standard deviation) and wins in at least 4 of the 5 folds; otherwise the simpler version stays.
No iteration passed it (the best, `cnn_plateau_wa`, gains +0.006).

`src.evaluate.result_dir(name)` resolves a run name to `results/<name>` or `results/experiments/<name>`,
so code and notebooks refer to runs by name only. New runs are written to `results/<name>`.
