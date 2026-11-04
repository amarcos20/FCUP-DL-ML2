#!/bin/bash
# Bring training outputs back from Deucalion: results/<model>/ (oof.csv, cv_summary.json,
# history_fold*.csv, fold*.pt models) and the SLURM logs.
#
#   bash scripts/fetch_deucalion.sh            # all results + logs
#   bash scripts/fetch_deucalion.sh cnn        # only results/cnn/ + logs
set -euo pipefail
REMOTE=deucalion:/projects/F202500007HPCVLABUPORTO/amarcos.up/FCUP-DL-ML2
cd "$(dirname "$0")/.."

SRC=results/${1:-}
mkdir -p "$SRC" logs
rsync -av --info=progress2 "$REMOTE/$SRC/" "$SRC/"
rsync -av "$REMOTE/logs/" logs/

for f in results/*/cv_summary.json; do
    python3 -c "import json,sys; s=json.load(open('$f')); print(f\"{'$f'.split('/')[1]:>10}: WA {s['wa_mean']:.3f} ± {s['wa_std']:.3f} | recall\", {k: round(v, 3) for k, v in s['recall'].items()})"
done
