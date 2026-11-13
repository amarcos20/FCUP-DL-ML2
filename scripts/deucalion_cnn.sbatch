#!/bin/bash
# Train only the CNN (src.models.cnn, 5-fold CV) on one A100 of Deucalion -- unlike deucalion_train.sbatch, which
# also retrains the MLP into results/mlp.
#
#   sbatch -A <account_g> -t 01:00:00 scripts/deucalion_cnn.sbatch --epochs 50 --patience 8 --scheduler plateau --name cnn_plateau
#
# Always pass --epochs 50 --patience 8 (the settings of the reference run results/cnn) and a new --name.
# Arguments are passed to `python -m src.models.cnn`.
# Expects the repo in $PROJ/FCUP-DL-ML2 with data/processed/ and the venv in $PROJ/venv_ml2_x86.
#SBATCH --job-name=ml2-cnn
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
PROJ=${PROJ:?set PROJ to your project folder, e.g. export PROJ=/projects/<project-ID>/$USER}
cd "$PROJ/FCUP-DL-ML2"
source "$PROJ/venv_ml2_x86/bin/activate"

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

python -m src.models.cnn --workers 16 --in-memory "$@"
