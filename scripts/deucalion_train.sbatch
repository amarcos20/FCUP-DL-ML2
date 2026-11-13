#!/bin/bash
# Train the MLP and the CNN (5-fold CV) on one A100 of Deucalion.
#
#   sbatch -A <account_g> scripts/deucalion_train.sbatch                  # defaults
#   sbatch -A <account_g> -p dev-a100-40 -t 00:30:00 scripts/deucalion_train.sbatch --epochs 2   # quick test
#
# Extra arguments are passed to `python -m src.models.cnn` (e.g. --epochs 50 --patience 8 --name cnn_v2).
# Expects the repo in $PROJ/FCUP-DL-ML2 with data/processed/ and the venv in $PROJ/venv_ml2_x86.
#SBATCH --job-name=ml2-train
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
PROJ=${PROJ:?set PROJ to your project folder, e.g. export PROJ=/projects/<project-ID>/$USER}
cd "$PROJ/FCUP-DL-ML2"
source "$PROJ/venv_ml2_x86/bin/activate"

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

python -m src.models.mlp
python -m src.models.cnn --workers 16 --in-memory "$@"
