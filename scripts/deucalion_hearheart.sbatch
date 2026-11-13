#!/bin/bash
# Train the HearHeart CNN (Challenge 2022 winner, src.models.hearheart, 5-fold CV) on one A100 of Deucalion.
#
#   sbatch -A <account_g> -t 02:00:00 scripts/deucalion_hearheart.sbatch --mode arch     # B -> results/hearheart_arch
#   sbatch -A <account_g> -t 04:00:00 scripts/deucalion_hearheart.sbatch --mode recipe   # A -> results/hearheart
#   sbatch -A <account_g> -p dev-a100-40 -t 00:30:00 scripts/deucalion_hearheart.sbatch --mode recipe --epochs 2 --folds 0 --name smoke_hh
#
# --mode recipe needs data/processed/waves_int16.npy + waves.csv (python -m src.models.prepare_hearheart, or copy
# them from a machine that has them). Arguments are passed to `python -m src.models.hearheart`.
# Expects the repo in $PROJ/FCUP-DL-ML2 with data/processed/ and the venv in $PROJ/venv_ml2_x86.
#SBATCH --job-name=ml2-hearheart
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

python -m src.models.hearheart --workers 16 --in-memory "$@"
