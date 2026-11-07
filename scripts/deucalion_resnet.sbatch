#!/bin/bash
# Train the ResNet50 comparison models (src.resnet, 5-fold CV) on one A100 of Deucalion.
#
#   sbatch -A <account_g> scripts/deucalion_resnet.sbatch --task multiclass            # step 1 -> results/resnet50
#   sbatch -A <account_g> scripts/deucalion_resnet.sbatch --task dual                  # step 2 -> results/dual_resnet50
#   sbatch -A <account_g> scripts/deucalion_resnet.sbatch --task dual --pretrained --lr 3e-5 --name dual_resnet50_imagenet
#   sbatch -A <account_g> -p dev-a100-40 -t 00:30:00 scripts/deucalion_resnet.sbatch --task dual --epochs 1 --folds 0 --name smoke
#
# Arguments are passed to `python -m src.resnet`. The dual task trains 2 networks per fold.
# --pretrained downloads the ImageNet weights; if the compute nodes have no internet, cache them
# once on the login node first:
#   python -c "from src.resnet import ResNet50, load_imagenet; load_imagenet(ResNet50())"
# Expects the repo in $PROJ/FCUP-DL-ML2 with data/processed/ and the venv in $PROJ/venv_ml2_x86.
#SBATCH --job-name=ml2-resnet
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
PROJ=${PROJ:?set PROJ to your project folder, e.g. export PROJ=/projects/<project-ID>/$USER}
cd "$PROJ/FCUP-DL-ML2"
source "$PROJ/venv_ml2_x86/bin/activate"

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

python -m src.resnet --workers 16 --in-memory "$@"
