#!/bin/bash
# Run any python module of the repo on one A100 of Deucalion (short jobs: stacking, comparisons, predictions).
#
#   sbatch -A <account_g> scripts/deucalion_py.sbatch src.stacking --runs rnn rnn_s1 rnn_s2 rnn_s3 --name stack_rnn
#   sbatch -A <account_g> scripts/deucalion_py.sbatch src.compare_runs cnn,cnn_s1,cnn_s2,cnn_s3 cnn_noaug_s42,cnn_noaug_s1,cnn_noaug_s2,cnn_noaug_s3
#   sbatch -A <account_g> scripts/deucalion_py.sbatch src.predict --wav-dir data/hidden/CirCor_test_students/test_data --models rnn ...
#
# The first argument is the module (python -m <module>), the others are passed to it.
# Expects the repo in $PROJ/FCUP-DL-ML2 with data/processed/ and the venv in $PROJ/venv_ml2_x86.
#SBATCH --job-name=ml2-py
#SBATCH --partition=dev-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
PROJ=${PROJ:?set PROJ to your project folder, e.g. export PROJ=/projects/<project-ID>/$USER}
cd "$PROJ/FCUP-DL-ML2"
source "$PROJ/venv_ml2_x86/bin/activate"

MODULE=$1; shift
python -m "$MODULE" "$@"
