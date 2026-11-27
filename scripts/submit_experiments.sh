#!/bin/bash
# Final experiments of the CNN and the RNN on the 5 CV folds (test fold not used), seeds 42, 1, 2, 3 as the
# reference runs cnn, cnn_s1..s3 / rnn, rnn_s1..s3. Run on the Deucalion login node, from the repo folder:
#
#   export PROJ=/projects/F202500007HPCVLABUPORTO/amarcos.up ACC=<account_g>
#   bash scripts/submit_experiments.sh            # everything
#   bash scripts/submit_experiments.sh noaug      # only one block: noaug | demo | stack
#
# Blocks:
#   noaug  data-augmentation ablation: --no-spec-augment --no-random-crop     -> results/{cnn,rnn}_noaug_s<seed>
#   demo   demographics from the start (wide branch): --demo $DEMO           -> results/{cnn,rnn}_demo_s<seed>
#   stack  stacking on the existing runs (no training):                       -> results/stack_{cnn,rnn}/
# With "all", one comparison job per model runs after the trainings -> results/comparison_{cnn,rnn}.json.
# DEMO defaults to the features chosen in notebooks/01_eda.ipynb (age height).
set -euo pipefail
cd "$(dirname "$0")/.."
ACC=${ACC:?set ACC to your GPU account (the one you pass to sbatch -A)}
DEMO=${DEMO:-"age height"}
SEEDS=(42 1 2 3)
WHAT=${1:-all}
mkdir -p logs

sub() { sbatch --parsable -A "$ACC" "$@"; }
JOBS=()

if [[ $WHAT == all || $WHAT == noaug ]]; then
    for s in "${SEEDS[@]}"; do
        JOBS+=("$(sub -t 01:00:00 --job-name=cnn-noaug-s$s scripts/deucalion_cnn.sbatch --epochs 50 --patience 8 \
                   --no-spec-augment --no-random-crop --seed "$s" --name "cnn_noaug_s$s")")
        JOBS+=("$(sub -t 02:00:00 --job-name=rnn-noaug-s$s scripts/deucalion_rnn.sbatch --epochs 50 --patience 8 \
                   --no-spec-augment --no-random-crop --seed "$s" --name "rnn_noaug_s$s")")
    done
fi

if [[ $WHAT == all || $WHAT == demo ]]; then
    for s in "${SEEDS[@]}"; do
        JOBS+=("$(sub -t 01:00:00 --job-name=cnn-demo-s$s scripts/deucalion_cnn.sbatch --epochs 50 --patience 8 \
                   --demo $DEMO --seed "$s" --name "cnn_demo_s$s")")
        JOBS+=("$(sub -t 02:00:00 --job-name=rnn-demo-s$s scripts/deucalion_rnn.sbatch --epochs 50 --patience 8 \
                   --demo $DEMO --seed "$s" --name "rnn_demo_s$s")")
    done
fi

if [[ $WHAT == all || $WHAT == stack ]]; then
    sub --job-name=stack-cnn scripts/deucalion_py.sbatch src.stacking --runs cnn cnn_s1 cnn_s2 cnn_s3 --name stack_cnn
    sub --job-name=stack-rnn scripts/deucalion_py.sbatch src.stacking --runs rnn rnn_s1 rnn_s2 rnn_s3 --name stack_rnn
fi

if [[ $WHAT == all ]]; then
    DEP=$(IFS=:; echo "${JOBS[*]}")
    for m in cnn rnn; do      # one comparison per architecture: the reference is its runs with augmentation
        sub --dependency=afterok:$DEP --job-name=compare-$m scripts/deucalion_py.sbatch src.compare_runs \
            $m,${m}_s1,${m}_s2,${m}_s3 \
            ${m}_noaug_s42,${m}_noaug_s1,${m}_noaug_s2,${m}_noaug_s3 \
            ${m}_demo_s42,${m}_demo_s1,${m}_demo_s2,${m}_demo_s3 --json results/comparison_$m.json
    done
fi
echo "submitted ${#JOBS[@]} training jobs (+ stacking / comparison). squeue -u \$USER to follow them."
