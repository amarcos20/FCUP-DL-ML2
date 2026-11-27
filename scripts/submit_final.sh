#!/bin/bash
# Final model of the hidden set: 4 seeds trained with --include-test (CV folds + test fold -1), then the
# prediction of the hidden set. Run on the Deucalion login node, from the repo folder:
#
#   export PROJ=/projects/F202500007HPCVLABUPORTO/amarcos.up ACC=<account_g>
#   bash scripts/submit_final.sh rnn                       # the submitted model (RNN, audio only, augmentation on)
#   bash scripts/submit_final.sh rnn --demo age height     # other configurations, e.g. with demographics
#   STACK=meta_age bash scripts/submit_final.sh rnn        # + stacking refitted on the final runs
#
# Everything after the model name is passed to training (src.models.<model>). Outputs:
#   results/<model>_final_s<seed>/                     fold models trained with the test patients
#   results/stack_<model>_final/                       (only with STACK) stacking refitted on the final runs
#   submission/PL3_G08_test_results.csv (+ _scores.csv, _info.json)
# The hidden set must be in data/hidden/CirCor_test_students/ (or set HIDDEN=<folder>).
set -euo pipefail
cd "$(dirname "$0")/.."
ACC=${ACC:?set ACC to your GPU account (the one you pass to sbatch -A)}
MODEL=${1:?usage: bash scripts/submit_final.sh cnn|rnn [training options]}; shift
SEEDS=(42 1 2 3)
HIDDEN=${HIDDEN:-$( [[ -d data/hidden/CirCor_test_students ]] && echo data/hidden/CirCor_test_students || echo data/CirCor_test_students )}
OUT=${OUT:-submission/PL3_G08_test_results.csv}
TIME=$([[ $MODEL == rnn ]] && echo 02:00:00 || echo 01:00:00)
mkdir -p logs submission
[[ -d $HIDDEN/test_data ]] || { echo "missing $HIDDEN/test_data: copy the hidden set to Deucalion first"; exit 1; }

sub() { sbatch --parsable -A "$ACC" "$@"; }
JOBS=(); RUNS=()
for s in "${SEEDS[@]}"; do
    name="${MODEL}_final_s$s"
    RUNS+=("$name")
    JOBS+=("$(sub -t "$TIME" --job-name="$MODEL-final-s$s" "scripts/deucalion_$MODEL.sbatch" \
               --epochs 50 --patience 8 --include-test --seed "$s" --name "$name" "$@")")
done
DEP=afterok:$(IFS=:; echo "${JOBS[*]}")

PRED=(src.predict --wav-dir "$HIDDEN/test_data" --models "${RUNS[@]}"
      --demographics "$HIDDEN/test_data_participants.csv"
      --template "$HIDDEN/PLX_GYY_test_results.csv" --out "$OUT" --probs)
if [[ -n ${STACK:-} ]]; then
    ST=$(sub --dependency="$DEP" --job-name="stack-$MODEL-final" scripts/deucalion_py.sbatch src.stacking \
             --runs "${RUNS[@]}" --name "stack_${MODEL}_final" --variants meta_audio "$STACK")
    DEP=afterok:$ST
    PRED+=(--stack "results/stack_${MODEL}_final/stack_$STACK.json")
fi
sub --dependency="$DEP" --job-name="predict-$MODEL-final" scripts/deucalion_py.sbatch "${PRED[@]}"
echo "submitted ${#JOBS[@]} training jobs + prediction -> $OUT (squeue -u \$USER)"
