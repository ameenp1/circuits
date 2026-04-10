#!/bin/bash
set -euo pipefail

# Run MLP neuron sparsity vs SAE MLP-out sparsity on CausalGym agr_sv_num_pp.

CONFIG="sweep/fc_evaluation/causalgym_pair.yaml"
DATASET="causalgym_pp"
N_EXAMPLES=300
RESULTS_DIR="${CIRCUITS_RESULTS_DIR:-results}/causalgym/pair"

mkdir -p "${RESULTS_DIR}"

COMMON_ARGS="--config ${CONFIG} \
    --method nap \
    --num_train_examples ${N_EXAMPLES} \
    --dataset ${DATASET} \
    --data_path data/causalgym \
    --save_path ${RESULTS_DIR}"

echo "============================================"
echo "Experiment 1: MLP neurons (activations), IG"
echo "============================================"
python scripts/train.py ${COMMON_ARGS} \
    --steps 10 \
    --use_neurons \
    --use_mlp_acts \
    --disable_stop_grad

python scripts/eval.py ${COMMON_ARGS} \
    --steps 10 \
    --use_neurons \
    --use_mlp_acts \
    --disable_stop_grad

echo "============================================"
echo "Experiment 2: SAE MLP outputs, IG"
echo "============================================"
python scripts/train.py ${COMMON_ARGS} \
    --steps 10 \
    --disable_stop_grad \
    --auc_test_handle_errors default

python scripts/eval.py ${COMMON_ARGS} \
    --steps 10 \
    --disable_stop_grad \
    --auc_test_handle_errors default

echo "============================================"
echo "Done! Results in: ${RESULTS_DIR}/"
echo "============================================"
