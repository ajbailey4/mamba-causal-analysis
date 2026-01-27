#!/bin/bash
set -e

echo "==== Job environment ===="
hostname
date
nvidia-smi || true
echo "========================="

cd /workspace/mamba-causal-analysis

python experiments/causal_trace.py \
  --num_workers ${NUM_WORKERS} \
  --worker_id ${WORKER_ID} \
  --model_name ${MODEL_NAME} \
  --noise_level ${NOISE_LEVEL}
