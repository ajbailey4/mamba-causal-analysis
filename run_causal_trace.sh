#!/bin/bash

export HOME=$PWD
export MPLCONFIGDIR=$PWD/.matplotlib
export XDG_CACHE_HOME=$PWD/.cache
export TORCHINDUCTOR_CACHE_DIR=$PWD/.torchinductor
export TRANSFORMERS_CACHE=$PWD/.hf_cache
export HF_HOME=$PWD/.hf_home
export TORCHDYNAMO_DISABLE=1

mkdir -p \
    $MPLCONFIGDIR \
    $XDG_CACHE_HOME \
    $TORCHINDUCTOR_CACHE_DIR \
    $TRANSFORMERS_CACHE \
    $HF_HOME

python -m experiments.causal_trace \
  --num_workers ${NUM_WORKERS} \
  --worker_id ${WORKER_ID} \
  --model_name ${MODEL_NAME} \
  --noise_level ${NOISE_LEVEL}
