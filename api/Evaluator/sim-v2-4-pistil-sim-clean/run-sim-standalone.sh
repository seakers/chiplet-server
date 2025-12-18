#!/bin/bash

clear

# Note: Prior to running llama3/gamma2 must log in with token: huggingface-cli login
# create hugging face token: https://huggingface.co/settings/tokens
# log-into hugging face and request access gamma2 repo: https://huggingface.co/google/gemma-2-2b-it/tree/main

export OMP_NUM_THREADS=1 # running on sapphire rapids llama3 needs 32 otherwise there's an error
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

TRACE_DIR=./traces
RESULTS_DIR=./trace-results

# hw/sw simulation
MODEL_CONFIG=llama3-8b                 # llama3-8b llama3-70b llama3-405b llama4-maverik llama4-scout 
SYS_CONFIG=pistil-sys-base

# select correct trace
NUM_LAYERS=1 # this is just a trace multiplier, not the number of transformer layers... be careful when using Maverik/Scout... (48 total layers, but interleaved x2 = 24)
SIM_BATCH_SIZE=1
SIM_KV_CACHE=8192
LM_HEAD=False # True or False

# Sweepable System Config Parameters
SIM_NUM_CHIPLETS=64 # requires unique trace to ensure sharding
SIM_BANK_GROUPS=1
SIM_CH_PER_LAYER=1
SIM_RANKS=1
SIM_FRAC_BANK_CAP=1.0
ON_CHIP_BUF=524288
KV_DTYPE=1.0

PLOT_EXE=True

numactl -m 0 python main-sim-trace.py \
    --model-cfg=./configs/model_configs/$MODEL_CONFIG.json \
    --sys-cfg=./configs/sys_configs/$SYS_CONFIG.json  \
    --trace-file=$TRACE_DIR/$MODEL_CONFIG-chiplets-$SIM_NUM_CHIPLETS-bs-$SIM_BATCH_SIZE-sl-$SIM_KV_CACHE \
    --results-dir=$RESULTS_DIR \
    --sim-batch-size=$SIM_BATCH_SIZE \
    --sim-kv-cache=$SIM_KV_CACHE \
    --sim-num-layers=$NUM_LAYERS \
    --sim-lm-head=$LM_HEAD \
    --max-new-tokens=2 \
    --sim-num-chiplets=$SIM_NUM_CHIPLETS \
    --sim-bank-groups=$SIM_BANK_GROUPS \
    --sim-ch-per-layer=$SIM_CH_PER_LAYER \
    --sim-ranks=$SIM_RANKS \
    --sim-frac-bank-cap=$SIM_FRAC_BANK_CAP \
    --w-dtype=0.5 \
    --kv-dtype=$KV_DTYPE \
    --plot-exe=$PLOT_EXE \
    --sim-on-chip-buffer=$ON_CHIP_BUF \
    --verbose=1 \
    --inference="False"


