#!/bin/bash

clear

######################################
# CREATE CONDA ENVIRONMMENT
# conda create -n pistil-sim python=3.12
# conda activate pistil-sim
# pip install torch numpy matplotlib scipy tqdm
######################################


######################################
# RUN SETUP
GEN_CONFIG=True
GEN_TRACE=True
SIM_STANDALONE=True
######################################


######################################
# HARDWARE KNOBS
# COMPUTE KNOBS
NUM_TMACS=8
MEM_BUF_CAP=0.25
NET_BUF_CAP=1.0

# MEMORY KNOBS
MEM_BANKS_PER_GROUP=2
MEM_RANKS=4
MEM_FRAC_BANK_CAP=1.0

# SCALE UP KNOBS
NUM_CUS=64
######################################


######################################
# APPLICATION KNOBS
MODEL=llama3-70b # llama3-8b llama3-70b llama3-405b llama4-maverick llama4-scout
BATCH_SIZE=16
KV_CACHE=8192
W_DTYPE=0.5
KV_DTYPE=1.0

# Prefill Knobs
PREFILL=False                # consider first batch a prefill with specific chunk size
PREFILL_CHUNK_SIZE=256      # chunk of KV which is batched and needs to be written back
PREFILL_CACHED=1024         # amount of KV already cached for prefill - could start at 0
######################################


######################################
# SIMULATION KNOBS
SIM_NUM_LAYERS=-1
LM_HEAD=False # True or False
PLOT_EXE=True # This can take some time and memory if simulating lots of layers
######################################


######################################
# CONFIG AND TRACE NAMING
BASE_CONFIG=pistil-sys-base.json
CONFIG_NAME=pistil-config-num_cus-${NUM_CUS}-tmacs-${NUM_TMACS}-mem_buf_cap-${MEM_BUF_CAP}-net_buf_cap-${NET_BUF_CAP}-mem_bank_groups-${MEM_BANKS_PER_GROUP}-mem_ranks-${MEM_RANKS}-mem_frac_bank_cap-${MEM_FRAC_BANK_CAP}.json
TRACE_NAME=${MODEL}-chiplets-${NUM_CUS}-bs-${BATCH_SIZE}-sl-${KV_CACHE}
######################################


######################################
# PATH SETUP
OUTPUT_CONFIG_DIR=./configs/gen_configs
BASE_CONFIG_DIR=./configs/sys_configs
MODEL_CONFIG_DIR=./configs/model_configs
TRACE_DIR=./traces
RESULTS_DIR=./trace-results
######################################


if [ "$GEN_CONFIG" = "True" ]; then
    python ./main-gen-config.py \
        --name=$CONFIG_NAME \
        --output-config-dir=$OUTPUT_CONFIG_DIR \
        --base-sys-cfg=$BASE_CONFIG_DIR/$BASE_CONFIG \
        --tmacs=$NUM_TMACS \
        --mem-buf-cap=$MEM_BUF_CAP \
        --net-buf-cap=$NET_BUF_CAP \
        --mem-bank-groups=$MEM_BANKS_PER_GROUP \
        --mem-ranks=$MEM_RANKS \
        --mem-frac-bank-cap=$MEM_FRAC_BANK_CAP \
        --num-cus=$NUM_CUS
    if [ $? -ne 0 ]; then
        echo "Gen Config failed, exiting..."
        exit 1
    fi
fi


if [ "$GEN_TRACE" = "True" ]; then
    python main-gen-trace.py \
        --model-name=$MODEL \
        --model-cfg=$MODEL_CONFIG_DIR/$MODEL.json \
        --sys-cfg=$OUTPUT_CONFIG_DIR/$CONFIG_NAME \
        --trace-file=$TRACE_DIR/$TRACE_NAME \
        --w-dtype=$W_DTYPE \
        --kv-dtype=$KV_DTYPE \
        --sim-batch-size=$BATCH_SIZE \
        --sim-prefill-batch=$PREFILL \
        --sim-prefill-chunk-size=$PREFILL_CHUNK_SIZE \
        --sim-prefill-cached=$PREFILL_CACHED \
        --sim-kv-cache=$KV_CACHE \
        --sim-vocab-size=128000 \
        --sim-num-chiplets=$NUM_CUS \
        --verbose=1 \
        --inference="True"
    if [ $? -ne 0 ]; then
        echo "Gen Trace failed, exiting..."
        exit 1
    fi
fi


if [ "$SIM_STANDALONE" = "True" ]; then
    numactl -m 0 python main-sim-trace.py \
        --model-name=$MODEL \
        --model-cfg=$MODEL_CONFIG_DIR/$MODEL.json \
        --sys-cfg=$OUTPUT_CONFIG_DIR/$CONFIG_NAME  \
        --trace-file=$TRACE_DIR/$TRACE_NAME \
        --results-dir=$RESULTS_DIR \
        --sim-batch-size=$BATCH_SIZE \
        --sim-prefill-batch=$PREFILL \
        --sim-prefill-chunk-size=$PREFILL_CHUNK_SIZE \
        --sim-prefill-cached=$PREFILL_CACHED \
        --sim-kv-cache=$KV_CACHE \
        --sim-num-layers=$SIM_NUM_LAYERS \
        --sim-lm-head=$LM_HEAD \
        --w-dtype=$W_DTYPE \
        --kv-dtype=$KV_DTYPE \
        --sim-num-chiplets=$NUM_CUS \
        --plot-exe=$PLOT_EXE \
        --verbose=1 \
        --inference="False"
fi

