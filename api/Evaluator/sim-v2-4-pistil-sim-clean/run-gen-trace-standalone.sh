#!/bin/bash

clear

# hw/sw simulation
SYS_CONFIG=pistil-sys-base
TRACE_DIR=./traces

MODEL=llama3-8b # llama3-8b llama3-70b llama3-405b llama4-maverik llama4-scout
CHIPLETS=(64) # 64
SIM_BATCH_SIZE=128
SIM_KV_CACHE=8192


trace_file=$TRACE_DIR/${MODEL}-chiplets-${CHIPLETS}-bs-${SIM_BATCH_SIZE}-sl-${SIM_KV_CACHE}

python main-gen-trace.py \
  --model-cfg=./configs/model_configs/$MODEL.json \
  --sys-cfg=./configs/sys_configs/$SYS_CONFIG.json \
  --trace-file=$trace_file \
  --sim-num-layers=1 \
  --max-new-tokens=2 \
  --w-dtype=0.5 \
  --kv-dtype=1.0 \
  --sim-batch-size=$SIM_BATCH_SIZE \
  --sim-kv-cache=$SIM_KV_CACHE \
  --sim-vocab-size=128000 \
  --sim-num-chiplets=$CHIPLETS \
  --verbose=1 \
  --inference="True"

