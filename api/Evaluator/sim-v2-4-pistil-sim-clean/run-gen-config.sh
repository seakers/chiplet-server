#!/bin/bash

clear

rm ./configs/gen_configs/* > /dev/null

python ./main-gen-config.py \
    --output-config-dir=./configs/gen_configs/ \
    --base-sys-cfg=./configs/sys_configs/pistil-sys-base.json