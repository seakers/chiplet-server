#!/bin/bash

clear

WORKSPACE=../chiplet-server/chiplet-model
TRACE_DIR=$WORKSPACE/traces
CHIPLET_LIBRARY=$WORKSPACE/dse/chiplet-library


#EXPERIMENT_DIR=$WORKSPACE/dse/experiments/gpt-j-1024-weighted.json
#EXPERIMENT_DIR=$WORKSPACE/dse/experiments/resnet50-test.json
#EXPERIMENT_DIR=$WORKSPACE/dse/experiments/ogbn-products-test.json
#EXPERIMENT_DIR=$WORKSPACE/dse/experiments/gpt-j-65536-weighted.json
EXPERIMENT_DIR=$WORKSPACE/dse/experiments/sd-test.json


python chiplet-model/dse/test_system_eval.py --chiplet-library=$CHIPLET_LIBRARY --trace-dir=$TRACE_DIR --experiment=$EXPERIMENT_DIR

#python ./test_chiplet_dse.py --chiplet-library=$CHIPLET_LIBRARY --trace-dir=$TRACE_DIR --experiment=$EXPERIMENT_DIR

