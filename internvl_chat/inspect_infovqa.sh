#!/bin/bash
# Simple script to inspect InfoVQA samples with InternVL
source /data/isaackang/anaconda3/bin/activate internvl
cd /data/isaackang/Others/InternVL/internvl_chat
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

# Default: 10 samples, change with --num_samples argument
python eval/vqa/inspect_infovqa.py \
  --checkpoint OpenGVLab/InternVL3-1B \
  --num_samples 10 \
  "$@"

