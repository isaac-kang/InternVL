#!/bin/bash
# Script to inspect STR LMDB benchmarks with InternVL
source /data/isaackang/anaconda3/bin/activate internvl
cd /data/isaackang/Others/InternVL/internvl_chat
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
export STR_DATA_DIR=~/data/STR/english_case-sensitive/lmdb/evaluation

# Default: process all samples per dataset with optimized prompt
# Options:
#   --num_samples N           : Number of samples per dataset (default: -1 for all)
#   --batch_size N            : Batch size for inference (default: 8, faster!)
#   --save-images true        : Save images to disk (default: false, saves space!)
#   --datasets "CUTE80,SVT"   : Specific datasets to evaluate
#   --prompt "text"           : Custom prompt
#   --case-sensitive false    : Case-insensitive matching (default: false)
#   --ignore-punctuation true : Ignore punctuation (default: true)
#   --ignore-space true       : Ignore spaces (default: true)
#   --checkpoint MODEL        : Different model
python eval/vqa/inspect_str_lmdb.py \
  --checkpoint OpenGVLab/InternVL3-8B \
  --num_samples 2 \
  --batch_size 1 \
  --save-images false \
  --datasets "CUTE80,SVT,SVTP,IC13_857,IC15_1811,IIIT5k_3000" \
  --prompt "What is the main word in the image? Output only the text." \
  --case-sensitive false \
  --ignore-punctuation true \
  --ignore-space true \
  "$@"

