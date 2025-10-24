#!/bin/bash
# Script to inspect custom OCR dataset with InternVL
source /data/isaackang/anaconda3/bin/activate internvl
cd /data/isaackang/Others/InternVL/internvl_chat
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

# Default: process 10 samples with a concise prompt
# Options:
#   --num_samples N           : Number of samples (default: 10, -1 for all)
#   --prompt "text"           : Custom prompt
#   --case-sensitive false    : Case-insensitive matching (default: false)
#   --ignore-punctuation true : Ignore punctuation in matching (default: true)
#   --ignore-space true       : Ignore spaces in matching (default: true)
#   --checkpoint MODEL        : Different model
python eval/vqa/inspect_custom_ocr.py \
  --checkpoint OpenGVLab/InternVL3-8B \
  "$@"

