#!/bin/bash
source /data/isaackang/anaconda3/bin/activate internvl
cd /data/isaackang/Others/InternVL/internvl_chat
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

torchrun --nproc_per_node=1 --master_port=29503 eval/vqa/evaluate_vqa.py \
  --checkpoint OpenGVLab/InternVL3-1B \
  --datasets infographicsvqa_val \
  --dynamic

