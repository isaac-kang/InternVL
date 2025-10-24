# InfoVQA Custom Data Path Configuration Guide

This guide explains how to use a custom data path for InfoVQA evaluation in InternVL, changing from the default `./data/infographicsvqa` to `~/data/internvl/info_qa`.

## Solution Summary

Two solutions have been implemented:

### ✅ Solution 1: Symbolic Link (Already Applied - Recommended)

A symbolic link has been created that redirects the default path to your custom location. This requires **no code changes** and is the simplest approach.

**What was done:**
```bash
cd /data/isaackang/Others/InternVL/internvl_chat
mkdir -p data
ln -s ~/data/internvl/info_qa data/infographicsvqa
```

**Verification:**
```bash
ls -la /data/isaackang/Others/InternVL/internvl_chat/data/infographicsvqa
# Output: lrwxrwxrwx ... infographicsvqa -> /data/isaackang/data/internvl/info_qa
```

**Usage:** Just run evaluation normally:
```bash
cd /data/isaackang/Others/InternVL/internvl_chat
GPUS=8 bash evaluate.sh <CHECKPOINT> vqa-infovqa-val --dynamic --max-num 24
GPUS=8 bash evaluate.sh <CHECKPOINT> vqa-infovqa-test --dynamic --max-num 24
```

---

### ✅ Solution 2: Environment Variable (Also Implemented)

The evaluation script has been modified to support custom paths via environment variables. This provides more flexibility.

**What was modified:**
- File: `/data/isaackang/Others/InternVL/internvl_chat/eval/vqa/evaluate_vqa.py`
- Added support for `INFOVQA_DATA_PATH` environment variable
- Default value remains `data/infographicsvqa` for backward compatibility

**Usage with environment variable:**
```bash
cd /data/isaackang/Others/InternVL/internvl_chat

# Set the custom path
export INFOVQA_DATA_PATH=~/data/internvl/info_qa

# Run evaluation
GPUS=8 bash evaluate.sh <CHECKPOINT> vqa-infovqa-val --dynamic --max-num 24
```

**Or in a single command:**
```bash
INFOVQA_DATA_PATH=~/data/internvl/info_qa GPUS=8 bash evaluate.sh <CHECKPOINT> vqa-infovqa-val --dynamic --max-num 24
```

---

## Data Directory Structure

Make sure your custom InfoVQA data directory (`~/data/internvl/info_qa`) contains the following files:

```
~/data/internvl/info_qa/
├── infographicsvqa_images/          # Image directory
├── infographicsVQA_test_v1.0.json   # Test annotations
├── infographicsVQA_val_v1.0_withQT.json   # Validation annotations with question types
├── infographicVQA_train_v1.0.json   # Train annotations (optional)
├── test.jsonl                        # Converted test file
├── train.jsonl                       # Converted train file (optional)
└── val.jsonl                         # Converted validation file
```

## How to Download InfoVQA Data

If you haven't downloaded the data yet, follow these steps:

```bash
# Step 1: Create your custom directory
mkdir -p ~/data/internvl/info_qa
cd ~/data/internvl/info_qa

# Step 2: Download images and annotations from https://rrc.cvc.uab.es/?ch=17&com=downloads
# You'll need to download:
# - infographicsVQA_test_v1.0.json
# - infographicsVQA_val_v1.0_withQT.json
# - infographicVQA_train_v1.0.json
# - infographicsvqa_images/ (image directory)

# Step 3: Download converted files
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/infographicsvqa_val.jsonl -O val.jsonl
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/infographicsvqa_test.jsonl -O test.jsonl
```

## Evaluation Examples

### Example 1: Using Symbolic Link (Current Setup)
```bash
cd /data/isaackang/Others/InternVL/internvl_chat

# Validate on InfoVQA val set
GPUS=8 bash evaluate.sh /path/to/checkpoint vqa-infovqa-val --dynamic --max-num 24

# Test on InfoVQA test set
GPUS=8 bash evaluate.sh /path/to/checkpoint vqa-infovqa-test --dynamic --max-num 24
```

### Example 2: Using Environment Variable
```bash
cd /data/isaackang/Others/InternVL/internvl_chat

# Set custom path and run evaluation
export INFOVQA_DATA_PATH=~/data/internvl/info_qa
GPUS=8 bash evaluate.sh /path/to/checkpoint vqa-infovqa-val --dynamic --max-num 24
```

### Example 3: Direct Script Execution
```bash
cd /data/isaackang/Others/InternVL/internvl_chat

# With environment variable
export INFOVQA_DATA_PATH=~/data/internvl/info_qa
torchrun --nproc_per_node=8 eval/vqa/evaluate_vqa.py \
    --checkpoint /path/to/checkpoint \
    --datasets infographicsvqa_val \
    --dynamic \
    --max-num 24
```

## Troubleshooting

### Issue 1: Symbolic link not working
```bash
# Check if symbolic link exists
ls -la /data/isaackang/Others/InternVL/internvl_chat/data/infographicsvqa

# If broken, recreate it
rm /data/isaackang/Others/InternVL/internvl_chat/data/infographicsvqa
ln -s ~/data/internvl/info_qa /data/isaackang/Others/InternVL/internvl_chat/data/infographicsvqa
```

### Issue 2: Files not found
```bash
# Verify your data directory has all required files
ls -la ~/data/internvl/info_qa/

# Check if paths are correct
echo "INFOVQA_DATA_PATH: $INFOVQA_DATA_PATH"
```

### Issue 3: Permission issues
```bash
# Ensure you have read permissions
chmod -R u+r ~/data/internvl/info_qa/
```

## Which Solution Should You Use?

**Recommendation: Use Solution 1 (Symbolic Link)**
- ✅ No code maintenance required
- ✅ Works with any future updates to the codebase
- ✅ Simple and transparent
- ✅ Already set up and working

**Use Solution 2 (Environment Variable) if:**
- You need to frequently switch between different data directories
- You want to programmatically control the data path
- You're running automated experiments with different data locations

## Reverting Changes

### To remove the symbolic link:
```bash
rm /data/isaackang/Others/InternVL/internvl_chat/data/infographicsvqa
```

### To revert code changes:
```bash
cd /data/isaackang/Others/InternVL
git checkout internvl_chat/eval/vqa/evaluate_vqa.py
```

## Additional Notes

- The `--max-num 24` parameter is recommended for InfoVQA to handle high-resolution infographic images
- The `--dynamic` flag enables dynamic resolution preprocessing, which is essential for InternVL models
- Both validation and test sets are supported: `vqa-infovqa-val` and `vqa-infovqa-test`
- Results will be saved in the `results/` directory

## Contact & Support

If you encounter any issues, check:
1. Data directory structure matches the expected format
2. File permissions are correct
3. Symbolic link or environment variable is properly set
4. CUDA and GPU availability for multi-GPU evaluation

