# InfoVQA Quick Start - Custom Data Path

## ✅ Configuration Complete!

Your InfoVQA data path has been successfully changed from `./data/infographicsvqa` to `~/data/internvl/info_qa`

## 📁 Data Location

**Your InfoVQA data should be placed in:**
```
~/data/internvl/info_qa/
```
(Expands to: `/data/isaackang/data/internvl/info_qa/`)

## 🚀 How to Run Evaluation

### Method 1: Using Symbolic Link (Recommended - Already Set Up)

Simply run the evaluation command as usual:

```bash
cd /data/isaackang/Others/InternVL/internvl_chat

# InfoVQA validation set
GPUS=8 bash evaluate.sh <CHECKPOINT_PATH> vqa-infovqa-val --dynamic --max-num 24

# InfoVQA test set
GPUS=8 bash evaluate.sh <CHECKPOINT_PATH> vqa-infovqa-test --dynamic --max-num 24
```

### Method 2: Using Environment Variable

If you need to use a different path temporarily:

```bash
cd /data/isaackang/Others/InternVL/internvl_chat

# Set custom path and run
INFOVQA_DATA_PATH=/your/custom/path GPUS=8 bash evaluate.sh <CHECKPOINT_PATH> vqa-infovqa-val --dynamic --max-num 24
```

## 📋 Data Directory Structure

Ensure your data directory contains these files:

```
~/data/internvl/info_qa/
├── infographicsvqa_images/               # Image directory
├── infographicsVQA_test_v1.0.json        # Test annotations
├── infographicsVQA_val_v1.0_withQT.json  # Validation annotations
├── infographicVQA_train_v1.0.json        # Training annotations (optional)
├── test.jsonl                            # Converted test questions
├── train.jsonl                           # Converted train questions (optional)
└── val.jsonl                             # Converted validation questions
```

## 📥 Download InfoVQA Data

If you haven't downloaded the data yet:

```bash
# Create directory
mkdir -p ~/data/internvl/info_qa
cd ~/data/internvl/info_qa

# Download annotations from https://rrc.cvc.uab.es/?ch=17&com=downloads
# (Manual download required - register on the website)
# - infographicsVQA_test_v1.0.json
# - infographicsVQA_val_v1.0_withQT.json
# - infographicVQA_train_v1.0.json
# - infographicsvqa_images/ directory

# Download converted files
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/infographicsvqa_val.jsonl -O val.jsonl
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/infographicsvqa_test.jsonl -O test.jsonl
```

## 🔍 Verify Setup

Check if the symbolic link is working:

```bash
ls -la /data/isaackang/Others/InternVL/internvl_chat/data/infographicsvqa
# Should show: infographicsvqa -> /data/isaackang/data/internvl/info_qa

# Check if data exists
ls ~/data/internvl/info_qa/
```

## 💡 Example Commands

```bash
# Example 1: Evaluate InternVL2.5-8B on InfoVQA validation
GPUS=8 bash evaluate.sh OpenGVLab/InternVL2_5-8B vqa-infovqa-val --dynamic --max-num 24

# Example 2: Single GPU evaluation
GPUS=1 bash evaluate.sh OpenGVLab/InternVL2_5-8B vqa-infovqa-val --dynamic --max-num 24

# Example 3: With 8-bit quantization for memory efficiency
GPUS=8 bash evaluate.sh OpenGVLab/InternVL2_5-8B vqa-infovqa-val --dynamic --max-num 24 --load-in-8bit
```

## ❓ Troubleshooting

**Issue: "FileNotFoundError: data/infographicsvqa/..."**
- Check if symbolic link exists: `ls -la /data/isaackang/Others/InternVL/internvl_chat/data/infographicsvqa`
- Recreate if needed: `ln -s ~/data/internvl/info_qa /data/isaackang/Others/InternVL/internvl_chat/data/infographicsvqa`

**Issue: "Directory is empty"**
- Download the InfoVQA dataset files to `~/data/internvl/info_qa/`
- Verify files exist: `ls ~/data/internvl/info_qa/`

**Issue: Want to use original path**
- Remove symbolic link: `rm /data/isaackang/Others/InternVL/internvl_chat/data/infographicsvqa`
- Create directory: `mkdir -p /data/isaackang/Others/InternVL/internvl_chat/data/infographicsvqa`
- Place data in the original location

## 📚 More Information

For detailed documentation, see: `INFOVQA_PATH_GUIDE.md`

For general evaluation guide, see: `/data/isaackang/Others/InternVL/internvl_chat/eval/vqa/README.md`

