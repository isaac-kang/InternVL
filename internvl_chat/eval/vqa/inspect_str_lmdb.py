#!/usr/bin/env python3
"""
Script to inspect STR (Scene Text Recognition) LMDB datasets with InternVL
Processes datasets from ~/data/STR/english_case-sensitive/lmdb/evaluation/
"""
import argparse
import json
import os
import io
import shutil
import torch
import lmdb
import six
from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess
from transformers import AutoTokenizer
from PIL import Image
from tqdm import tqdm

def normalize_text(text, ignore_punctuation=False, ignore_space=True):
    """Normalize text for comparison"""
    import string
    if ignore_punctuation:
        # Remove all punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
    if ignore_space:
        # Remove all spaces
        text = text.replace(' ', '')
    return text.strip()

def check_match(pred, gt, case_sensitive=True, ignore_punctuation=False, ignore_space=True):
    """Check if prediction matches ground truth"""
    pred = normalize_text(pred, ignore_punctuation, ignore_space)
    gt = normalize_text(gt, ignore_punctuation, ignore_space)
    
    if not case_sensitive:
        pred = pred.lower()
        gt = gt.lower()
    
    return pred == gt

class LMDBDataset:
    """Simple LMDB dataset reader for STR benchmarks"""
    def __init__(self, root):
        self.root = root
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        
        if not self.env:
            raise Exception(f'Cannot open LMDB dataset from {root}')
        
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
    
    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, index):
        assert index < len(self), f'Index {index} out of range'
        index += 1  # LMDB index starts from 1
        
        with self.env.begin(write=False) as txn:
            label_key = f'label-{index:09d}'.encode()
            label = txn.get(label_key).decode('utf-8')
            
            img_key = f'image-{index:09d}'.encode()
            imgbuf = txn.get(img_key)
            
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            
            try:
                img = Image.open(buf).convert('RGB')
            except Exception as e:
                print(f'Error reading image {index}: {e}')
                img = Image.new('RGB', (100, 32), color='white')
            
        return img, label, index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='OpenGVLab/InternVL3-8B')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples per dataset (default: 10, -1 for all)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference (default: 1)')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--datasets', type=str, default='CUTE80,SVT,SVTP,IC13_857,IC15_1811,IIIT5k_3000',
                        help='Comma-separated list of datasets to evaluate')
    parser.add_argument('--output_dir', type=str, default='results/str_lmdb_inspection')
    parser.add_argument('--prompt', type=str, default='What is the main word in the image? Output only the text.', help='Instruction prompt for the model')
    parser.add_argument('--case-sensitive', type=lambda x: x.lower() == 'true', default=False,
                        help='Whether to do case-sensitive matching (default: False)')
    parser.add_argument('--ignore-punctuation', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether to ignore punctuation in matching (default: True)')
    parser.add_argument('--ignore-space', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether to ignore spaces in matching (default: True)')
    parser.add_argument('--save-images', type=lambda x: x.lower() == 'true', default=False,
                        help='Whether to save images to disk (default: False)')
    args = parser.parse_args()
    
    # Get data_root from environment variable if not provided
    if args.data_root is None:
        args.data_root = os.environ.get('STR_DATA_DIR')
        if args.data_root is None:
            raise ValueError(
                "STR_DATA_DIR environment variable is not set. "
                "Please set it using: export STR_DATA_DIR=/path/to/str/data "
                "or provide --data_root argument."
            )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.checkpoint}")
    model = InternVLChatModel.from_pretrained(
        args.checkpoint,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval().cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    
    print(f"Model loaded! Image size: {image_size}, Use thumbnail: {use_thumbnail}")
    print(f"Using prompt: '{args.prompt}'")
    print(f"Batch size: {args.batch_size}")
    print(f"Save images: {args.save_images}")
    print(f"Matching - Case-sensitive: {args.case_sensitive}, Ignore punct: {args.ignore_punctuation}, Ignore space: {args.ignore_space}")
    
    transform = build_transform(is_train=False, input_size=image_size)
    
    # Process each dataset
    data_root = os.path.expanduser(args.data_root)
    dataset_names = args.datasets.split(',')
    
    all_results = {}
    
    for ds_name in dataset_names:
        ds_path = os.path.join(data_root, ds_name)
        if not os.path.exists(ds_path):
            print(f"Warning: Dataset {ds_name} not found at {ds_path}, skipping...")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing dataset: {ds_name}")
        print(f"{'='*80}")
        
        # Load LMDB dataset
        try:
            dataset = LMDBDataset(ds_path)
            total_samples = len(dataset)
            print(f"Total samples in {ds_name}: {total_samples}")
        except Exception as e:
            print(f"Error loading {ds_name}: {e}")
            continue
        
        # Determine how many to process
        num_to_process = args.num_samples if args.num_samples > 0 else total_samples
        num_to_process = min(num_to_process, total_samples)
        
        # Create dataset-specific output directory
        ds_output_dir = os.path.join(args.output_dir, ds_name)
        os.makedirs(ds_output_dir, exist_ok=True)
        
        if args.save_images:
            images_dir = os.path.join(ds_output_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)
        else:
            images_dir = None
        
        # Open output text file
        output_txt = os.path.join(ds_output_dir, 'predictions.txt')
        f_out = open(output_txt, 'w', encoding='utf-8')
        f_out.write("=" * 100 + "\n")
        f_out.write(f"InternVL STR LMDB Dataset Inspection - {ds_name}\n")
        f_out.write(f"Model: {args.checkpoint}\n")
        f_out.write(f"Prompt: {args.prompt}\n")
        f_out.write(f"Matching - Case-sensitive: {args.case_sensitive}, Ignore punct: {args.ignore_punctuation}, Ignore space: {args.ignore_space}\n")
        f_out.write("=" * 100 + "\n\n")
        
        correct = 0
        total = 0
        
        # Process in batches
        batch_size = args.batch_size
        num_batches = (num_to_process + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc=ds_name):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_to_process)
            
            batch_images = []
            batch_pixel_values = []
            batch_num_patches = []
            batch_gt_texts = []
            batch_img_ids = []
            batch_indices = []
            
            # Load batch
            for idx in range(start_idx, end_idx):
                try:
                    image, gt_text, img_id = dataset[idx]
                except Exception as e:
                    print(f"Error loading sample {idx}: {e}")
                    continue
                
                # Save image (optional)
                output_image_name = f"sample_{idx:04d}_id_{img_id}.png"
                if args.save_images:
                    output_image_path = os.path.join(images_dir, output_image_name)
                    image.save(output_image_path)
                
                # Process image for model
                images = dynamic_preprocess(image, image_size=image_size, use_thumbnail=use_thumbnail, max_num=6)
                pixel_values = [transform(img) for img in images]
                pixel_values = torch.stack(pixel_values)
                
                batch_images.append(image)
                batch_pixel_values.append(pixel_values)
                batch_num_patches.append(pixel_values.shape[0])
                batch_gt_texts.append(gt_text)
                batch_img_ids.append(img_id)
                batch_indices.append(idx)
            
            if len(batch_pixel_values) == 0:
                continue
            
            # Concatenate all pixel values
            pixel_values_batch = torch.cat(batch_pixel_values, dim=0).to(torch.bfloat16).cuda()
            
            # Generate answers
            generation_config = dict(
                num_beams=1,
                max_new_tokens=100,
                min_new_tokens=1,
                do_sample=False,
            )
            
            with torch.no_grad():
                if batch_size == 1:
                    # Single sample
                    response = model.chat(
                        tokenizer=tokenizer,
                        pixel_values=pixel_values_batch,
                        question=args.prompt,
                        generation_config=generation_config,
                        verbose=False
                    )
                    responses = [response]
                else:
                    # Batch inference
                    questions = [args.prompt] * len(batch_num_patches)
                    responses = model.batch_chat(
                        tokenizer=tokenizer,
                        pixel_values=pixel_values_batch,
                        num_patches_list=batch_num_patches,
                        questions=questions,
                        generation_config=generation_config,
                        verbose=False
                    )
            
            # Process responses
            for i, (response, gt_text, img_id, idx) in enumerate(zip(responses, batch_gt_texts, batch_img_ids, batch_indices)):
                response = response.strip()
                
                # Accuracy check
                is_correct = check_match(response, gt_text,
                                         case_sensitive=args.case_sensitive,
                                         ignore_punctuation=args.ignore_punctuation,
                                         ignore_space=args.ignore_space)
                if is_correct:
                    correct += 1
                total += 1
                
                # Write to text file
                output_image_name = f"sample_{idx:04d}_id_{img_id}.png"
                f_out.write(f"Sample {idx + 1}/{num_to_process}\n")
                f_out.write("-" * 100 + "\n")
                f_out.write(f"Image:          {output_image_name}\n")
                f_out.write(f"Image ID:       {img_id}\n")
                f_out.write(f"Prompt:         {args.prompt}\n")
                f_out.write(f"Model Answer:   {response}\n")
                f_out.write(f"Ground Truth:   {gt_text}\n")
                f_out.write(f"Correct:        {'✓' if is_correct else '✗'}\n")
                f_out.write("\n\n")
        
        # Write dataset summary
        accuracy = correct / total * 100 if total > 0 else 0
        f_out.write("=" * 100 + "\n")
        f_out.write(f"Dataset {ds_name} Complete!\n")
        f_out.write(f"Accuracy: {correct}/{total} = {accuracy:.2f}%\n")
        f_out.write("=" * 100 + "\n")
        f_out.close()
        
        all_results[ds_name] = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        }
        
        print(f"\n{ds_name}: {correct}/{total} = {accuracy:.2f}%")
        print(f"Results saved to: {output_txt}")
    
    # Write overall summary
    summary_file = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"InternVL STR Benchmarks Summary\n")
        f.write(f"Model: {args.checkpoint}\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Matching - Case-sensitive: {args.case_sensitive}, Ignore punct: {args.ignore_punctuation}, Ignore space: {args.ignore_space}\n")
        f.write("=" * 100 + "\n\n")
        
        for ds_name, result in all_results.items():
            f.write(f"{ds_name:20s}: {result['correct']:4d}/{result['total']:4d} = {result['accuracy']:6.2f}%\n")
        
        # Calculate average
        if all_results:
            avg_acc = sum(r['accuracy'] for r in all_results.values()) / len(all_results)
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"Average Accuracy: {avg_acc:.2f}%\n")
            f.write("=" * 100 + "\n")
    
    print(f"\n{'='*80}")
    print(f"✓ All evaluations complete!")
    print(f"✓ Summary saved to: {summary_file}")
    if args.save_images:
        print(f"✓ Images saved to: {args.output_dir}/*/images/")
    print(f"{'='*80}")
    
    # Print summary to console
    print(f"\nSummary:")
    for ds_name, result in all_results.items():
        print(f"  {ds_name:20s}: {result['correct']:4d}/{result['total']:4d} = {result['accuracy']:6.2f}%")
    
    if all_results:
        avg_acc = sum(r['accuracy'] for r in all_results.values()) / len(all_results)
        print(f"\n  Average: {avg_acc:.2f}%")

if __name__ == "__main__":
    main()

