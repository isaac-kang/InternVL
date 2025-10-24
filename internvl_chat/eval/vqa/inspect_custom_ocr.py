#!/usr/bin/env python3
"""
Script to inspect custom OCR dataset with InternVL
Saves images and predictions for manual inspection
"""
import argparse
import json
import os
import shutil
import torch
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='OpenGVLab/InternVL3-8B')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to process (default: 10, -1 for all)')
    parser.add_argument('--data_path', type=str, default='~/data/internvl/example_custom_dataset')
    parser.add_argument('--output_dir', type=str, default='results/custom_ocr_inspection')
    parser.add_argument('--prompt', type=str, default='What is the main word in the image? Output only the text.', help='Instruction prompt for the model')
    parser.add_argument('--case-sensitive', type=lambda x: x.lower() == 'true', default=False, 
                        help='Whether to do case-sensitive matching (default: False)')
    parser.add_argument('--ignore-punctuation', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether to ignore punctuation in matching (default: True)')
    parser.add_argument('--ignore-space', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether to ignore spaces in matching (default: True)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
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
    print(f"Matching options - Case-sensitive: {args.case_sensitive}, Ignore punctuation: {args.ignore_punctuation}, Ignore space: {args.ignore_space}")
    
    # Load data
    data_path = os.path.expanduser(args.data_path)
    labels_file = os.path.join(data_path, 'labels.json')
    images_path = os.path.join(data_path, 'images')
    
    with open(labels_file, 'r') as f:
        data = json.load(f)
    
    # Take only num_samples (default 10, or -1 for all)
    if args.num_samples > 0:
        data = data[:args.num_samples]
    
    # Verify all images exist
    verified_data = []
    for item in data:
        image_filename = item['image_filename']
        image_path = os.path.join(images_path, image_filename)
        if os.path.exists(image_path):
            verified_data.append(item)
        else:
            print(f"Warning: Image not found: {image_path}")
    
    data = verified_data
    
    print(f"Processing {len(data)} samples...")
    
    transform = build_transform(is_train=False, input_size=image_size)
    
    # Open output text file
    output_txt = os.path.join(args.output_dir, 'predictions.txt')
    f_out = open(output_txt, 'w', encoding='utf-8')
    f_out.write("=" * 100 + "\n")
    f_out.write(f"InternVL Custom OCR Dataset Inspection\n")
    f_out.write(f"Model: {args.checkpoint}\n")
    f_out.write(f"Prompt: {args.prompt}\n")
    f_out.write(f"Matching - Case-sensitive: {args.case_sensitive}, Ignore punctuation: {args.ignore_punctuation}, Ignore space: {args.ignore_space}\n")
    f_out.write("=" * 100 + "\n\n")
    
    correct = 0
    total = 0
    
    for idx, item in enumerate(tqdm(data)):
        image_id = item['image_id']
        image_filename = item['image_filename']
        gt_text = item['text']
        
        # Load image
        image_path = os.path.join(images_path, image_filename)
        image = Image.open(image_path).convert('RGB')
        
        # Copy image to output directory
        output_image_name = f"sample_{idx:03d}_id_{image_id}_{image_filename}"
        output_image_path = os.path.join(images_dir, output_image_name)
        shutil.copy(image_path, output_image_path)
        
        # Process image for model
        images = dynamic_preprocess(image, image_size=image_size, use_thumbnail=use_thumbnail, max_num=6)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
        
        # Generate answer
        generation_config = dict(
            num_beams=1,
            max_new_tokens=100,
            min_new_tokens=1,
            do_sample=False,
        )
        
        with torch.no_grad():
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=args.prompt,
                generation_config=generation_config,
                verbose=False
            )
        
        response = response.strip()
        
        # Accuracy check with configurable options
        is_correct = check_match(response, gt_text, 
                                 case_sensitive=args.case_sensitive,
                                 ignore_punctuation=args.ignore_punctuation,
                                 ignore_space=args.ignore_space)
        if is_correct:
            correct += 1
        total += 1
        
        # Write to text file
        f_out.write(f"Sample {idx + 1}/{len(data)}\n")
        f_out.write("-" * 100 + "\n")
        f_out.write(f"Image:          {output_image_name}\n")
        f_out.write(f"Image ID:       {image_id}\n")
        f_out.write(f"Prompt:         {args.prompt}\n")
        f_out.write(f"Model Answer:   {response}\n")
        f_out.write(f"Ground Truth:   {gt_text}\n")
        f_out.write(f"Correct:        {'✓' if is_correct else '✗'}\n")
        f_out.write("\n\n")
        
        # Also print to console
        print(f"\n{'='*80}")
        print(f"Sample {idx + 1}: {output_image_name}")
        print(f"Prompt: {args.prompt}")
        print(f"Model: {response}")
        print(f"GT:    {gt_text}")
        print(f"{'✓ CORRECT' if is_correct else '✗ WRONG'}")
        print(f"{'='*80}\n")
    
    # Write summary
    accuracy = correct / total * 100 if total > 0 else 0
    f_out.write("=" * 100 + "\n")
    f_out.write(f"Inspection Complete!\n")
    f_out.write(f"Accuracy: {correct}/{total} = {accuracy:.2f}%\n")
    f_out.write("=" * 100 + "\n")
    f_out.close()
    
    print(f"\n{'='*80}")
    print(f"✓ Inspection complete!")
    print(f"✓ Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print(f"✓ Images saved to: {images_dir}")
    print(f"✓ Predictions saved to: {output_txt}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

