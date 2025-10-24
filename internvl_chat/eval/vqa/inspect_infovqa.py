#!/usr/bin/env python3
"""
Simple script to inspect a few InfoVQA samples with InternVL3-1B
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='OpenGVLab/InternVL3-1B')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--data_path', type=str, default='~/data/internvl/infographicsvqa')
    parser.add_argument('--output_dir', type=str, default='results/inspection')
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
    
    # Load data
    data_path = os.path.expanduser(args.data_path)
    test_file = os.path.join(data_path, 'val.jsonl')
    test_data = []
    with open(test_file, 'r') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    
    # Take only num_samples
    test_data = test_data[:args.num_samples]
    
    print(f"Processing {len(test_data)} samples...")
    
    transform = build_transform(is_train=False, input_size=image_size)
    
    # Open output text file
    output_txt = os.path.join(args.output_dir, 'predictions.txt')
    f_out = open(output_txt, 'w', encoding='utf-8')
    f_out.write("=" * 100 + "\n")
    f_out.write(f"InternVL InfoVQA Inspection - Model: {args.checkpoint}\n")
    f_out.write("=" * 100 + "\n\n")
    
    for idx, item in enumerate(tqdm(test_data)):
        image_path = item['image']
        question = item['question']
        question_id = item['question_id']
        gt_answer = item.get('answer', '')
        
        # Fix image path
        if not os.path.isabs(image_path):
            if image_path.startswith('data/infographicsvqa/'):
                image_path = image_path.replace('data/infographicsvqa/', os.path.expanduser('~/data/internvl/infographicsvqa/'))
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        
        # Copy image to output directory with a readable name
        image_ext = os.path.splitext(image_path)[1]
        output_image_name = f"sample_{idx:03d}_qid_{question_id}{image_ext}"
        output_image_path = os.path.join(images_dir, output_image_name)
        shutil.copy(image_path, output_image_path)
        
        # Process image for model
        images = dynamic_preprocess(image, image_size=image_size, use_thumbnail=use_thumbnail, max_num=6)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
        
        # Prepare prompt
        prompt = 'Answer the question using a single word or phrase.'
        full_question = f"{question} {prompt}"
        
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
                question=full_question,
                generation_config=generation_config,
                verbose=False
            )
        
        # Format ground truth
        if isinstance(gt_answer, list):
            gt_str = ' | '.join(gt_answer)
        else:
            gt_str = str(gt_answer)
        
        # Write to text file
        f_out.write(f"Sample {idx + 1}/{len(test_data)}\n")
        f_out.write("-" * 100 + "\n")
        f_out.write(f"Image:          {output_image_name}\n")
        f_out.write(f"Question ID:    {question_id}\n")
        f_out.write(f"Question:       {question}\n")
        f_out.write(f"Model Answer:   {response.strip()}\n")
        f_out.write(f"Ground Truth:   {gt_str}\n")
        f_out.write("\n\n")
        
        # Also print to console
        print(f"\n{'='*80}")
        print(f"Sample {idx + 1}: {output_image_name}")
        print(f"Q: {question}")
        print(f"A: {response.strip()}")
        print(f"GT: {gt_str}")
        print(f"{'='*80}\n")
    
    f_out.write("=" * 100 + "\n")
    f_out.write("Inspection Complete!\n")
    f_out.write("=" * 100 + "\n")
    f_out.close()
    
    print(f"\n{'='*80}")
    print(f"✓ Inspection complete!")
    print(f"✓ Images saved to: {images_dir}")
    print(f"✓ Predictions saved to: {output_txt}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

