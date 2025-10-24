#!/usr/bin/env python3
"""
Evaluation script for InternVL3.5 (GitHub format) on InfoVQA
"""
import json
import os
import torch
from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess
from transformers import AutoTokenizer
from PIL import Image
from tqdm import tqdm

def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def anls_metric(pred, gt, threshold=0.5):
    """Compute ANLS metric"""
    dist = levenshtein_distance(pred.lower(), gt.lower())
    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 1.0
    nl = dist / max_len
    return 1.0 - nl if nl < threshold else 0.0

def main():
    # Configuration
    model_name = "OpenGVLab/InternVL3_5-1B"
    data_path = os.path.expanduser("~/data/internvl/infographicsvqa")
    output_file = "results/infovqa_internvl35_1b_github_results.json"
    
    os.makedirs("results", exist_ok=True)
    
    print(f"Loading model: {model_name}")
    model = InternVLChatModel.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval().cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    
    print(f"Model loaded! Image size: {image_size}, Use thumbnail: {use_thumbnail}")
    
    # Load data
    print(f"Loading InfoVQA data from: {data_path}")
    test_file = os.path.join(data_path, 'val.jsonl')
    test_data = []
    with open(test_file, 'r') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    
    print(f"Processing {len(test_data)} samples...")
    results = []
    
    transform = build_transform(is_train=False, input_size=image_size)
    
    for item in tqdm(test_data):
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
        
        results.append({
            "questionId": question_id,
            "answer": response.strip(),
            "annotation": gt_answer
        })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total samples: {len(results)}")
    
    # Calculate ANLS metric
    print("\nCalculating ANLS metric...")
    anls_scores = []
    for result in results:
        pred = result['answer']
        gt_answers = result['annotation'] if isinstance(result['annotation'], list) else [result['annotation']]
        score = max([anls_metric(pred, gt) for gt in gt_answers])
        anls_scores.append(score)
    
    average_anls = sum(anls_scores) / len(anls_scores) if anls_scores else 0
    print(f"\n{'='*50}")
    print(f"Average ANLS Score: {average_anls:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()

