import json
import random

def extract_coco_prompts(json_path, count=10):
    """
    Randomly extracts captions from a COCO annotation file.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # extract only the captions from the annotations list
    all_annotations = data.get('annotations', [])
    
    # randomly sample the desired number
    if len(all_annotations) >= count:
        sampled_annos = random.sample(all_annotations, count)
    else:
        sampled_annos = all_annotations

    # return only the caption text as a list
    return [anno['caption'] for anno in sampled_annos]

# --- 실행 예제 ---

# 1. get 10 prompts from the file
json_file = 'annotations/captions_val2014.json'
try:
    prompts = extract_coco_prompts(json_file, 100)

    # 2. print and verify the result
    print(f"✅ Successfully fetched {len(prompts)} prompts.\n")
    
    for i, p in enumerate(prompts, 1):
        print(f"Prompt {i}: {p}")

except FileNotFoundError:
    print(f"❌ Could not find the file: {json_file}")
except Exception as e:
    print(f"❌ An error occurred: {e}")
