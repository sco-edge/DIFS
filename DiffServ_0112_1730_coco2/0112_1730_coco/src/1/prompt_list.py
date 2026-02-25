import json
import random

def extract_coco_prompts(json_path, count=10000):
    with open(json_path, 'r') as f:
        data = json.load(f)
   
    # 모든 캡션 문장만 추출
    all_captions = [ann['caption'] for ann in data['annotations']]
   
    # 중복을 피하기 위해 무작위로 count만큼 선택
    selected_prompts = random.sample(all_captions, count)
   
    return selected_prompts

# 사용 예시
# prompts = extract_coco_prompts('annotations/captions_val2014.json', 10000)

def run():
    prompts = extract_coco_prompts('annotations/captions_val2014.json', 10)
    print("[클라이언트] 요청 프롬프트 목록:")
    for i, p in enumerate(prompts):
        print(f"  - {i+1}: {p}")

if __name__ == "__main__":
    run()