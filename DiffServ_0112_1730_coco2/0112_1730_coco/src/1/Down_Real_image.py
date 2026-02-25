from datasets import load_dataset
from PIL import Image
import os

# 1. AFHQ 데이터셋 로드 (강아지, 고양이 이미지 포함)
# 'dogs'나 'cats' 카테고리를 선택할 수 있습니다.
dataset = load_dataset("huggan/afhq", split="train", streaming=True)

save_dir = "./real_images"
os.makedirs(save_dir, exist_ok=True)

targets = {"dog": 5, "cat": 5}  # 각각 5장씩 총 10장 목표
counts = {"dog": 0, "cat": 0}

print("실사 이미지 수집 시작...")
for example in dataset:
    label = example['label'] # 0: cat, 1: dog, 2: wildlife (데이터셋 사양에 따라 다름)
    
    # 레이블 판별 (AFHQ 기준: 0-cat, 1-dog)
    category = "cat" if label == 0 else "dog" if label == 1 else None
    
    if category and counts[category] < targets[category]:
        img = example['image'].convert("RGB").resize((512, 512), Image.LANCZOS)
        img.save(f"{save_dir}/{category}_{counts[category]}.png")
        counts[category] += 1
        print(f"저장 완료: {category}_{counts[category]}.png")
        
    if sum(counts.values()) >= 10:
        break