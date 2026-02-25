import json
import random

def extract_coco_prompts(json_path, count=10):
    """
    COCO 주석 파일에서 무작위로 캡션을 추출합니다.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # annotations 리스트에서 캡션들만 추출
    all_annotations = data.get('annotations', [])
    
    # 원하는 개수만큼 무작위 샘플링
    if len(all_annotations) >= count:
        sampled_annos = random.sample(all_annotations, count)
    else:
        sampled_annos = all_annotations

    # 캡션 텍스트만 리스트로 반환
    return [anno['caption'] for anno in sampled_annos]

# --- 실행 예제 ---

# 1. 파일에서 10개의 프롬프트 가져오기
json_file = 'annotations/captions_val2014.json'
try:
    prompts = extract_coco_prompts(json_file, 100)

    # 2. 결과 출력 및 확인
    print(f"✅ 성공적으로 {len(prompts)}개의 프롬프트를 가져왔습니다.\n")
    
    for i, p in enumerate(prompts, 1):
        print(f"Prompt {i}: {p}")

except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {json_file}")
except Exception as e:
    print(f"❌ 오류 발생: {e}")
