import torch
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image

def get_real_images_fixed():
    save_dir = "./real_images"
    os.makedirs(save_dir, exist_ok=True)

    # 1. CIFAR10 데이터셋 로드 (강아지 레이블은 5번)
    # 인터넷 상황이 안 좋으면 여기서 시간이 좀 걸릴 수 있으나 404는 나지 않습니다.
    #dataset = torchvision.datasets.CIFar10(root='./data', train=True, download=True)
    # 기존 CIFAR10 부분을 아래로 교체
    dataset = torchvision.datasets.STL10(root='./data', split='train', download=True)
    # STL10에서 새(bird)의 레이블은 2번입니다.
    target_label = 3
    
    dog_count = 0
    print("🚀 내장 데이터셋에서 강아지 이미지 추출 중...")

    for img, label in dataset:
        if label == 5: # 2번이 dog입니다.
            # FID 계산을 위해 512x512로 확대 (원본은 32x32)
            img_resized = img.resize((512, 512), Image.LANCZOS)
            
            file_path = f"{save_dir}/real_{dog_count}.png"
            img_resized.save(file_path)
            dog_count += 1
            print(f"✅ [{dog_count}/10] 이미지 생성 완료")
            
        if dog_count >= 10:
            break

    print(f"\n📂 성공! '{save_dir}' 폴더에 10장의 이미지가 저장되었습니다.")

if __name__ == "__main__":
    get_real_images_fixed()
