import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import numpy as np
import os
import sys

def load_images_from_folder(folder_path):
    images = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    
    # 폴더 내 모든 이미지 파일을 읽어서 리스트에 저장
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(valid_extensions):
            img = Image.open(os.path.join(folder_path, filename)).convert('RGB')
            # FID 계산을 위해 모든 이미지를 동일한 크기(299x299)로 리사이징
            img = img.resize((299, 299), Image.LANCZOS)
            img_array = np.array(img)
            images.append(img_array)
    
    # (N, H, W, C) -> (N, C, H, W) 형태로 변환 후 텐서 생성
    images_array = np.array(images).transpose(0, 3, 1, 2)
    return torch.from_numpy(images_array).type(torch.uint8)

def calculate_fid(real_path, fake_path):
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # FID 객체 생성 (Inception v3 모델의 2048차원 특징 벡터 사용)
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # 이미지 로드
    print(f"📦 Loading real images from: {real_path}")
    real_images = load_images_from_folder(real_path).to(device)
    
    print(f"📦 Loading fake images from: {fake_path}")
    fake_images = load_images_from_folder(fake_path).to(device)

    # 데이터 업데이트 (실제 이미지)
    fid.update(real_images, real=True)
    # 데이터 업데이트 (생성 이미지)
    fid.update(fake_images, real=False)

    # 최종 FID 점수 계산
    score = fid.compute()
    return score.item()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python calc_fid.py <real_folder_path> <fake_folder_path>")
    else:
        real_folder = sys.argv[1]
        fake_folder = sys.argv[2]
        
        fid_value = calculate_fid(real_folder, fake_folder)
        print(f"\n✅ FID Score: {fid_value:.4f}")
        print("(점수가 낮을수록 실제 이미지와 유사함을 의미합니다.)")