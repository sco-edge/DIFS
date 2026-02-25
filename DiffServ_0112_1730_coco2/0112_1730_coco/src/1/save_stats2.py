import torch
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import os

def save_real_statistics(real_dir, save_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # 1. FID 객체 생성 (reset_real_features=False로 설정하여 특징값을 유지하도록 함)
    # 최신 버전에서는 특징 추출 후 직접 계산을 위해 아래 방식을 사용합니다.
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # 2. 이미지 로드 및 전처리
    images = []
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = sorted([f for f in os.listdir(real_dir) if f.lower().endswith(valid_extensions)])
    
    if not image_files:
        print(f"❌ '{real_dir}' 폴더에 이미지가 없습니다.")
        return

    for filename in image_files:
        img = Image.open(os.path.join(real_dir, filename)).convert('RGB')
        img = img.resize((299, 299), Image.LANCZOS)
        images.append(np.array(img))

    # (N, H, W, C) -> (N, C, H, W) 변환 및 텐서화
    imgs_tensor = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2).to(device)

    # 3. 특징 추출
    print(f"📊 {len(image_files)}개의 이미지에서 특징 추출 중...")
    # real=True로 업데이트하면 내부적으로 real_features_sum과 관련 행렬이 업데이트됩니다.
    fid.update(imgs_tensor, real=True)

    # 4. mu와 sigma 직접 계산
    # 최신 torchmetrics는 인스턴스 내부에 저장된 값들을 이용하여 수치적으로 접근합니다.
    # 안전하게 접근하기 위해 필터를 통과한 특징을 직접 가져오는 방식 대신 
    # 통계량을 직접 산출합니다.
    
    # 특징 벡터 합산과 개수를 이용한 평균(mu) 계산
    mu = (fid.real_features_sum / fid.real_features_num_samples).cpu().numpy()
    print(f"✅ mu: {mu}")

    # 공분산(sigma) 계산
    # (E[XX^T] - E[X]E[X]^T) 공식을 사용하여 sigma를 구합니다.
    num_samples = fid.real_features_num_samples.item()
    features_cov_sum = fid.real_features_cov_sum.cpu().numpy()
    
    # 공분산 행렬 산출
    sigma = (features_cov_sum / num_samples) - np.outer(mu, mu)
    print(f"✅ sigma: {sigma}")

    # 5. .npz 파일로 저장
    np.savez(save_path, mu=mu, sigma=sigma)
    print(f"✅ 통계 데이터가 성공적으로 저장되었습니다: {save_path}")

if __name__ == "__main__":
    save_real_statistics("./real_images", "real_stats.npz")