import torch
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import os
import sys
from scipy import linalg

# --- 1. 통계량(mu, sigma) 추출 함수 ---
def get_statistics(image_dir, device):
    fid = FrechetInceptionDistance(feature=2048).to(device)
    
    valid_extensions = ('.png', '.jpg', '.jpeg')
    images = []
    files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)])
    
    if not files:
        raise ValueError(f"❌ 폴더에 이미지가 없습니다: {image_dir}")

    print(f"📦 {len(files)}개 이미지 로드 중...")
    for filename in files:
        img = Image.open(os.path.join(image_dir, filename)).convert('RGB')
        img = img.resize((299, 299), Image.LANCZOS)
        images.append(np.array(img))

    # (N, H, W, C) -> (N, C, H, W)
    imgs_tensor = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2).to(device)
    
    fid.update(imgs_tensor, real=True)
    
    # 통계량 산출
    mu = (fid.real_features_sum / fid.real_features_num_samples).cpu().numpy()
    num_samples = fid.real_features_num_samples.item()
    features_cov_sum = fid.real_features_cov_sum.cpu().numpy()
    sigma = (features_cov_sum / num_samples) - np.outer(mu, mu)
    
    return mu, sigma

# --- 2. 두 통계량으로 FID 계산하는 함수 ---
def calculate_fid_from_stats(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    
    # 1. 행렬 곱셈
    cov_dot = sigma1.dot(sigma2)
    
    # 2. 행렬 제곱근 계산 (disp 인자 제거)
    covmean = linalg.sqrtm(cov_dot)
    
    # 3. 수치적 불안정성으로 인한 허수 제거
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 4. FID 공식 적용
    # Tr(sigma1 + sigma2 - 2*covmean)
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# --- 3. 메인 실행부 ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if len(sys.argv) < 3:
        print("사용법:")
        print("  1. 실제 이미지 통계 저장: python calc_fid_v3.py --save ./real_images real_stats.npz")
        print("  2. 생성 이미지와 비교:   python calc_fid_v3.py --compare real_stats.npz ./fake_images")
        return

    mode = sys.argv[1]

    # 모드 1: 통계 저장
    if mode == "--save":
        src_dir = sys.argv[2]
        out_file = sys.argv[3]
        mu, sigma = get_statistics(src_dir, device)
        np.savez(out_file, mu=mu, sigma=sigma)
        print(f"✅ 통계 저장 완료: {out_file}")

    # 모드 2: 비교 계산
    elif mode == "--compare":
        real_stats_path = sys.argv[2]
        fake_dir = sys.argv[3]
        
        # 실제 이미지 통계 로드
        print(f"📖 로딩 중: {real_stats_path}")
        real_data = np.load(real_stats_path)
        mu_r, sigma_r = real_data['mu'], real_data['sigma']
        
        # 생성 이미지 통계 즉석 추출
        print(f"📊 생성 이미지 분석 중: {fake_dir}")
        mu_f, sigma_f = get_statistics(fake_dir, device)
        
        # FID 계산
        score = calculate_fid_from_stats(mu_r, sigma_r, mu_f, sigma_f)
        print(f"\n🔥 최종 FID Score: {score:.4f}")

if __name__ == "__main__":
    main()