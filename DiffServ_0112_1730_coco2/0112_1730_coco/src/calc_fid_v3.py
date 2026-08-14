import torch
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import os
import sys
from scipy import linalg

# --- 1. function to extract the statistics (mu, sigma) ---
def get_statistics(image_dir, device):
    fid = FrechetInceptionDistance(feature=2048).to(device)
    
    valid_extensions = ('.png', '.jpg', '.jpeg')
    images = []
    files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)])
    
    if not files:
        raise ValueError(f"❌ no images in the folder: {image_dir}")

    print(f"📦 loading {len(files)} images...")
    for filename in files:
        img = Image.open(os.path.join(image_dir, filename)).convert('RGB')
        img = img.resize((299, 299), Image.LANCZOS)
        images.append(np.array(img))

    # (N, H, W, C) -> (N, C, H, W)
    imgs_tensor = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2).to(device)
    
    fid.update(imgs_tensor, real=True)
    
    # compute the statistics
    mu = (fid.real_features_sum / fid.real_features_num_samples).cpu().numpy()
    num_samples = fid.real_features_num_samples.item()
    features_cov_sum = fid.real_features_cov_sum.cpu().numpy()
    sigma = (features_cov_sum / num_samples) - np.outer(mu, mu)
    
    return mu, sigma

# --- 2. function to compute FID from the two statistics ---
def calculate_fid_from_stats(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    
    # 1. matrix multiplication
    cov_dot = sigma1.dot(sigma2)
    
    # 2. compute the matrix square root (disp argument removed)
    covmean = linalg.sqrtm(cov_dot)
    
    # 3. remove imaginary parts caused by numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 4. apply the FID formula
    # Tr(sigma1 + sigma2 - 2*covmean)
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# --- 3. main execution section ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if len(sys.argv) < 3:
        print("usage:")
        print("  1. save real-image statistics: python calc_fid_v3.py --save ./real_images real_stats.npz")
        print("  2. compare with generated images:   python calc_fid_v3.py --compare real_stats.npz ./fake_images")
        return

    mode = sys.argv[1]

    # mode 1: save statistics
    if mode == "--save":
        src_dir = sys.argv[2]
        out_file = sys.argv[3]
        mu, sigma = get_statistics(src_dir, device)
        np.savez(out_file, mu=mu, sigma=sigma)
        print(f"✅ statistics saved: {out_file}")

    # mode 2: compute the comparison
    elif mode == "--compare":
        real_stats_path = sys.argv[2]
        fake_dir = sys.argv[3]
        
        # load the real-image statistics
        print(f"📖 loading: {real_stats_path}")
        real_data = np.load(real_stats_path)
        mu_r, sigma_r = real_data['mu'], real_data['sigma']
        
        # extract the generated-image statistics on the fly
        print(f"📊 analyzing the generated images: {fake_dir}")
        mu_f, sigma_f = get_statistics(fake_dir, device)
        
        # compute FID
        score = calculate_fid_from_stats(mu_r, sigma_r, mu_f, sigma_f)
        print(f"\n🔥 final FID Score: {score:.4f}")

if __name__ == "__main__":
    main()