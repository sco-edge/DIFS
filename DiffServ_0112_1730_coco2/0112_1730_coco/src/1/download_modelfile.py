from huggingface_hub import hf_hub_download
import os

def download_sd_model(save_dir="/tmp/model"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Stable Diffusion v1.2 아카이브 레포지토리 정보
    repo_id = "CompVis/stable-diffusion-v-1-1-archive"
    filename = "sd-v1-2.ckpt" # 또는 해당 레포에 있는 정확한 파일명
    
    print(f"🚀 {filename} 다운로드 시작...")
    
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=save_dir,
        local_dir_use_symlinks=False
    )
    
    print(f"✅ 다운로드 완료: {path}")

if __name__ == "__main__":
    download_sd_model()