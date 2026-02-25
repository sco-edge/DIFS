from huggingface_hub import hf_hub_download
import os
import pathlib

def download_sd_model(save_dir="./model_checkpoints"):
    # 1. 권한 문제가 적은 현재 작업 디렉토리 하위로 기본 경로 변경 권장
    # 만약 꼭 /tmp/model을 써야 한다면, 해당 폴더의 권한을 미리 확인해야 합니다.
    abs_save_dir = os.path.abspath(save_dir)
    os.makedirs(abs_save_dir, exist_ok=True)
   
    repo_id = "CompVis/stable-diffusion-v-1-1-archive"
    filename = "sd-v1-2.ckpt"
   
    print(f"🚀 {filename} 다운로드 시작 (저장위치: {abs_save_dir})...")
   
    try:
        # 2. 경고를 발생시키는 local_dir_use_symlinks 제거
        # local_dir만 설정하면 최신 라이브러리 방식대로 바로 다운로드됩니다.
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=abs_save_dir
        )
        print(f"✅ 다운로드 완료: {path}")
       
    except PermissionError:
        print(f"❌ 권한 오류: '{abs_save_dir}'에 쓸 권한이 없습니다.")
        print("해결책: sudo 권한으로 실행하거나, 권한이 있는 다른 디렉토리로 변경하세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    # 로컬 경로인 ./models 에 저장하도록 호출
    download_sd_model(save_dir="./models")