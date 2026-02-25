import torch
from diffusers import StableDiffusionPipeline
import os

def load_and_save_ckpt(ckpt_path, save_dir="./models/sd-v1-2-diffusers"):
    # 1. 저장 경로 생성
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"📦 모델 로드 중: {ckpt_path}")
    
    try:
        # 2. .ckpt 단일 파일로부터 파이프라인 로드
        # v1.2는 초기 모델이므로 가중치 타입(fp16)과 스케줄러 설정을 자동으로 맞춥니다.
        pipe = StableDiffusionPipeline.from_single_file(
            ckpt_path,
            torch_dtype=torch.float16,
            load_safety_checker=True # 안전 필터 포함 여부
        )
        
        # GPU 메모리가 있다면 할당 (속도 향상용, 필수는 아님)
        if torch.cuda.is_available():
            pipe.to("cuda")
        else:
            pipe.to("cpu")


        print(f"💾 Diffusers 형식으로 저장 중: {save_dir}")
        
        # 3. 모델 저장 (구조화된 폴더 형식으로 저장됨)
        # 이후에는 from_pretrained(save_dir)로 바로 로드 가능합니다.
        pipe.save_pretrained(save_dir, safe_serialization=True)
        
        print("✅ 저장 완료!")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("힌트: .ckpt 파일 경로가 정확한지, 혹은 필요한 의존성(omegaconf 등)이 설치되었는지 확인하세요.")

if __name__ == "__main__":
    # 다운로드 받은 ckpt 파일의 실제 경로를 입력하세요.
    target_ckpt = "./models/sd-v1-2.ckpt"
    
    if os.path.exists(target_ckpt):
        load_and_save_ckpt(target_ckpt)
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {target_ckpt}")