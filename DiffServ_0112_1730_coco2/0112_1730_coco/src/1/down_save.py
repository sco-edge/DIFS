import torch
import os
from diffusers import StableDiffusionPipeline

# 1. 설정
model_url = "https://huggingface.co/CompVis/stable-diffusion-v-1-1-archive/resolve/main/sd-v1-2.ckpt"
save_path = "./models/sd-v1-2-diffusers"

# 2. 장치 판단 및 데이터 타입 설정
# CPU에서 float16은 지원되지 않는 연산이 많으므로 반드시 float32를 사용해야 합니다.
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

print(f"🚀 {device} 모드(Dtype: {torch_dtype})로 v1.2 로드 및 저장 시작...")

# 3. 모델 로드 (Hugging Face에서 다운로드 후 메모리에 올림)
pipe = StableDiffusionPipeline.from_single_file(
    model_url,
    torch_dtype=torch_dtype,
    use_safetensors=False,
    # CPU 메모리 절약을 위해 필요한 경우 설정
    low_cpu_mem_usage=(device == "cpu") 
)
pipe.to(device)

# 4. 모델 로컬 저장 (이후에는 인터넷 연결 없이 이 폴더만 있으면 됨)
if not os.path.exists(save_path):
    print(f"💾 모델을 로컬에 저장 중: {save_path}")
    pipe.save_pretrained(save_path, safe_serialization=True)
    print("✅ 저장 완료!")
else:
    print("ℹ️ 이미 동일한 경로에 모델이 저장되어 있습니다.")

# 5. 테스트 생성
print("🎨 이미지 생성 테스트 중...")
image = pipe("a professional photograph of an astronaut riding a horse").images[0]
image.save("test_v1_2.png")