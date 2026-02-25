"""
import torch
from safetensors.torch import save_file

# .ckpt 파일 로드
ckpt_path = "sd-v1-4.ckpt"
weights = torch.load(ckpt_path, map_location="cpu")

# 가중치 데이터 추출 (보통 "state_dict" 키 안에 들어있습니다)
if "state_dict" in weights:
    state_dict = weights["state_dict"]
else:
    state_dict = weights

# .safetensors로 저장
save_file(state_dict, "sd-v1-4.safetensors")
print("변환 완료!")
"""

import torch
from safetensors.torch import save_file
import os

ckpt_path = "sd-v1-4.ckpt" # 여기에 실제 경로를 입력하세요.

if not os.path.exists(ckpt_path):
    print(f"❌ 에러: '{ckpt_path}' 파일을 찾을 수 없습니다. 경로를 다시 확인하세요.")
    print(f"현재 작업 디렉토리: {os.getcwd()}")
else:
    print(f"🔄 파일을 불러오는 중: {ckpt_path}")
    weights = torch.load(ckpt_path, map_location="cpu", weights_only=False)
   
    state_dict = weights["state_dict"] if "state_dict" in weights else weights
   
    output_path = "sd-v1-4.safetensors"
    save_file(state_dict, output_path)
    print(f"✅ 변환 완료: {output_path}")