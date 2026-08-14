"""
import torch
from safetensors.torch import save_file

# load the .ckpt file
ckpt_path = "sd-v1-4.ckpt"
weights = torch.load(ckpt_path, map_location="cpu")

# extract the weight data (usually stored under the "state_dict" key)
if "state_dict" in weights:
    state_dict = weights["state_dict"]
else:
    state_dict = weights

# save as .safetensors
save_file(state_dict, "sd-v1-4.safetensors")
print("Conversion complete!")
"""

import torch
from safetensors.torch import save_file
import os

ckpt_path = "sd-v1-4.ckpt" # enter the actual path here.

if not os.path.exists(ckpt_path):
    print(f"❌ 에러: Could not find the file '{ckpt_path}'. Please check the path again.")
    print(f"current working directory: {os.getcwd()}")
else:
    print(f"🔄 loading the file: {ckpt_path}")
    weights = torch.load(ckpt_path, map_location="cpu", weights_only=False)
   
    state_dict = weights["state_dict"] if "state_dict" in weights else weights
   
    output_path = "sd-v1-4.safetensors"
    save_file(state_dict, output_path)
    print(f"✅ conversion complete: {output_path}")