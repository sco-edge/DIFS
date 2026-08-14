# Author: KB
# Date: 20251110
# Purpose: INFaaS does not come with supported models so I created this script to download compatible supported models from torchvision

import torch
import torchvision.models as models

model = models.resnet50(weights="IMAGENET1K_V1")
model.eval()
scripted = torch.jit.script(model)
scripted.save("resnet50_fp32.pt")
