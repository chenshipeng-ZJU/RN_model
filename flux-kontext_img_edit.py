import os
# 必须先设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"

import torch
from diffusers import FluxKontextPipeline
from PIL import Image  # 用于读取图片

pipe = FluxKontextPipeline.from_pretrained("/dev/yowant2_cfs/tianyi/csp/VideoReproduce/models/flux1-kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

input_image = Image.open("/dev/yowant2_cfs/tianyi/csp/EEG-VisRL-GenLoop/data/IMG/S_0/raw_img/raw_img_1.jpg")

# image = pipe(
#   image=input_image,
#   prompt="Transform it into a cyberpunk style, main editing on Leifeng Pagoda, conveying a sense of excitement.",
#   guidance_scale=2.5,
#   num_inference_steps=50
# ).images[0]
image = pipe(
  image=input_image,
  prompt="Transform it into cartoon style, main editing on man, conveying a sense of happiness.",
  guidance_scale=2.5,
  num_inference_steps=50
).images[0]

# 保存结果（可选）
image.save("/dev/yowant2_cfs/tianyi/csp/EEG-VisRL-GenLoop/data/IMG/S_0/gen_img/gen_img_1_0.jpg")
