import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

# load model and scheduler
model_id = "/data-cbs1/ubuntu/checkpoints/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")
pipeline.enable_attention_slicing()

# let's download an  image
#url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
#response = requests.get(url)
#low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
#low_res_img = low_res_img.resize((128, 128))
low_res_img = Image.open('tmp/tmp4fbiyqkq.png')
w, h = low_res_img.size
low_res_img = low_res_img.resize((h//8, w//8))

prompt = "high quality, detailed"

upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save("upsampled.png")