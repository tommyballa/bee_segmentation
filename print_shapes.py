import torch
from PIL import Image
from lang_sam.models.gdino import GDINO
gdino = GDINO()
gdino.build_model()
image = Image.new('RGB', (100, 100))
texts = ["bee."]
inputs = gdino.processor(images=[image], text=texts, padding=True, return_tensors="pt")
for k, v in inputs.items():
    print(f"{k}: {v.shape}")
