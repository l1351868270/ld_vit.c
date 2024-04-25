from transformers import ViTImageProcessor, ViTForImageClassification
import torch
import numpy as np
from PIL import Image
import requests
from torchvision import transforms

import struct

# transform = transforms.Compose([transforms.ToTensor()])
# img_convert_to_tensor0 = transform(np.array(Image.open("./0.jpg")))
# img_convert_to_tensor1 = transform(np.array(Image.open("./1.jpg")))

image = Image.open("./0.jpg")
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")
inputs = inputs["pixel_values"]

with open("image.bin", "wb") as file:
    file.write(struct.pack("4i", *[2, inputs.shape[1], inputs.shape[2], inputs.shape[3]]))
    file.write(inputs.detach().numpy().astype("float32").tobytes())
    file.write(inputs.detach().numpy().astype("float32").tobytes())