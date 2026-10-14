from PIL import Image
from pathlib import Path
import torch

from transformers import ViTImageProcessor
from transformers import ViTModel

MODEL_NAME = "google/vit-base-patch16-224"
#IMAGE_PATH = Path("vision-rag/data/images/alexander-lunyov-nlY-pr6Fr1w-unsplash.jpg")
image = Image.open("data/images/alexander-lunyov-nlY-pr6Fr1w-unsplash.jpg").convert("RGB")

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = ViTModel.from_pretrained(MODEL_NAME)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = model.to(device)
model.eval()

inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

print("Input shape:", inputs["pixel_values"].shape)

with torch.no_grad():
    outputs = model(**inputs)

print("Last hidden state:", outputs.last_hidden_state.shape)
last_hidden_state = outputs.last_hidden_state

print("Number of tokens:", last_hidden_state.shape[1])
print("Embedding size:", last_hidden_state.shape[2])

cls_token = last_hidden_state[:, 0, :]
patch_tokens = last_hidden_state[:, 1:, :]

print("CLS token shape:", cls_token.shape)
print("Patch tokens shape:", patch_tokens.shape)
last_hidden_state = outputs.last_hidden_state


print("Number of tokens:", last_hidden_state.shape[1])
print("Embedding size:", last_hidden_state.shape[2])

cls_token = last_hidden_state[:, 0, :]
patch_tokens = last_hidden_state[:, 1:, :]

print("CLS token shape:", cls_token.shape)
print("Patch tokens shape:", patch_tokens.shape)
