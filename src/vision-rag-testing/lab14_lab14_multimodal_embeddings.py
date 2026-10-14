from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel


# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = Path(
    "data/images/alexander-lunyov-nlY-pr6Fr1w-unsplash.jpg"
)

MODEL_ID = "google/siglip-base-patch16-224"


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    device = "cuda"

elif torch.backends.mps.is_available():
    device = "mps"

else:
    device = "cpu"


print("Device:", device)


# ============================================================
# CHECK IMAGE
# ============================================================

if not IMAGE_PATH.exists():

    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

print("Loading image...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# ============================================================
# LOAD SIGLIP
# ============================================================

print("Loading SigLIP processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)


print("Loading SigLIP model...")

model = AutoModel.from_pretrained(
    MODEL_ID
)

model = model.to(device)

model.eval()


# ============================================================
# TEXT QUERIES
# ============================================================

texts = [
    "a person riding a bicycle",
    "a car parked on a street",
    "a cat sitting on a sofa",
    "an airplane flying in the sky",
]


# ============================================================
# PREPROCESS IMAGE + TEXT
# ============================================================

print("Preparing image and text...")

inputs = processor(
    text=texts,
    images=image,

    # Important for SigLIP text processing
    padding="max_length",

    return_tensors="pt",
)


inputs = {
    key: value.to(device)
    if hasattr(value, "to")
    else value

    for key, value in inputs.items()
}


# ============================================================
# RUN SIGLIP
# ============================================================

print("Generating multimodal embeddings...")

with torch.no_grad():

    outputs = model(
        **inputs
    )


# ============================================================
# EMBEDDINGS
# ============================================================

image_embedding = outputs.image_embeds
text_embeddings = outputs.text_embeds


print("\n======================================")
print("EMBEDDING SHAPES")
print("======================================")

print(
    "Image embedding shape:",
    image_embedding.shape
)

print(
    "Text embeddings shape:",
    text_embeddings.shape
)


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

image_embedding = (
    image_embedding
    / image_embedding.norm(
        dim=-1,
        keepdim=True
    )
)


text_embeddings = (
    text_embeddings
    / text_embeddings.norm(
        dim=-1,
        keepdim=True
    )
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

similarities = (
    image_embedding
    @ text_embeddings.T
)


# ============================================================
# PRINT SIMILARITY RESULTS
# ============================================================

print("\n======================================")
print("IMAGE ↔ TEXT SIMILARITY")
print("======================================")


for text, score in zip(
    texts,
    similarities[0]
):

    print(
        f"{score.item():.4f}"
        f"  ->  "
        f"{text}"
    )


# ============================================================
# BEST MATCH
# ============================================================

best_index = int(
    similarities[0]
    .argmax()
    .item()
)


print("\n======================================")
print("BEST MATCH")
print("======================================")

print(
    texts[
        best_index
    ]
)

print(
    "Similarity:",
    round(
        similarities[
            0,
            best_index
        ].item(),
        4
    )
)
