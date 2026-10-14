from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)


# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = Path(
    "data/images/alexander-lunyov-nlY-pr6Fr1w-unsplash.jpg"
)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


# ============================================================
# CHECK IMAGE
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )


# ============================================================
# DEVICE
# ============================================================

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# LOAD IMAGE
# ============================================================

print("Loading image...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)


# ============================================================
# LOAD VLM
# ============================================================

print("Loading Qwen3-VL model...")

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
)

model = model.to(device)

model.eval()


# ============================================================
# QUESTION / PROMPT
# ============================================================

question = """
Analyse this image carefully.

Tell me:

1. What important objects are visible?
2. How many people are visible?
3. What are the people doing?
4. What vehicles are visible?
5. Are there trees, houses, road barriers or signs?
6. What text or numbers are visible?
7. Explain the important relationships between the objects.
8. Give a short overall description of the scene.

Do not invent details that are not clearly visible.
"""


# ============================================================
# MULTIMODAL MESSAGE
# ============================================================

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {
                "type": "text",
                "text": question,
            },
        ],
    }
]


# ============================================================
# PREPARE INPUT
# ============================================================

print("Preparing multimodal input...")

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = {
    key: value.to(device)
    if hasattr(value, "to")
    else value
    for key, value in inputs.items()
}


# ============================================================
# RUN VLM
# ============================================================

print("Running VLM scene understanding...")

with torch.no_grad():

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False,
    )


# ============================================================
# REMOVE INPUT TOKENS
# ============================================================

input_length = inputs[
    "input_ids"
].shape[-1]

generated_only = generated_ids[
    :,
    input_length:
]


# ============================================================
# DECODE ANSWER
# ============================================================

answer = processor.batch_decode(
    generated_only,
    skip_special_tokens=True,
)[0]


# ============================================================
# PRINT
# ============================================================

print("\n======================================")
print("VLM SCENE UNDERSTANDING")
print("======================================")

print(answer)
