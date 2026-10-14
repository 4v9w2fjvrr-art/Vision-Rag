from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)


# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = Path(
    "data/images/alexander-lunyov-nlY-pr6Fr1w-unsplash.jpg"
)

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

TEXT_LABELS = [
    "tree",
    "house",
    "barricade",
    "road barrier",
    "person",
    "bicycle",
    "car",
]

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25


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
# LOAD MODEL + PROCESSOR
# ============================================================

print("Loading Grounding DINO...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)

model = (
    AutoModelForZeroShotObjectDetection
    .from_pretrained(
        MODEL_ID
    )
)

model = model.to(
    device
)

model.eval()


# ============================================================
# PREPARE TEXT PROMPTS
# ============================================================

# Current Hugging Face API accepts a list of labels.
# This represents the concepts we want the model to search for.

text_labels = [
    TEXT_LABELS
]


# ============================================================
# PREPROCESS
# ============================================================

inputs = processor(
    images=image,
    text=text_labels,
    return_tensors="pt",
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# ============================================================
# RUN MODEL
# ============================================================

print("Running open-vocabulary detection...")

with torch.no_grad():

    outputs = model(
        **inputs
    )


# ============================================================
# POST PROCESS
# ============================================================

results = (
    processor
    .post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[
            image.size[::-1]
        ],
    )
)

result = results[0]


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n======================================")
print("OPEN-VOCABULARY DETECTIONS")
print("======================================")

detections = []

for index, (
    box,
    score,
    label
) in enumerate(
    zip(
        result["boxes"],
        result["scores"],
        result["labels"],
    )
):

    x1, y1, x2, y2 = (
        box.tolist()
    )

    detection = {
        "object_id": index + 1,
        "label": label,
        "confidence": round(
            float(score),
            3
        ),
        "bounding_box": {
            "x1": round(x1),
            "y1": round(y1),
            "x2": round(x2),
            "y2": round(y2),
        }
    }

    detections.append(
        detection
    )


for item in detections:

    print(
        f"\nObject ID : "
        f"{item['object_id']}"
    )

    print(
        f"Label     : "
        f"{item['label']}"
    )

    print(
        f"Confidence: "
        f"{item['confidence']}"
    )

    print(
        f"Box       : "
        f"{item['bounding_box']}"
    )


print(
    "\nTotal detections:",
    len(detections)
)
