from pathlib import Path
import json

import torch
from PIL import Image
from ultralytics import YOLO
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
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

OUTPUT_DIR = Path("outputs/lab11")
OUTPUT_JSON = OUTPUT_DIR / "merged_scene.json"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

YOLO_MODEL = "yolo11s.pt"

GROUNDING_DINO_MODEL = (
    "IDEA-Research/grounding-dino-tiny"
)

OPEN_VOCAB_LABELS = [
    "tree",
    "house",
    "barricade",
    "road barrier",
    "person",
    "bicycle",
    "car",
]

YOLO_CONF = 0.20
DINO_BOX_THRESHOLD = 0.30
DINO_TEXT_THRESHOLD = 0.25
OCR_CONFIDENCE_THRESHOLD = 0.60

MERGE_IOU_THRESHOLD = 0.50


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
# IOU
# ============================================================

def calculate_iou(box_a, box_b):

    x_left = max(
        box_a["x1"],
        box_b["x1"]
    )

    y_top = max(
        box_a["y1"],
        box_b["y1"]
    )

    x_right = min(
        box_a["x2"],
        box_b["x2"]
    )

    y_bottom = min(
        box_a["y2"],
        box_b["y2"]
    )

    if (
        x_right <= x_left
        or y_bottom <= y_top
    ):
        return 0.0

    intersection = (
        x_right - x_left
    ) * (
        y_bottom - y_top
    )

    area_a = (
        box_a["x2"] - box_a["x1"]
    ) * (
        box_a["y2"] - box_a["y1"]
    )

    area_b = (
        box_b["x2"] - box_b["x1"]
    ) * (
        box_b["y2"] - box_b["y1"]
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    return intersection / union


# ============================================================
# YOLO DETECTION
# ============================================================

print("\nRunning YOLO...")

yolo = YOLO(
    YOLO_MODEL
)

yolo_results = yolo.predict(
    source=str(IMAGE_PATH),
    conf=YOLO_CONF,
    imgsz=1280,
    device=device,
    verbose=False,
)

yolo_result = yolo_results[0]

yolo_objects = []

for box in yolo_result.boxes:

    class_id = int(
        box.cls.item()
    )

    confidence = float(
        box.conf.item()
    )

    label = yolo_result.names[
        class_id
    ]

    x1, y1, x2, y2 = (
        box.xyxy[0].tolist()
    )

    yolo_objects.append(
        {
            "label": label,
            "confidence": round(
                confidence,
                3
            ),
            "source": "yolo",
            "bounding_box": {
                "x1": round(x1),
                "y1": round(y1),
                "x2": round(x2),
                "y2": round(y2),
            }
        }
    )


# ============================================================
# GROUNDING DINO
# ============================================================

print("Running Grounding DINO...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")

processor = AutoProcessor.from_pretrained(
    GROUNDING_DINO_MODEL
)

dino = (
    AutoModelForZeroShotObjectDetection
    .from_pretrained(
        GROUNDING_DINO_MODEL
    )
)

dino = dino.to(
    device
)

dino.eval()

text_labels = [
    OPEN_VOCAB_LABELS
]

inputs = processor(
    images=image,
    text=text_labels,
    return_tensors="pt",
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

with torch.no_grad():

    outputs = dino(
        **inputs
    )

results = (
    processor
    .post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=DINO_BOX_THRESHOLD,
        text_threshold=DINO_TEXT_THRESHOLD,
        target_sizes=[
            image.size[::-1]
        ],
    )
)

dino_result = results[0]

dino_objects = []

labels = (
    dino_result.get("text_labels")
    or dino_result.get("labels")
)

for box, score, label in zip(
    dino_result["boxes"],
    dino_result["scores"],
    labels,
):

    x1, y1, x2, y2 = (
        box.tolist()
    )

    dino_objects.append(
        {
            "label": str(label),
            "confidence": round(
                float(score),
                3
            ),
            "source": "grounding_dino",
            "bounding_box": {
                "x1": round(x1),
                "y1": round(y1),
                "x2": round(x2),
                "y2": round(y2),
            }
        }
    )


# ============================================================
# OCR
# ============================================================

print("Running docTR OCR...")

ocr_model = ocr_predictor(
    pretrained=True
)

document = DocumentFile.from_images(
    str(IMAGE_PATH)
)

ocr_result = ocr_model(
    document
)

exported = ocr_result.export()

page = exported["pages"][0]

page_height = page["dimensions"][0]
page_width = page["dimensions"][1]

texts = []

for block in page["blocks"]:

    for line in block["lines"]:

        for word in line["words"]:

            confidence = float(
                word["confidence"]
            )

            if (
                confidence
                < OCR_CONFIDENCE_THRESHOLD
            ):
                continue

            (
                x1_norm,
                y1_norm
            ), (
                x2_norm,
                y2_norm
            ) = word["geometry"]

            texts.append(
                {
                    "text": word["value"],
                    "confidence": round(
                        confidence,
                        3
                    ),
                    "bounding_box": {
                        "x1": int(
                            x1_norm
                            * page_width
                        ),
                        "y1": int(
                            y1_norm
                            * page_height
                        ),
                        "x2": int(
                            x2_norm
                            * page_width
                        ),
                        "y2": int(
                            y2_norm
                            * page_height
                        ),
                    }
                }
            )


# ============================================================
# MERGE YOLO + DINO
# ============================================================

print("Merging detections...")

all_objects = (
    yolo_objects
    + dino_objects
)

all_objects.sort(
    key=lambda x: x["confidence"],
    reverse=True
)

merged_objects = []

for candidate in all_objects:

    matched = False

    for existing in merged_objects:

        if (
            candidate["label"]
            != existing["label"]
        ):
            continue

        iou = calculate_iou(
            candidate[
                "bounding_box"
            ],
            existing[
                "bounding_box"
            ]
        )

        if (
            iou
            >= MERGE_IOU_THRESHOLD
        ):

            if (
                candidate["source"]
                not in existing[
                    "sources"
                ]
            ):
                existing[
                    "sources"
                ].append(
                    candidate["source"]
                )

            matched = True
            break

    if not matched:

        merged_objects.append(
            {
                "label":
                    candidate["label"],

                "confidence":
                    candidate[
                        "confidence"
                    ],

                "sources": [
                    candidate[
                        "source"
                    ]
                ],

                "bounding_box":
                    candidate[
                        "bounding_box"
                    ],
            }
        )


# ============================================================
# ASSIGN IDS
# ============================================================

for index, obj in enumerate(
    merged_objects
):

    obj["object_id"] = (
        index + 1
    )


for index, text in enumerate(
    texts
):

    text["text_id"] = (
        index + 1
    )


# ============================================================
# PRINT
# ============================================================

print(
    "\n======================================"
)
print("MERGED OBJECTS")
print(
    "======================================"
)

for obj in merged_objects:

    print(
        f"\nObject ID : "
        f"{obj['object_id']}"
    )

    print(
        f"Label     : "
        f"{obj['label']}"
    )

    print(
        f"Confidence: "
        f"{obj['confidence']}"
    )

    print(
        f"Sources   : "
        f"{obj['sources']}"
    )

    print(
        f"Box       : "
        f"{obj['bounding_box']}"
    )


print(
    "\n======================================"
)
print("OCR TEXT")
print(
    "======================================"
)

for text in texts:

    print(
        f"{text['text_id']} "
        f"{text['text']} "
        f"({text['confidence']})"
    )


# ============================================================
# SAVE SCENE JSON
# ============================================================

scene = {

    "image": str(
        IMAGE_PATH
    ),

    "objects":
        merged_objects,

    "texts":
        texts,
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        scene,
        file,
        indent=4,
        ensure_ascii=False
    )


print(
    "\nMerged scene saved:"
)

print(
    OUTPUT_JSON
)
