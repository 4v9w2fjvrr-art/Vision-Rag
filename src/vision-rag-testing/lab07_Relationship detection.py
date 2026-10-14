from pathlib import Path
import json
import math
import torch
from ultralytics import YOLO
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = Path("data/images/alexander-lunyov-nlY-pr6Fr1w-unsplash.jpg")

OUTPUT_DIR = Path("outputs/lab07")
OUTPUT_JSON = OUTPUT_DIR / "scene_relationships.json"

OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

OBJECT_CONFIDENCE_THRESHOLD = 0.20
OCR_CONFIDENCE_THRESHOLD = 0.60

YOLO_MODEL = "yolo11s.pt"


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
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
# LOAD YOLO
# ============================================================

print("\nLoading YOLO...")

yolo_model = YOLO(YOLO_MODEL)


# ============================================================
# RUN OBJECT DETECTION
# ============================================================

print("Running object detection...")

yolo_results = yolo_model.predict(
    source=str(IMAGE_PATH),
    conf=OBJECT_CONFIDENCE_THRESHOLD,
    imgsz=1280,
    device=device,
    verbose=False,
)

yolo_result = yolo_results[0]


# ============================================================
# EXTRACT OBJECTS
# ============================================================

objects = []

for index, box in enumerate(
    yolo_result.boxes
):

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

    center_x = (
        x1 + x2
    ) / 2

    center_y = (
        y1 + y2
    ) / 2

    objects.append(
        {
            "object_id": index + 1,
            "label": label,
            "confidence": round(
                confidence,
                3
            ),
            "bounding_box": {
                "x1": round(x1),
                "y1": round(y1),
                "x2": round(x2),
                "y2": round(y2),
            },
            "center": {
                "x": round(center_x),
                "y": round(center_y),
            }
        }
    )


# ============================================================
# LOAD docTR
# ============================================================

print("\nLoading docTR...")

ocr_model = ocr_predictor(
    pretrained=True
)


# ============================================================
# RUN OCR
# ============================================================

print("Running OCR...")

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


# ============================================================
# EXTRACT OCR TEXT
# ============================================================

texts = []

text_id = 1

for block in page["blocks"]:

    for line in block["lines"]:

        for word in line["words"]:

            confidence = float(
                word["confidence"]
            )

            # Filter weak OCR
            if (
                confidence
                < OCR_CONFIDENCE_THRESHOLD
            ):
                continue

            text = word["value"]

            geometry = word["geometry"]

            (
                x1_norm,
                y1_norm
            ), (
                x2_norm,
                y2_norm
            ) = geometry

            x1 = int(
                x1_norm * page_width
            )

            y1 = int(
                y1_norm * page_height
            )

            x2 = int(
                x2_norm * page_width
            )

            y2 = int(
                y2_norm * page_height
            )

            center_x = (
                x1 + x2
            ) / 2

            center_y = (
                y1 + y2
            ) / 2

            texts.append(
                {
                    "text_id": text_id,
                    "text": text,
                    "confidence": round(
                        confidence,
                        3
                    ),
                    "bounding_box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
                    "center": {
                        "x": round(
                            center_x
                        ),
                        "y": round(
                            center_y
                        ),
                    }
                }
            )

            text_id += 1


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def calculate_distance(
    center_a,
    center_b
):

    dx = (
        center_a["x"]
        - center_b["x"]
    )

    dy = (
        center_a["y"]
        - center_b["y"]
    )

    return math.sqrt(
        dx ** 2 + dy ** 2
    )


# ============================================================
# OBJECT ↔ OBJECT RELATIONSHIPS
# ============================================================

object_relationships = []

for i in range(
    len(objects)
):

    for j in range(
        i + 1,
        len(objects)
    ):

        object_a = objects[i]
        object_b = objects[j]

        distance = calculate_distance(
            object_a["center"],
            object_b["center"]
        )

        # Simple first version:
        # closer than 600 pixels
        if distance < 600:

            relationship = {
                "type": "near",
                "source": {
                    "id": object_a[
                        "object_id"
                    ],
                    "label": object_a[
                        "label"
                    ],
                },
                "target": {
                    "id": object_b[
                        "object_id"
                    ],
                    "label": object_b[
                        "label"
                    ],
                },
                "distance_pixels": round(
                    distance
                )
            }

            object_relationships.append(
                relationship
            )


# ============================================================
# OBJECT ↔ TEXT RELATIONSHIPS
# ============================================================

object_text_relationships = []

for obj in objects:

    for text in texts:

        distance = calculate_distance(
            obj["center"],
            text["center"]
        )

        if distance < 500:

            relationship = {
                "type": "text_near_object",

                "object": {
                    "id": obj[
                        "object_id"
                    ],
                    "label": obj[
                        "label"
                    ],
                },

                "text": {
                    "id": text[
                        "text_id"
                    ],
                    "value": text[
                        "text"
                    ],
                },

                "distance_pixels": round(
                    distance
                )
            }

            object_text_relationships.append(
                relationship
            )


# ============================================================
# PRINT OBJECTS
# ============================================================

print(
    "\n======================================"
)
print("OBJECTS")
print(
    "======================================"
)

for obj in objects:

    print(
        f"{obj['object_id']} "
        f"{obj['label']} "
        f"({obj['confidence']})"
    )


# ============================================================
# PRINT TEXT
# ============================================================

print(
    "\n======================================"
)
print("TRUSTED OCR TEXT")
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
# PRINT OBJECT RELATIONSHIPS
# ============================================================

print(
    "\n======================================"
)
print("OBJECT RELATIONSHIPS")
print(
    "======================================"
)

for rel in object_relationships:

    print(
        f"{rel['source']['label']} "
        f"{rel['source']['id']} "
        f"is near "
        f"{rel['target']['label']} "
        f"{rel['target']['id']} "
        f"(distance={rel['distance_pixels']})"
    )


# ============================================================
# PRINT OBJECT-TEXT RELATIONSHIPS
# ============================================================

print(
    "\n======================================"
)
print("OBJECT ↔ TEXT RELATIONSHIPS")
print(
    "======================================"
)

for rel in object_text_relationships:

    print(
        f"{rel['object']['label']} "
        f"{rel['object']['id']} "
        f"is near text "
        f"'{rel['text']['value']}' "
        f"(distance={rel['distance_pixels']})"
    )


# ============================================================
# SAVE COMPLETE SCENE JSON
# ============================================================

scene_data = {

    "image": str(
        IMAGE_PATH
    ),

    "objects": objects,

    "texts": texts,

    "object_relationships":
        object_relationships,

    "object_text_relationships":
        object_text_relationships,
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        scene_data,
        file,
        indent=4,
        ensure_ascii=False
    )


print(
    "\nScene JSON saved:"
)
print(
    OUTPUT_JSON
)
