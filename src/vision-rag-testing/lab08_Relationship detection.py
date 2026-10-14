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
OUTPUT_DIR = Path("outputs/lab08")
OUTPUT_JSON = OUTPUT_DIR / "spatial_relationships.json"

OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
YOLO_MODEL = "yolo11s.pt"
OBJECT_CONFIDENCE_THRESHOLD = 0.20
OCR_CONFIDENCE_THRESHOLD = 0.60
NEAR_DISTANCE = 500


# ============================================================
# DEVICE
# ============================================================

device = ("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def center(box):
    return {"x": (box["x1"] + box["x2"]) / 2,"y": (box["y1"] + box["y2"]) / 2,}


def distance(box_a, box_b):
    center_a = center(box_a)
    center_b = center(box_b)
    dx = (center_a["x"]- center_b["x"])
    dy = (center_a["y"]- center_b["y"])
    return math.sqrt(dx ** 2 + dy ** 2)


def intersection_area(box_a,box_b):
    x_left = max(box_a["x1"],box_b["x1"])
    y_top = max(box_a["y1"],box_b["y1"])
    x_right = min(box_a["x2"],box_b["x2"])
    y_bottom = min(box_a["y2"],box_b["y2"])
    if (x_right <= x_left or y_bottom <= y_top):
        return 0
    return (x_right - x_left) * (y_bottom - y_top)


def box_area(box):
    return (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])


def calculate_iou(box_a,box_b):
    intersection = intersection_area(box_a,box_b)
    if intersection == 0:
        return 0.0
    union = (box_area(box_a)+ box_area(box_b)- intersection)
    return intersection / union


def contains(outer_box,inner_box):
    return (inner_box["x1"] >= outer_box["x1"] and inner_box["y1"]>= outer_box["y1"] and inner_box["x2"]<= outer_box["x2"] and inner_box["y2"]<= outer_box["y2"])


def get_spatial_relationships(box_a,box_b):
    relationships = []
    center_a = center(box_a)
    center_b = center(box_b)
    dist = distance(box_a,box_b)
    iou = calculate_iou(box_a,box_b)

    # ----------------------------------------
    # NEAR
    # ----------------------------------------

    if dist < NEAR_DISTANCE:
        relationships.append("near")

    # ----------------------------------------
    # OVERLAP
    # ----------------------------------------

    if iou > 0:
        relationships.append("overlaps")

    # ----------------------------------------
    # LEFT / RIGHT
    # ----------------------------------------

    if (center_a["x"] < center_b["x"]):
        relationships.append("left_of")
    elif (center_a["x"] > center_b["x"]):
        relationships.append("right_of")

    # ----------------------------------------
    # ABOVE / BELOW
    # ----------------------------------------

    if (center_a["y"] < center_b["y"]):
        relationships.append("above")

    elif (center_a["y"] > center_b["y"]):
        relationships.append("below")

    # ----------------------------------------
    # CONTAINMENT
    # ----------------------------------------

    if contains(box_a,box_b):
        relationships.append("contains")

    if contains(box_b,box_a):
        relationships.append("inside")

    return {"relationships": relationships,"distance": round(dist),"iou": round(iou, 3),}


# ============================================================
# OBJECT DETECTION
# ============================================================

print("\nLoading YOLO...")
yolo = YOLO(YOLO_MODEL)
print("Running object detection...")
results = yolo.predict(source=str(IMAGE_PATH),conf=OBJECT_CONFIDENCE_THRESHOLD,imgsz=1280,device=device,verbose=False,)
result = results[0]
objects = []
for index, box in enumerate(result.boxes):
    class_id = int(box.cls.item())
    confidence = float(box.conf.item())
    label = result.names[class_id]
    x1, y1, x2, y2 = (box.xyxy[0].tolist())
    objects.append(
        {
            "object_id": index + 1,
            "label": label,
            "confidence": round(confidence,3),
            "bounding_box": {"x1": round(x1),"y1": round(y1),"x2": round(x2),"y2": round(y2),}
        }
    )


# ============================================================
# OCR
# ============================================================

print("\nLoading docTR...")

ocr_model = ocr_predictor(
    pretrained=True
)

document = DocumentFile.from_images(
    str(IMAGE_PATH)
)

print("Running OCR...")

ocr_result = ocr_model(
    document
)

exported = ocr_result.export()

page = exported["pages"][0]

page_height = page["dimensions"][0]
page_width = page["dimensions"][1]

texts = []

text_id = 1

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

            geometry = word[
                "geometry"
            ]

            (
                x1_norm,
                y1_norm
            ), (
                x2_norm,
                y2_norm
            ) = geometry

            box = {
                "x1": int(
                    x1_norm * page_width
                ),

                "y1": int(
                    y1_norm * page_height
                ),

                "x2": int(
                    x2_norm * page_width
                ),

                "y2": int(
                    y2_norm * page_height
                ),
            }

            texts.append(
                {
                    "text_id": text_id,

                    "text": word[
                        "value"
                    ],

                    "confidence": round(
                        confidence,
                        3
                    ),

                    "bounding_box": box,
                }
            )

            text_id += 1


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

        spatial = (
            get_spatial_relationships(
                object_a[
                    "bounding_box"
                ],
                object_b[
                    "bounding_box"
                ]
            )
        )

        object_relationships.append(
            {
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

                "relationships":
                    spatial[
                        "relationships"
                    ],

                "distance":
                    spatial[
                        "distance"
                    ],

                "iou":
                    spatial[
                        "iou"
                    ],
            }
        )


# ============================================================
# OBJECT ↔ TEXT RELATIONSHIPS
# ============================================================

object_text_relationships = []

for obj in objects:

    for text in texts:

        spatial = (
            get_spatial_relationships(
                obj[
                    "bounding_box"
                ],
                text[
                    "bounding_box"
                ]
            )
        )

        if (
            spatial[
                "distance"
            ]
            < NEAR_DISTANCE
            or spatial[
                "iou"
            ] > 0
        ):

            object_text_relationships.append(
                {
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

                    "relationships":
                        spatial[
                            "relationships"
                        ],

                    "distance":
                        spatial[
                            "distance"
                        ],

                    "iou":
                        spatial[
                            "iou"
                        ],
                }
            )


# ============================================================
# PRINT IMPORTANT RELATIONSHIPS
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
        f"\n{rel['source']['label']} "
        f"{rel['source']['id']}"

        f" ↔ "

        f"{rel['target']['label']} "
        f"{rel['target']['id']}"
    )

    print(
        "Relations:",
        rel[
            "relationships"
        ]
    )

    print(
        "Distance:",
        rel[
            "distance"
        ]
    )

    print(
        "IoU:",
        rel[
            "iou"
        ]
    )


print(
    "\n======================================"
)
print("OBJECT ↔ TEXT")
print(
    "======================================"
)

for rel in object_text_relationships:

    print(
        f"\n{rel['object']['label']} "
        f"{rel['object']['id']}"

        f" ↔ text "
        f"'{rel['text']['value']}'"
    )

    print(
        "Relations:",
        rel[
            "relationships"
        ]
    )

    print(
        "Distance:",
        rel[
            "distance"
        ]
    )

    print(
        "IoU:",
        rel[
            "iou"
        ]
    )


# ============================================================
# SAVE JSON
# ============================================================

output = {

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
        output,
        file,
        indent=4,
        ensure_ascii=False
    )


print(
    "\nSaved:"
)
print(
    OUTPUT_JSON
)
