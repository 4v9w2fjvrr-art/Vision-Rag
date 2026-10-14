from pathlib import Path

import torch
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = Path(
    "data/images/alexander-lunyov-nlY-pr6Fr1w-unsplash.jpg"
)

MODEL_NAME = "yolo11s.pt"

CONFIDENCE_THRESHOLD = 0.20
DUPLICATE_IOU_THRESHOLD = 0.50


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
# IOU FUNCTION
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
# LOAD YOLO
# ============================================================

print("Loading YOLO...")

model = YOLO(
    MODEL_NAME
)


# ============================================================
# RUN DETECTION
# ============================================================

print("Running detection...")

results = model.predict(
    source=str(IMAGE_PATH),
    conf=CONFIDENCE_THRESHOLD,
    imgsz=1280,
    device=device,
    verbose=False,
)

result = results[0]


# ============================================================
# RAW DETECTIONS
# ============================================================

raw_objects = []

for index, box in enumerate(
    result.boxes
):

    class_id = int(
        box.cls.item()
    )

    confidence = float(
        box.conf.item()
    )

    label = result.names[
        class_id
    ]

    x1, y1, x2, y2 = (
        box.xyxy[0].tolist()
    )

    raw_objects.append(
        {
            "object_id": index + 1,
            "label": label,
            "confidence": confidence,
            "bounding_box": {
                "x1": round(x1),
                "y1": round(y1),
                "x2": round(x2),
                "y2": round(y2),
            }
        }
    )


# ============================================================
# SORT BY CONFIDENCE
# ============================================================

raw_objects.sort(
    key=lambda x: x["confidence"],
    reverse=True
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

clean_objects = []

for candidate in raw_objects:

    duplicate = False

    for existing in clean_objects:

        # Only compare same class
        if (
            candidate["label"]
            != existing["label"]
        ):
            continue

        iou = calculate_iou(
            candidate["bounding_box"],
            existing["bounding_box"]
        )

        if (
            iou
            >= DUPLICATE_IOU_THRESHOLD
        ):
            duplicate = True

            print(
                f"Duplicate removed: "
                f"{candidate['label']} "
                f"conf={candidate['confidence']:.3f} "
                f"IoU={iou:.3f}"
            )

            break

    if not duplicate:
        clean_objects.append(
            candidate
        )


# ============================================================
# REASSIGN CLEAN IDS
# ============================================================

for index, obj in enumerate(
    clean_objects
):
    obj["object_id"] = index + 1


# ============================================================
# PRINT RAW OBJECTS
# ============================================================

print("\n======================================")
print("RAW DETECTIONS")
print("======================================")

for obj in raw_objects:

    print(
        f"{obj['label']} "
        f"conf={obj['confidence']:.3f}"
    )


print(
    "\nRaw count:",
    len(raw_objects)
)


# ============================================================
# PRINT CLEAN OBJECTS
# ============================================================

print("\n======================================")
print("CLEAN DETECTIONS")
print("======================================")

for obj in clean_objects:

    print(
        f"Object {obj['object_id']} "
        f"{obj['label']} "
        f"conf={obj['confidence']:.3f}"
    )


print(
    "\nClean count:",
    len(clean_objects)
)
