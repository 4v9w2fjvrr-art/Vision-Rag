from pathlib import Path
import re,torch,cv2
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
IMAGE_PATH = "/Users/sriramapadmanabhan/vision-rag/data/images/alexander-lunyov-nlY-pr6Fr1w-unsplash.jpg"
OUTPUT_DIR = "outputs"
# Small pretrained YOLO model
MODEL_NAME = "yolo11s.pt"

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Device:", device)
# ============================================================
# CHECK INPUT IMAGE
# ============================================================
image_path = Path(IMAGE_PATH)
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path.resolve()}")
# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)
# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================
print("Loading YOLO model...")
model = YOLO(MODEL_NAME)

# ============================================================
# RUN OBJECT DETECTION
# ============================================================
print("Running object detection...")
#results = model(source=str(image_path),conf=0.25,verbose=False)
results = model.predict(source=str(IMAGE_PATH),conf=0.20,imgsz=1280,device=device,verbose=False,)

# ============================================================
# READ FIRST IMAGE RESULT
# ============================================================
result = results[0]
# ============================================================
# EXTRACT OBJECTS
# ============================================================
objects = []

for index, box in enumerate(result.boxes):
    class_id = int(box.cls.item())
    label = result.names[class_id]
    confidence = float(box.conf.item())
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    detected_object = {"object_id": index + 1,"label": label,"confidence": round(confidence, 3),"bounding_box": {"x1": round(x1),"y1": round(y1),"x2": round(x2),"y2": round(y2),}
    }
    objects.append(detected_object)
# ============================================================
# PRINT RESULTS
# ============================================================
print()
print("======================================")
print("DETECTED OBJECTS")
print("======================================")
if not objects:
    print("No objects detected.")

for obj in objects:
    print()
    print(f"Object ID : {obj['object_id']}")
    print(f"Label     : {obj['label']}")
    print(f"Confidence: {obj['confidence']}")
    print(f"Box       : {obj['bounding_box']}")


print()
print(f"Total objects detected: {len(objects)}")
# ============================================================
# SAVE ANNOTATED IMAGE
# ============================================================
annotated_image = result.plot()
print(annotated_image)
output_image = output_dir / "detected_objects.jpg"
# result.plot() returns a NumPy image.
# OpenCV is already installed as an Ultralytics dependency.
cv2.imwrite(str(output_image),annotated_image)
print()
print("Annotated image saved:")
print(output_image.resolve())
