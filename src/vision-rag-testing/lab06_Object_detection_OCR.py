from pathlib import Path
import json
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


# ============================================================
# CONFIGURATION
# ============================================================
IMAGE_PATH = Path("data/images/alexander-lunyov-nlY-pr6Fr1w-unsplash.jpg")
OUTPUT_DIR = Path("outputs/lab06")
OUTPUT_JSON = OUTPUT_DIR / "ocr_results.json"
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)


# ============================================================
# CHECK IMAGE
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# ============================================================
# LOAD OCR MODEL
# ============================================================

print("Loading docTR OCR model...")
model = ocr_predictor(pretrained=True)


# ============================================================
# LOAD IMAGE
# ============================================================
print("Loading image...")
document = DocumentFile.from_images(str(IMAGE_PATH))

# ============================================================
# RUN OCR
# ============================================================
print("Running OCR...")
result = model(document)


# ============================================================
# EXPORT RESULT
# ============================================================

exported = result.export()



# ============================================================
# IMAGE SIZE
# ============================================================

page = exported["pages"][0]
page_height = page["dimensions"][0]
page_width = page["dimensions"][1]


# ============================================================
# EXTRACT TEXT
# ============================================================

detected_text = []
text_id = 1
for block in page["blocks"]:
    for line in block["lines"]:
        for word in line["words"]:
            text = word["value"]
            confidence = float(word["confidence"])
            geometry = word["geometry"]

            # docTR geometry is normalized:
            # values are between 0 and 1

            (x1_norm, y1_norm), (x2_norm,y2_norm) = geometry


            # Convert normalized values
            # to actual image pixel coordinates

            x1 = int(x1_norm * page_width)
            y1 = int(y1_norm * page_height)
            x2 = int(x2_norm * page_width)
            y2 = int(y2_norm * page_height)
            item = {"text_id": text_id,"text": text,"confidence": round(confidence,3),"bounding_box": {"x1": x1,"y1": y1,"x2": x2,"y2": y2}}
            detected_text.append(item)
            text_id += 1


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n======================================")
print("DETECTED TEXT")
print("======================================")

for item in detected_text:
    print(f"\nText ID    : {item['text_id']}")
    print(f"Text       : {item['text']}")
    print(f"Confidence : {item['confidence']}")
    print(f"Box        : {item['bounding_box']}")

print("\nTotal text regions detected:",len(detected_text))


# ============================================================
# SAVE JSON
# ============================================================

output_data = {"image": str(IMAGE_PATH),
    "image_size": {"width": page_width,"height": page_height},
    "total_text_regions": len(detected_text),
    "texts": detected_text}


with open(OUTPUT_JSON,"w",encoding="utf-8") as file:
    json.dump(output_data,file,indent=4,ensure_ascii=False)
print("\nOCR JSON saved:")
print(OUTPUT_JSON)
