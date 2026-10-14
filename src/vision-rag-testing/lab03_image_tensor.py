from pathlib import Path
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification




def main():

    IMAGE_PATH = Path("data/images/alexander-lunyov-nlY-pr6Fr1w-unsplash.jpg")
    MODEL_NAME = "google/vit-base-patch16-224"
    # ---------------------------------------------------------
    # 1. Select device
    # ---------------------------------------------------------

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=== DEVICE ===")
    print(device)

    # ---------------------------------------------------------
    # 2. Load image
    # ---------------------------------------------------------

    image = Image.open(IMAGE_PATH).convert("RGB")

    print("\n=== IMAGE ===")
    print(f"Size        : {image.size}")
    print(f"Mode        : {image.mode}")

    # ---------------------------------------------------------
    # 3. Load pretrained processor
    # ---------------------------------------------------------

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

    # ---------------------------------------------------------
    # 4. Preprocess image
    # ---------------------------------------------------------

    inputs = processor(images=image, return_tensors="pt")

    print("\n=== MODEL INPUT ===")

    for name, tensor in inputs.items():
        print(f"{name:12} : shape={tensor.shape}, dtype={tensor.dtype}")

    # ---------------------------------------------------------
    # 5. Load pretrained Vision Transformer
    # ---------------------------------------------------------

    model = ViTForImageClassification.from_pretrained(MODEL_NAME)

    model = model.to(device)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    # ---------------------------------------------------------
    # 6. Inference
    # ---------------------------------------------------------

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    # ---------------------------------------------------------
    # 7. Inspect output
    # ---------------------------------------------------------

    print("\n=== MODEL OUTPUT ===")
    print(f"Logits shape : {outputs.logits.shape}")

    predicted_class = outputs.logits.argmax(-1).item()

    label = model.config.id2label[predicted_class]

    print(f"Class ID     : {predicted_class}")
    print(f"Prediction   : {label}")


if __name__ == "__main__":
    main()
