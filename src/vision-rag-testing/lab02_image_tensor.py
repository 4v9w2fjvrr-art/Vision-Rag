from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms


def main():
    IMAGE_PATH = Path("data/images/alexander-lunyov-nlY-pr6Fr1w-unsplash.jpg")
    # Load image
    image = Image.open(IMAGE_PATH).convert("RGB")

    print("=== ORIGINAL ===")
    print(f"Size        : {image.size}")
    print(f"Mode        : {image.mode}")

    # Resize + convert to tensor
    transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),])

    tensor = transform(image)

    print("\n=== AFTER PREPROCESSING ===")
    print(f"Shape       : {tensor.shape}")
    print(f"Dtype       : {tensor.dtype}")
    print(f"Min value   : {tensor.min().item():.4f}")
    print(f"Max value   : {tensor.max().item():.4f}")

    # Add batch dimension
    batch = tensor.unsqueeze(0)

    print("\n=== BATCH ===")
    print(f"Shape       : {batch.shape}")
    print(f"Dtype       : {batch.dtype}")


if __name__ == "__main__":
    main()
