from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    # ---------------------------------------------------------
    # 1. Load image
    # ---------------------------------------------------------
    IMAGE_PATH = Path("data/images/alexander-lunyov-nlY-pr6Fr1w-unsplash.jpg")
    image = Image.open(IMAGE_PATH)

    print("=== PIL IMAGE ===")
    print(f"Format      : {image.format}")
    print(f"Mode        : {image.mode}")
    print(f"Width       : {image.width}")
    print(f"Height      : {image.height}")

    # ---------------------------------------------------------
    # 2. Convert PIL image -> NumPy array
    # ---------------------------------------------------------
    image_np = np.array(image)

    print("\n=== NUMPY ===")
    print(f"Shape       : {image_np.shape}")
    print(f"Dtype       : {image_np.dtype}")
    print(f"Min value   : {image_np.min()}")
    print(f"Max value   : {image_np.max()}")

    # ---------------------------------------------------------
    # 3. Inspect one pixel
    # ---------------------------------------------------------
    pixel = image_np[0, 0]

    print("\n=== ONE PIXEL ===")
    print(f"Pixel       : {pixel}")

    # ---------------------------------------------------------
    # 4. Convert NumPy -> PyTorch tensor
    # ---------------------------------------------------------
    tensor_hwc = torch.from_numpy(image_np)

    print("\n=== PYTORCH TENSOR ===")
    print(f"Shape       : {tensor_hwc.shape}")
    print(f"Dtype       : {tensor_hwc.dtype}")

    # ---------------------------------------------------------
    # 5. Convert HWC -> CHW
    # ---------------------------------------------------------
    tensor_chw = tensor_hwc.permute(2, 0, 1)

    print("\n=== CHW TENSOR ===")
    print(f"Shape       : {tensor_chw.shape}")
    print(f"Dtype       : {tensor_chw.dtype}")

    # ---------------------------------------------------------
    # 6. Add batch dimension -> BCHW
    # ---------------------------------------------------------
    tensor_bchw = tensor_chw.unsqueeze(0)

    print("\n=== BCHW TENSOR ===")
    print(f"Shape       : {tensor_bchw.shape}")
    print(f"Dtype       : {tensor_bchw.dtype}")


if __name__ == "__main__":
    main()
