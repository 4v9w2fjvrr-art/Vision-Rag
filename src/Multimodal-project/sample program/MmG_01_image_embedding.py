#from bdb import _Backend
from pathlib import Path
import uuid,sys
import chromadb
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel


PHOTOS_DIR = Path("data/Valencia Tour/Photos")
CHROMA_DIR = Path("outputs/chroma_valencia")
MODEL_ID = "google/siglip-base-patch16-224"
COLLECTION_NAME = "valencia_photos"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
device= "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if not PHOTOS_DIR.exists():
    raise FileNotFoundError(f"Photo directory not found: {PHOTOS_DIR}")



print("Loading SigLIP...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(device)
model.eval()


print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(name=COLLECTION_NAME,metadata={"hnsw:space": "cosine"})


image_paths = [path for path in PHOTOS_DIR.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS]
print("Photos found:", len(image_paths))
if not image_paths:
    raise RuntimeError("No supported photos found.")
print("\nGenerating image embeddings...")


for index, image_path in enumerate(image_paths, start=1):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            image_output = model.get_image_features(**inputs)
            image_embedding = image_output.pooler_output
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        embedding = image_embedding.squeeze(0).cpu().tolist()
        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL,str(image_path.resolve())))
        collection.upsert(ids=[document_id],embeddings=[embedding],metadatas=[{"file_name": image_path.name,"path": str(image_path),"type": "photo"}])
        print(f"{index}/{len(image_paths)} -> {image_path.name}")

    except Exception as error:
        print(f"Skipped {image_path.name}: {error}")


print("\n======================================")
print("INDEXING COMPLETE")
print("======================================")

print("Total images stored:", collection.count())
print("ChromaDB path:", CHROMA_DIR)
