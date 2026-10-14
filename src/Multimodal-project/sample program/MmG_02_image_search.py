from pathlib import Path
import chromadb
import torch
from transformers import AutoProcessor, AutoModel

CHROMA_DIR = Path("outputs/chroma_valencia")
MODEL_ID = "google/siglip-base-patch16-224"
COLLECTION_NAME = "valencia_photos"


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


print("Device:", device)
if not CHROMA_DIR.exists():
    raise FileNotFoundError(f"ChromaDB not found: {CHROMA_DIR}. Run MmG_01_build_image_index.py first.")
print("Loading SigLIP...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(device)
model.eval()

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(name=COLLECTION_NAME)
print("Indexed images:", collection.count())

query = input("\nEnter search query: ").strip()
if not query:
    raise ValueError("Search query cannot be empty.")
print("Generating query embedding...")
text_inputs = processor(text=[query],padding="max_length",return_tensors="pt")
text_inputs = {key: value.to(device) for key, value in text_inputs.items()}


with torch.no_grad():
    text_output = model.get_text_features(**text_inputs)
    text_embedding = text_output.pooler_output
text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
query_vector = text_embedding.squeeze(0).cpu().tolist()
top_k = min(5, collection.count())
results = collection.query(query_embeddings=[query_vector],n_results=top_k,include=["metadatas", "distances"])


print("\n======================================")
print("SEARCH RESULTS")
print("======================================")


for rank in range(top_k):
    metadata = results["metadatas"][0][rank]
    distance = results["distances"][0][rank]

    similarity = 1 - distance

    print(f"\nRank: {rank + 1}")
    print(f"Similarity: {similarity:.4f}")
    print(f"File: {metadata['file_name']}")
    print(f"Path: {metadata['path']}")
