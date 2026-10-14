from pathlib import Path

import chromadb
import torch
from transformers import AutoProcessor, AutoModel


CHROMA_DIR = Path("outputs/chroma_valencia")

MODEL_ID = "google/siglip-base-patch16-224"

PHOTO_COLLECTION = "valencia_photos"
VIDEO_COLLECTION = "valencia_videos"


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Device:", device)


print("Loading SigLIP...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(device)
model.eval()


print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

photo_collection = client.get_collection(name=PHOTO_COLLECTION)
video_collection = client.get_collection(name=VIDEO_COLLECTION)

print("Indexed photos:", photo_collection.count())
print("Indexed video frames:", video_collection.count())


query = input("\nEnter search query: ").strip()

if not query:
    raise ValueError("Search query cannot be empty.")


text_inputs = processor(
    text=[query],
    padding="max_length",
    return_tensors="pt"
)

text_inputs = {key: value.to(device) for key, value in text_inputs.items()}


with torch.no_grad():
    text_output = model.get_text_features(**text_inputs)
    text_embedding = text_output.pooler_output


text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
query_vector = text_embedding.squeeze(0).cpu().tolist()


photo_top_k = min(5, photo_collection.count())
video_top_k = min(5, video_collection.count())


photo_results = photo_collection.query(
    query_embeddings=[query_vector],
    n_results=photo_top_k,
    include=["metadatas", "distances"]
)


video_results = video_collection.query(
    query_embeddings=[query_vector],
    n_results=video_top_k,
    include=["metadatas", "distances"]
)


combined_results = []


for rank in range(photo_top_k):
    metadata = photo_results["metadatas"][0][rank]
    distance = photo_results["distances"][0][rank]
    similarity = 1 - distance

    combined_results.append({
        "type": "photo",
        "similarity": similarity,
        "file_name": metadata["file_name"],
        "path": metadata["path"]
    })


for rank in range(video_top_k):
    metadata = video_results["metadatas"][0][rank]
    distance = video_results["distances"][0][rank]
    similarity = 1 - distance

    combined_results.append({
        "type": "video_frame",
        "similarity": similarity,
        "video_name": metadata["video_name"],
        "video_path": metadata["video_path"],
        "timestamp_seconds": metadata["timestamp_seconds"],
        "frame_number": metadata["frame_number"]
    })


combined_results.sort(
    key=lambda result: result["similarity"],
    reverse=True
)


print("\n======================================")
print("UNIFIED SEARCH RESULTS")
print("======================================")


for rank, result in enumerate(combined_results, start=1):

    print(f"\nRank: {rank}")
    print(f"Type: {result['type']}")
    print(f"Similarity: {result['similarity']:.4f}")

    if result["type"] == "photo":
        print(f"File: {result['file_name']}")
        print(f"Path: {result['path']}")

    else:
        print(f"Video: {result['video_name']}")
        print(f"Timestamp: {result['timestamp_seconds']} sec")
        print(f"Frame: {result['frame_number']}")
        print(f"Path: {result['video_path']}")
