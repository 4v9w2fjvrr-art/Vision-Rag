from pathlib import Path
import chromadb
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel


CHROMA_DIR = Path("outputs/chroma_valencia")
OUTPUT_DIR = Path("outputs/retrieved_media")
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


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


photo_results = photo_collection.query(
    query_embeddings=[query_vector],
    n_results=min(5, photo_collection.count()),
    include=["metadatas", "distances"]
)


video_results = video_collection.query(
    query_embeddings=[query_vector],
    n_results=min(5, video_collection.count()),
    include=["metadatas", "distances"]
)


combined_results = []


for metadata, distance in zip(
    photo_results["metadatas"][0],
    photo_results["distances"][0]
):
    combined_results.append({
        "type": "photo",
        "similarity": 1 - distance,
        "file_name": metadata["file_name"],
        "path": metadata["path"]
    })


for metadata, distance in zip(
    video_results["metadatas"][0],
    video_results["distances"][0]
):
    combined_results.append({
        "type": "video_frame",
        "similarity": 1 - distance,
        "video_name": metadata["video_name"],
        "video_path": metadata["video_path"],
        "frame_number": metadata["frame_number"],
        "timestamp_seconds": metadata["timestamp_seconds"]
    })


combined_results.sort(
    key=lambda result: result["similarity"],
    reverse=True
)


top_result = combined_results[0]


print("\n======================================")
print("TOP RETRIEVAL RESULT")
print("======================================")

print("Type:", top_result["type"])
print(f"Similarity: {top_result['similarity']:.4f}")


if top_result["type"] == "photo":

    source_path = Path(top_result["path"])
    image = Image.open(source_path).convert("RGB")

    output_path = OUTPUT_DIR / "top_retrieved_photo.jpg"
    image.save(output_path)

    print("File:", top_result["file_name"])
    print("Source:", source_path)
    print("Saved:", output_path)


else:

    video_path = Path(top_result["video_path"])
    frame_number = int(top_result["frame_number"])

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(f"Could not read frame {frame_number}")

    output_path = OUTPUT_DIR / "top_retrieved_video_frame.jpg"
    cv2.imwrite(str(output_path), frame)

    print("Video:", top_result["video_name"])
    print("Timestamp:", top_result["timestamp_seconds"], "sec")
    print("Frame:", frame_number)
    print("Source:", video_path)
    print("Saved:", output_path)


print("\nRetrieval complete.")
