from pathlib import Path
import uuid
import chromadb
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel


VIDEOS_DIR = Path("data/Valencia Tour/Videos")
CHROMA_DIR = Path("outputs/chroma_valencia")
MODEL_ID = "google/siglip-base-patch16-224"
COLLECTION_NAME = "valencia_videos"
FRAME_INTERVAL_SECONDS = 3
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Device:", device)


if not VIDEOS_DIR.exists():
    raise FileNotFoundError(f"Video directory not found: {VIDEOS_DIR}")


print("Loading SigLIP...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(device)
model.eval()


print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


video_paths = [
    path
    for path in VIDEOS_DIR.rglob("*")
    if path.suffix.lower() in SUPPORTED_EXTENSIONS
]

print("Videos found:", len(video_paths))

if not video_paths:
    raise RuntimeError("No supported videos found.")


for video_index, video_path in enumerate(video_paths, start=1):

    print(f"\nProcessing video {video_index}/{len(video_paths)}")
    print(video_path.name)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open video.")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print("Invalid FPS.")
        cap.release()
        continue

    frame_interval = int(fps * FRAME_INTERVAL_SECONDS)

    frame_number = 0
    stored_frames = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number % frame_interval == 0:

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)

            inputs = processor(images=image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                image_output = model.get_image_features(**inputs)
                image_embedding = image_output.pooler_output

            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            embedding = image_embedding.squeeze(0).cpu().tolist()

            timestamp_seconds = frame_number / fps

            document_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{video_path.resolve()}_{frame_number}"
                )
            )

            collection.upsert(
                ids=[document_id],
                embeddings=[embedding],
                metadatas=[{
                    "type": "video_frame",
                    "video_name": video_path.name,
                    "video_path": str(video_path),
                    "frame_number": frame_number,
                    "timestamp_seconds": round(timestamp_seconds, 2)
                }]
            )

            stored_frames += 1

            print(
                f"Stored frame {frame_number} "
                f"at {timestamp_seconds:.2f} sec"
            )

        frame_number += 1

    cap.release()

    print("Frames stored:", stored_frames)


print("\n======================================")
print("VIDEO INDEXING COMPLETE")
print("======================================")

print("Total indexed video frames:", collection.count())
