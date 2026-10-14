from pathlib import Path

import chromadb


CHROMA_DIR = Path("outputs/chroma_valencia")

PHOTO_COLLECTION = "valencia_photos"
VIDEO_COLLECTION = "valencia_videos"


client = chromadb.PersistentClient(path=str(CHROMA_DIR))


print("\nAvailable collections:")

for collection_info in client.list_collections():
    print("-", collection_info.name)


print("\n======================================")
print("VIDEO COLLECTION")
print("======================================")

video_collection = client.get_collection(name=VIDEO_COLLECTION)

print("Total video frame records:", video_collection.count())


video_results = video_collection.get(
    include=["metadatas"]
)


video_names = set()


for metadata in video_results["metadatas"]:
    video_names.add(metadata["video_name"])


print("\nVideo files found:")

for name in sorted(video_names):
    print("-", name)


print("\n======================================")
print("SAMPLE VIDEO FRAME RECORDS")
print("======================================")


sample_results = video_collection.get(
    limit=10,
    include=["metadatas"]
)


for index, metadata in enumerate(sample_results["metadatas"], start=1):
    print(f"\nRecord {index}")
    print("Video:", metadata["video_name"])
    print("Path:", metadata["video_path"])
    print("Frame:", metadata["frame_number"])
    print("Timestamp:", metadata["timestamp_seconds"], "seconds")


print("\n======================================")
print("PHOTO COLLECTION")
print("======================================")

photo_collection = client.get_collection(name=PHOTO_COLLECTION)

print("Total photo records:", photo_collection.count())


photo_results = photo_collection.get(
    limit=10,
    include=["metadatas"]
)


for index, metadata in enumerate(photo_results["metadatas"], start=1):
    print(f"\nPhoto {index}")
    print("File:", metadata["file_name"])
    print("Path:", metadata["path"])
