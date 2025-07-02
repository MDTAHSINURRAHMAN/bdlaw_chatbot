import os
import json
import uuid
import time
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# ✅ Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
JINA_API_URL = os.getenv("JINA_API_URL")
JINA_API_KEY = os.getenv("JINA_API_KEY")
COLLECTION_NAME = "bdlaws"

# ✅ Connect to Qdrant
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ✅ Load law sections
with open("bdlaws_merged.json", "r", encoding="utf-8") as f:
    laws = json.load(f)

# ✅ Jina headers
HEADERS = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json"
}

def get_embedding(text):
    for attempt in range(3):
        try:
            res = requests.post(
                JINA_API_URL,
                headers=HEADERS,
                json={"input": [text], "model": "jina-embeddings-v3"},
                timeout=30
            )
            if res.status_code == 200:
                return res.json()["data"][0]["embedding"]
            else:
                print(f"❌ Jina Error [{res.status_code}]: {res.text}")
                time.sleep(2)
        except Exception as e:
            print(f"⚠️ Retry due to error: {e}")
            time.sleep(2)
    return None

# ✅ Embed and upload
batch_size = 32
points_batch = []

for i, section in enumerate(tqdm(laws, desc="🚀 Embedding & Uploading")):
    text_input = "\n".join([
        section.get("actTitle", ""),
        section.get("sectionTitle", ""),
        section.get("chapterTitle", ""),
        section.get("text", ""),
        section.get("actSubtitle", "")
    ]).strip()

    embedding = get_embedding(text_input)
    if embedding is None:
        continue  # skip this section

    payload = {
        "sectionUrl": section.get("sectionUrl", ""),
        "actUrl": section.get("actUrl", ""),
        "actDate": section.get("actDate", ""),
        "actPreamble": section.get("actPreamble", ""),
        "actRepealed": section.get("actRepealed", False),
        "actTitle": section.get("actTitle", ""),
        "sectionTitle": section.get("sectionTitle", ""),
        "chapterTitle": section.get("chapterTitle", ""),
        "text": section.get("text", ""),
        "actSubtitle": section.get("actSubtitle", "")
    }

    points_batch.append(PointStruct(id=str(uuid.uuid4()), vector=embedding, payload=payload))

    if len(points_batch) >= batch_size:
        try:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points_batch)
            points_batch = []
        except Exception as e:
            print(f"❌ Qdrant upload error: {e}")
            time.sleep(3)

# ✅ Upload remaining
if points_batch:
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points_batch)

print("✅ All sections embedded and uploaded!")