import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Load .env file
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

client.recreate_collection(
    collection_name="bdlaws",
    vectors_config=VectorParams(
        size=1024,  # V3 size
        distance=Distance.COSINE
    )
)

print("✅ Collection created securely from .env")
