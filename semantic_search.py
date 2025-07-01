import os
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest
from jina import __version__ as jina_version

# ✅ Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
JINA_API_URL = os.getenv("JINA_API_URL")
JINA_API_KEY = os.getenv("JINA_API_KEY")
COLLECTION_NAME = "bdlaws"

# ✅ Qdrant connection
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ✅ Embed query


def embed_query(text: str):
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {"input": [text], "model": "jina-embeddings-v3"}
    response = requests.post(JINA_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        raise Exception(
            f"Jina embedding failed: {response.status_code} - {response.text}"
        )


# ✅ Perform semantic search


def semantic_search(query: str, top_k: int = 1):
    query_vector = embed_query(query)
    results = client.search(
        collection_name=COLLECTION_NAME, query_vector=query_vector, limit=top_k
    )
    return [hit.payload for hit in results]


# ✅ CLI test
if __name__ == "__main__":
    q = input("Ask your legal question: ")
    matches = semantic_search(q)
    for i, m in enumerate(matches, 1):
        # print(f"\n#{i} - {m['actTitle']} > {m['sectionTitle']}\n{m['text'][:300]}...")
        print(f"\n#{i} - Full payload from Qdrant:\n{m}\n")
