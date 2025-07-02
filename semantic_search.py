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

def get_relevance_score_gemini(query: str, section_text: str) -> dict:
    """
    Sends the query and section text to Gemini API (API key-based) to get a relevance score and reason.
    Returns: {"score": float, "reason": str}
    """
    import json

    GEMINI_API_URL = os.getenv("GEMINI_API_URL")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }

    prompt = f"""
    You are a legal assistant AI. A user asked: "{query}"

    Now read the following law section and rate how relevant it is on a scale from 0 to 10.
    Give a JSON response like this:
    {{"score": number, "reason": "..." }}

    LAW SECTION:
    {section_text}
        """

    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3}
    }

    response = requests.post(GEMINI_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        try:
            import re
            text_response = response.json()['candidates'][0]['content']['parts'][0]['text']

            # Use regex to extract the first JSON-like object
            match = re.search(r'\{.*?\}', text_response.strip(), re.DOTALL)
            if match:
                json_text = match.group(0)
                parsed = json.loads(json_text)
                return parsed
            else:
                raise Exception(f"Could not extract JSON from Gemini response:\n{text_response}")
        except Exception as e:
            raise Exception(f"Failed to parse Gemini response: {e}\nRaw response:\n{text_response}")
    else:
        raise Exception(f"Gemini request failed: {response.status_code} - {response.text}")



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
def semantic_search(query: str, top_k: int = 10):
    query_vector = embed_query(query)
    raw_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )

    scored_results = []
    for hit in raw_results:
        section_text = hit.payload['text']
        score_data = get_relevance_score_gemini(query, section_text)
        scored_results.append({
            "score": score_data['score'],
            "reason": score_data['reason'],
            "payload": hit.payload
        })

    # Sort by Gemini AI relevance score
    sorted_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)
    return sorted_results



# ✅ CLI test
if __name__ == "__main__":
    q = input("Ask your legal question: ")
    matches = semantic_search(q)

    for i, match in enumerate(matches, 1):
        payload = match["payload"]
        print(f"\n#{i} - {payload['actTitle']} > {payload['sectionTitle']}")
        print(f"Score: {match['score']} | Reason: {match['reason']}")
        print(f"{payload['text'][:300]}...")