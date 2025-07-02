import os
import json
import ast
import re
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


def call_gemini_final_summary(query: str, chunk_responses: list) -> dict:
    prompt = f"""
    You are a legal assistant. A user asked the following question:

    "{query}"

    You were given multiple JSON responses from analyzing relevant legal sections in chunks. Each JSON includes:
    - a `summary` of that chunk,
    - a list of `top_sections` with reasons,
    - and a `legal_insight`.

    Your job is to synthesize all this information and return a final analysis as a JSON.

    Output format:
    {{
    "final_summary": "...",
    "relevant_sections": [
        {{ "title": "...", "reason": "..." }},
        ...
    ],
    "final_legal_insight": "..."
    }}

    Here are the JSONs:
    {json.dumps(chunk_responses, indent=2)}
    """.strip()

    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2}
    }

    response = requests.post(
        os.getenv("GEMINI_API_URL"),
        headers=headers,
        params={"key": os.getenv("GEMINI_API_KEY")},
        json=data
    )

    if response.status_code != 200:
        raise Exception(f"Gemini request failed: {response.status_code} - {response.text}")

    gemini_text = response.json()['candidates'][0]['content']['parts'][0]['text']
    cleaned = clean_gemini_json(gemini_text)
    return json.loads(cleaned)



def clean_gemini_json(text: str) -> str:
    # Remove Markdown code block wrappers
    if text.startswith("```json"):
        text = text.strip().removeprefix("```json").removesuffix("```").strip()

    # Replace control characters inside string values
    def sanitize_string_literals(m):
        return '"' + m.group(1).replace('\n', '\\n').replace('\r', '').replace('\t', '\\t').replace('"', '\\"') + '"'

    # Replace double quotes and linebreaks inside string values
    return re.sub(r'"([^"]*?)"', sanitize_string_literals, text)


def call_gemini_chunk_analysis(query: str, sections: list) -> dict:
    combined_text = "\n\n".join([
        f"{sec['actTitle']} > {sec['sectionTitle']}\n{sec['text']}"
        for sec in sections
    ])

    prompt = f"""
    A user asked: "{query}"

    You are a legal assistant. Analyze the following legal sections. Return a JSON with:
    {{
      "summary": "...",
      "top_sections": [
        {{ "title": "Act > Section", "reason": "..." }},
        ...
      ],
      "legal_insight": "..."
    }}

    Laws:
    {combined_text}
    """

    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt.strip()}]}],
        "generationConfig": {"temperature": 0.2}
    }

    response = requests.post(
        os.getenv("GEMINI_API_URL"),
        headers=headers,
        params={"key": os.getenv("GEMINI_API_KEY")},
        json=data
    )

    if response.status_code != 200:
        raise Exception(f"Gemini request failed: {response.status_code} - {response.text}")

    try:
        gemini_text = response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        raise Exception(f"Gemini API structure changed or response missing: {e}\nRaw:\n{response.text}")

    if not gemini_text.strip():
        raise Exception(f"Gemini returned empty response!\nRaw:\n{response.text}")

    try:
        return json.loads(gemini_text.strip())
    except json.JSONDecodeError:
        # Try to clean the text using custom fixer
        fixed = clean_gemini_json(gemini_text)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            raise Exception(f"Gemini JSON parsing failed after cleaning: {e}\nFixed text:\n{fixed}")



def get_all_sections_grouped_by_chapter_from_file(filepath="bdlaws_merged.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        all_sections = json.load(f)

    grouped = {}
    for sec in all_sections:
        chapter = sec.get("chapterTitle", "").strip()
        grouped.setdefault(chapter, []).append(sec)

    return grouped



def expand_with_related_sections(top_matches: list, chapter_grouped: dict, neighbors: int = 10):
    """
    Expand legal context by:
    1. Fetching all sections from top N chapters
    2. If no chapter, fallback to ±neighbors using section index
    """
    related_sections = []
    seen_ids = set()

    # 1. Collect top 3 unique chapter titles
    top_chapters = []
    for match in top_matches:
        chap = match["payload"].get("chapterTitle", "").strip()
        if chap and chap not in top_chapters:
            top_chapters.append(chap)
        if len(top_chapters) >= 3:
            break

    # 2. Get all sections from those chapters
    for chap in top_chapters:
        for sec in chapter_grouped.get(chap, []):
            sec_id = sec['sectionUrl']
            if sec_id not in seen_ids:
                related_sections.append(sec)
                seen_ids.add(sec_id)

    # 3. If no chapter or section is flat, fallback to ±10
    all_flat = list(sum(chapter_grouped.values(), []))  # flatten all
    url_to_index = {s['sectionUrl']: idx for idx, s in enumerate(all_flat)}

    for match in top_matches[:3]:
        sec_id = match["payload"]["sectionUrl"]
        idx = url_to_index.get(sec_id, None)
        if idx is not None:
            for i in range(max(0, idx - neighbors), min(len(all_flat), idx + neighbors + 1)):
                sec = all_flat[i]
                sec_id = sec['sectionUrl']
                if sec_id not in seen_ids:
                    related_sections.append(sec)
                    seen_ids.add(sec_id)

    return related_sections


def get_relevance_score_gemini(query: str, section_text: str) -> dict:

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
    {{"score": number, "reason": "..."}}

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
            text_response = response.json()['candidates'][0]['content']['parts'][0]['text']

            # ✅ Clean up code blocks if present
            cleaned = re.sub(r"```json|```", "", text_response.strip(), flags=re.MULTILINE)

            return json.loads(cleaned.strip())
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
    top_matches = semantic_search(q)
    chapter_grouped = get_all_sections_grouped_by_chapter_from_file()
    expanded_sections = expand_with_related_sections(top_matches, chapter_grouped)

    print(f"\n🔍 Expanded Contextual Sections (Total: {len(expanded_sections)}):")
    for i, sec in enumerate(expanded_sections, 1):
        print(f"\n#{i} - {sec['actTitle']} > {sec['sectionTitle']}\n{sec['text'][:500]}...\n")
    print("\n⏳ Sending sections to Gemini for analysis...")

    # Split sections into batches of 10 (tune this if needed)
    chunks = [expanded_sections[i:i + 10] for i in range(0, len(expanded_sections), 10)]
    all_insights = []

    for i, chunk in enumerate(chunks):
        print(f"\n🔹 Analyzing chunk {i+1}/{len(chunks)}...")
        result = call_gemini_chunk_analysis(q, chunk)
        all_insights.append(result)

    # Optional: Print all responses
    for i, insight in enumerate(all_insights):
        print(f"\n🔍 Gemini Analysis (Chunk {i+1}):")
        print(json.dumps(insight, indent=2, ensure_ascii=False))
    
    # ✅ FINAL SYNTHESIS STEP — combine all chunks into one final answer
    print("\n📦 Compiling final answer from Gemini...")
    final_analysis = call_gemini_final_summary(q, all_insights)

    print("\n🧠 Final Gemini Response:")
    print(json.dumps(final_analysis, indent=2, ensure_ascii=False))