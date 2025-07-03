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


def fetch_sections_from_qdrant_by_title(gemini_titles: list, top_k=1):
    enriched = []

    for entry in gemini_titles:
        title_query = entry['title']

        # 🔍 Vector search with title as query
        try:
            title_vector = embed_query(title_query)
            results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=title_vector,
                limit=top_k
            )
        except Exception as e:
            print(f"❌ Qdrant search failed for: {title_query}\n{e}")
            continue

        if results:
            best_match = results[0]
            enriched.append({
                "title": entry["title"],
                "reason": entry["reason"],
                "section": re.search(r"\((\d+[A-Za-z]*)\)", entry["title"]).group(1) if re.search(r"\((\d+[A-Za-z]*)\)", entry["title"]) else None,
                **best_match.payload  # include all metadata fields like actTitle, text, etc.
            })
        else:
            print(f"⚠️ No match found in Qdrant for: {title_query}")

    return enriched

def enrich_relevant_sections(relevant_sections: list, all_sections: list) -> list:
    enriched = []

    for entry in relevant_sections:
        raw_title = entry["title"].strip()

        # ✅ Extract section number (optional, may not exist)
        match = re.search(r"\(Section\s+(\d+[A-Za-z]*)\)", raw_title)
        section_number = match.group(1) if match else None

        # ✅ Clean trailing section numbers or newline digits
        cleaned_title = re.sub(r"\s*\(Section\s+\d+[A-Za-z]*\)", "", raw_title).strip()
        cleaned_title = re.sub(r"\n\d+\.*$", "", cleaned_title).strip()

        # ✅ Split act and section titles
        if ">" in cleaned_title:
            act_title, section_title = map(str.strip, cleaned_title.split(">", 1))
        else:
            act_title = ""
            section_title = cleaned_title

        matched = None

        for sec in all_sections:
            sec_act = sec['actTitle'].strip()
            sec_title = sec['sectionTitle'].strip()

            # Match more flexibly:
            if (
                sec_act.endswith(act_title) and
                section_title.lower() in sec_title.lower()
            ):
                matched = sec
                break

        if matched:
            enriched.append({
                "title": raw_title,
                "reason": entry.get("reason"),
                "section": section_number,
                "actTitle": matched.get("actTitle"),
                "sectionTitle": matched.get("sectionTitle"),
                "chapterTitle": matched.get("chapterTitle"),
                "sectionUrl": matched.get("sectionUrl"),
                "actUrl": matched.get("actUrl"),
                "text": matched.get("text"),
                "actRepealed": matched.get("actRepealed"),
                "actPreamble": matched.get("actPreamble"),
                "actDate": matched.get("actDate"),
                "actSubtitle": matched.get("actSubtitle")
            })
        else:
            print(f"⚠️ Could not match: {raw_title}")

    return enriched


def call_openrouter_final_summary(query: str, chunk_responses: list) -> dict:
    prompt = f"""
    You are a legal assistant. A user asked the following question:

    "{query}"

    You were given multiple JSON responses from analyzing relevant legal sections. Each JSON includes:
    - a `summary` of that chunk,
    - a list of `top_sections`,
    - and a `legal_insight`.

    Return final JSON in this format:
    {{
    "final_summary": "...",
    "relevant_sections": [
        {{ "title": "...", "reason": "..." }}
    ],
    "final_legal_insight": "..."
    }}

    Here are the JSONs:
    {json.dumps(chunk_responses, indent=2)}
    """

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
    }

    response = requests.post(
        os.getenv("OPENROUTER_API_URL"),
        headers=headers,
        json=data
    )

    if response.status_code != 200:
        raise Exception(f"OpenRouter failed: {response.status_code} - {response.text}")

    reply = response.json()['choices'][0]['message']['content']
    fixed = clean_gemini_json(reply)
    return json.loads(fixed)




def clean_gemini_json(text: str) -> str:
    # Remove Markdown code block wrappers
    if text.startswith("```json"):
        text = text.strip().removeprefix("```json").removesuffix("```").strip()

    # Replace control characters inside string values
    def sanitize_string_literals(m):
        return '"' + m.group(1).replace('\n', '\\n').replace('\r', '').replace('\t', '\\t').replace('"', '\\"') + '"'

    # Replace double quotes and linebreaks inside string values
    return re.sub(r'"([^"]*?)"', sanitize_string_literals, text)


def call_openrouter_chunk_analysis(query: str, sections: list) -> dict:
    law_text = "\n\n".join(
    f"{sec['actTitle']} > {sec['sectionTitle']}\n{sec['text']}" for sec in sections
    )

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
    {law_text}
    """

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "google/gemini-2.5-flash",  # use your preferred model
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
    }

    response = requests.post(
        os.getenv("OPENROUTER_API_URL"),
        headers=headers,
        json=data
    )

    if response.status_code != 200:
        raise Exception(f"OpenRouter failed: {response.status_code} - {response.text}")

    reply = response.json()['choices'][0]['message']['content']
    try:
        return json.loads(reply.strip())
    except json.JSONDecodeError:
        fixed = clean_gemini_json(reply)  # Reuse your fixer
        return json.loads(fixed)




def get_all_sections_grouped_by_chapter_from_qdrant(limit=10_000):
    print("📡 Fetching all legal sections from Qdrant...")
    all_sections, _ = client.scroll(collection_name=COLLECTION_NAME, limit=limit)

    grouped = {}
    for sec in all_sections:
        chapter = sec.payload.get("chapterTitle", "").strip()
        grouped.setdefault(chapter, []).append(sec.payload)
    
    return grouped



# def expand_with_related_sections(top_matches: list, chapter_grouped: dict, neighbors: int = 10):
#     """
#     Expand legal context by:
#     1. Fetching all sections from top N chapters
#     2. If no chapter, fallback to ±neighbors using section index
#     """
#     related_sections = []
#     seen_ids = set()

#     # 1. Collect top 3 unique chapter titles
#     top_chapters = []
#     for match in top_matches:
#         chap = match["payload"].get("chapterTitle", "").strip()
#         if chap and chap not in top_chapters:
#             top_chapters.append(chap)
#         if len(top_chapters) >= 3:
#             break

#     # 2. Get all sections from those chapters
#     for chap in top_chapters:
#         for sec in chapter_grouped.get(chap, []):
#             sec_id = sec['sectionUrl']
#             if sec_id not in seen_ids:
#                 related_sections.append(sec)
#                 seen_ids.add(sec_id)

#     # 3. If no chapter or section is flat, fallback to ±10
#     all_flat = list(sum(chapter_grouped.values(), []))  # flatten all
#     url_to_index = {s['sectionUrl']: idx for idx, s in enumerate(all_flat)}

#     for match in top_matches[:3]:
#         sec_id = match["payload"]["sectionUrl"]
#         idx = url_to_index.get(sec_id, None)
#         if idx is not None:
#             for i in range(max(0, idx - neighbors), min(len(all_flat), idx + neighbors + 1)):
#                 sec = all_flat[i]
#                 sec_id = sec['sectionUrl']
#                 if sec_id not in seen_ids:
#                     related_sections.append(sec)
#                     seen_ids.add(sec_id)

#     return related_sections

def expand_with_related_sections(top_matches: list, chapter_grouped: dict, neighbors: int = 10):
    related_sections = []
    seen_ids = set()
    MAX_PER_CHAPTER = 10

    # 1. Top chapters
    top_chapters = []
    for match in top_matches:
        chap = match["payload"].get("chapterTitle", "").strip()
        if chap and chap not in top_chapters:
            top_chapters.append(chap)
        if len(top_chapters) >= 3:
            break

    for chap in top_chapters:
        count = 0
        for sec in chapter_grouped.get(chap, []):
            sec_id = sec['sectionUrl']
            if sec_id not in seen_ids:
                related_sections.append(sec)
                seen_ids.add(sec_id)
                count += 1
            if count >= MAX_PER_CHAPTER:
                break

    # 2. Optional fallback: ±neighbors from top 3
    if len(related_sections) < 30:
        all_flat = list(sum(chapter_grouped.values(), []))  # flatten
        url_to_index = {s['sectionUrl']: idx for idx, s in enumerate(all_flat)}

        for match in top_matches[:3]:
            sec_id = match["payload"]["sectionUrl"]
            idx = url_to_index.get(sec_id)
            if idx is not None:
                for i in range(max(0, idx - neighbors), min(len(all_flat), idx + neighbors + 1)):
                    sec = all_flat[i]
                    sec_id = sec['sectionUrl']
                    if sec_id not in seen_ids:
                        related_sections.append(sec)
                        seen_ids.add(sec_id)

    return related_sections


def get_relevance_score_openrouter(query: str, section_text: str) -> dict:
    prompt = f"""
    You are a legal assistant AI. A user asked: "{query}"

    Read this law section and rate relevance from 0 to 10.
    Return JSON: {{ "score": number, "reason": "..." }}

    LAW SECTION:
    {section_text}
    """

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
    }

    response = requests.post(
        os.getenv("OPENROUTER_API_URL"),
        headers=headers,
        json=data
    )

    if response.status_code != 200:
        raise Exception(f"OpenRouter failed: {response.status_code} - {response.text}")

    reply = response.json()['choices'][0]['message']['content']
    fixed = clean_gemini_json(reply)
    return json.loads(fixed)




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
        score_data = get_relevance_score_openrouter(query, section_text)
        scored_result = {
            "score": score_data['score'],
            "reason": score_data['reason'],
            "payload": hit.payload  # full section
        }
        scored_results.append(scored_result)
        

    # Sort by Gemini AI relevance score
    sorted_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)
    return sorted_results


# ✅ CLI test
if __name__ == "__main__":
    q = input("Ask your legal question: ")
    top_matches = semantic_search(q)
    chapter_grouped = get_all_sections_grouped_by_chapter_from_qdrant()
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
        result = call_openrouter_chunk_analysis(q, chunk)
        all_insights.append(result)

    # Optional: Print all responses
    for i, insight in enumerate(all_insights):
        print(f"\n🔍 Gemini Analysis (Chunk {i+1}):")
        print(json.dumps(insight, indent=2, ensure_ascii=False))
    
    # ✅ FINAL SYNTHESIS STEP — combine all chunks into one final answer
    print("\n📦 Compiling final answer from Gemini...")
    final_analysis = call_openrouter_final_summary(q, all_insights)

    print("\n🧠 Final Gemini Response:")
    print(json.dumps(final_analysis, indent=2, ensure_ascii=False))
    
    # Flatten all sections from grouped chapters
    flat_sections = sum(chapter_grouped.values(), [])
    # Enrich Gemini's response with full metadata
    enriched_sections = fetch_sections_from_qdrant_by_title(final_analysis["relevant_sections"])

    # Optionally update final_analysis with enriched result
    final_analysis["relevant_sections_full"] = enriched_sections

    # Pretty print output
    print("\n🧾 Enriched Relevant Sections:")
    for i, sec in enumerate(enriched_sections, 1):
        print(f"\n🔹 Section #{i}")
        print(f"📌 {sec['title']}")
        print(f"📚 Reason: {sec['reason']}")
        print(f"🏛️ Act Title: {sec['actTitle']}")
        print(f"📜 Section Title: {sec['sectionTitle']}")
        print(f"📖 Chapter: {sec.get('chapterTitle')}")
        print(f"📝 Text Preview: {sec['text'][:500]}...")
        print(f"📖 Act URL: {sec['actUrl']}")
        print(f"🔗 Section URL: {sec['sectionUrl']}")
        print(f"📆 Act Date: {sec.get('actDate')}")
        print(f"📝 Subtitle: {sec.get('actSubtitle')}")
        print(f"🚫 Repealed: {sec.get('actRepealed')}")
    
    # ✅ Create a final JSON object
    final_json_output = {
        "final_summary": final_analysis["final_summary"],
        "final_legal_insight": final_analysis["final_legal_insight"],
        "relevant_sections_full": enriched_sections
    }

    # ✅ Save to file or print
    with open("final_legal_response.json", "w", encoding="utf-8") as f:
        json.dump(final_json_output, f, indent=2, ensure_ascii=False)

    print("\n✅ Final legal response JSON saved as 'final_legal_response.json'")