from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from core import (
    semantic_search,
    get_all_sections_grouped_by_chapter_from_qdrant,
    expand_with_related_sections,
    call_openrouter_chunk_analysis,
    call_openrouter_final_summary,
    fetch_sections_from_qdrant_by_title
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or replace with your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Define request model for query input
class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"status": "✅ Legal Assistant API is live!"}

# ✅ POST endpoint with request model
@app.post("/legal-analyze")
async def analyze_legal_query(payload: QueryRequest):
    query = payload.query

    # Step 1: Semantic search
    top_matches = semantic_search(query)

    # Step 2: Expand sections
    chapter_grouped = get_all_sections_grouped_by_chapter_from_qdrant()
    expanded_sections = expand_with_related_sections(top_matches, chapter_grouped)

    # Step 3: Analyze chunks
    chunks = [expanded_sections[i:i + 10] for i in range(0, len(expanded_sections), 10)]
    all_insights = [call_openrouter_chunk_analysis(query, chunk) for chunk in chunks]

    # Step 4: Final Gemini summary
    final_analysis = call_openrouter_final_summary(query, all_insights)

    # Step 5: Get enriched metadata from Qdrant
    enriched_sections = fetch_sections_from_qdrant_by_title(final_analysis["relevant_sections"])
    final_analysis["relevant_sections_full"] = enriched_sections

    return JSONResponse(final_analysis)
