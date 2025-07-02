from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
# Import CORSMiddleware for CORS support
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import json
from core import (
    semantic_search,
    get_all_sections_grouped_by_chapter_from_file,
    expand_with_related_sections,
    call_openrouter_chunk_analysis,
    call_openrouter_final_summary,
    enrich_relevant_sections
)


app = FastAPI()

# ✅ Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Or ["*"] to allow all origins (for development only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/legal-analyze")
async def analyze_legal_query(payload: QueryRequest):
    try:
        q = payload.query.strip()

        # Step 1: Semantic Search
        top_matches = semantic_search(q)

        # Step 2: Load all grouped sections
        chapter_grouped = get_all_sections_grouped_by_chapter_from_file()

        # Step 3: Expand context
        expanded_sections = expand_with_related_sections(top_matches, chapter_grouped)

        # Step 4: Gemini analysis in chunks
        chunks = [expanded_sections[i:i + 10] for i in range(0, len(expanded_sections), 10)]
        all_insights = []
        for chunk in chunks:
            result = call_openrouter_chunk_analysis(q, chunk)
            all_insights.append(result)

        # Step 5: Final synthesis
        final_analysis = call_openrouter_final_summary(q, all_insights)

        # Step 6: Enrich final top sections
        flat_sections = sum(chapter_grouped.values(), [])
        enriched_sections = enrich_relevant_sections(final_analysis["relevant_sections"], flat_sections)

        # Step 7: Return final structured JSON
        return {
            "final_summary": final_analysis["final_summary"],
            "final_legal_insight": final_analysis["final_legal_insight"],
            "relevant_sections_full": enriched_sections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
