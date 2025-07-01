import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# ✅ Load env
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "bdlaws"

# ✅ Connect to Qdrant
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ✅ Load all laws
with open("bdlaws_merged.json", "r", encoding="utf-8") as f:
    all_laws = json.load(f)

# ✅ Fetch uploaded section URLs
print("🔍 Scanning Qdrant...")
existing_urls = set()
offset = None

while True:
    scroll_result, next_page = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        with_payload=True,
        limit=1000,
        offset=offset
    )
    for point in scroll_result:
        url = point.payload.get("sectionUrl")
        if url:
            existing_urls.add(url)

    if next_page is None:
        break
    offset = next_page

print(f"✅ Found {len(existing_urls)} items uploaded.")

# ✅ Compare and collect missing ones
missing = []
for law in all_laws:
    if law.get("sectionUrl") not in existing_urls:
        missing.append(law)

# ✅ Save missing to file
with open("missing_laws.json", "w", encoding="utf-8") as f:
    json.dump(missing, f, ensure_ascii=False, indent=2)

print(f"❌ Found {len(missing)} missing laws.")
print("💾 Saved to missing_laws.json")
