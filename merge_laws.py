import json
import math

with open("bdlaws_merged.json", "r", encoding="utf-8") as f:
    data = json.load(f)

required_fields = ["actTitle", "sectionTitle", "text", "sectionUrl"]
invalid_entries = []

for i, section in enumerate(data):
    for field in required_fields:
        value = section.get(field, None)

        if value is None:
            invalid_entries.append((i, field, "Missing or null"))
            break
        if isinstance(value, float) and math.isnan(value):
            invalid_entries.append((i, field, "NaN"))
            break

# 🧾 Report
if invalid_entries:
    print(f"⚠️ Found {len(invalid_entries)} invalid sections (null or NaN only).\n")
    for idx, field, reason in invalid_entries[:10]:  # First 10
        print(f"Index {idx} → Invalid: {field} ({reason})")

    with open("invalid_sections.txt", "w") as f:
        for idx, field, reason in invalid_entries:
            f.write(f"Index {idx} → Invalid: {field} ({reason})\n")
else:
    print("✅ No null or NaN values found in required fields. Your dataset is clean.")
