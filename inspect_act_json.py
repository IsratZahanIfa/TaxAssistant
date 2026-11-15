import json
from pathlib import Path
from collections import Counter

path = Path(r"E:\TaxAssist\data\Knowledge_Dataset\income_tax_act_2023.json")

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("JSON type:", type(data).__name__)

if isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))
    records = data.get("documents", data.get("data", data.get("records", [])))
elif isinstance(data, list):
    records = data
else:
    records = []

print("Total records:", len(records))

if records:
    print("\nFirst record:")
    print(json.dumps(records[0], ensure_ascii=False, indent=2)[:4000])

    print("\nField distribution:")
    counter = Counter()
    for record in records:
        if isinstance(record, dict):
            counter.update(record.keys())

    print(counter)
