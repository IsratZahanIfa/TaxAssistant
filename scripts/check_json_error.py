import json
from pathlib import Path

FILE = Path("data/Knowledge_Dataset/income_tax_act_2023.json")

print("Checking:", FILE)

try:
    with open(FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("JSON is valid!")
    print("Records:", len(data))

except json.JSONDecodeError as e:
    print("\nJSON ERROR")
    print("-" * 60)
    print("Message :", e.msg)
    print("Line    :", e.lineno)
    print("Column  :", e.colno)
    print("Position:", e.pos)

    lines = FILE.read_text(encoding="utf-8").splitlines()

    start = max(0, e.lineno - 6)
    end = min(len(lines), e.lineno + 5)

    print("\nContext:")
    print("-" * 60)

    for i in range(start, end):
        marker = "  >>>" if i + 1 == e.lineno else "     "
        print(f"{marker} {i + 1}: {lines[i]}")