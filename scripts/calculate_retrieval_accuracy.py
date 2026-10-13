import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "QA_Dataset"
    / "retrieval_review.json"
)


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

total = 0

top1_correct = 0
top3_correct = 0
top5_correct = 0

manually_validated = 0


for item in data:

    ground_truth = set(
        str(x)
        for x in item.get("ground_truth_sections", [])
    )

    if not ground_truth:
        continue

    manually_validated += 1

    candidates = item.get("candidates", [])

    retrieved_sections = [
        str(x.get("section"))
        for x in candidates
    ]

    total += 1

    if any(
        section in ground_truth
        for section in retrieved_sections[:1]
    ):
        top1_correct += 1

    if any(
        section in ground_truth
        for section in retrieved_sections[:3]
    ):
        top3_correct += 1

    # Top-5
    if any(
        section in ground_truth
        for section in retrieved_sections[:5]
    ):
        top5_correct += 1

print("=" * 70)
print("TaxAssist - Retrieval Accuracy")
print("=" * 70)

print(f"\nValidated questions: {total}")

if total == 0:

    print("""
No ground-truth sections have been added yet.

Open:

data/QA_Dataset/retrieval_review.json

For example:

"ground_truth_sections": ["166"]

Then run this script again.
""")

else:

    top1 = top1_correct / total * 100
    top3 = top3_correct / total * 100
    top5 = top5_correct / total * 100

    print("\nRESULTS")
    print("-" * 50)

    print(
        f"Top-1 Accuracy : "
        f"{top1_correct}/{total} "
        f"({top1:.2f}%)"
    )

    print(
        f"Top-3 Accuracy : "
        f"{top3_correct}/{total} "
        f"({top3:.2f}%)"
    )

    print(
        f"Top-5 Accuracy : "
        f"{top5_correct}/{total} "
        f"({top5:.2f}%)"
    )

    print("\nInterpretation:")

    if top5 >= 90:
        print("✓ Excellent candidate retrieval for RAG.")

    elif top5 >= 75:
        print("✓ Good candidate retrieval, but improvement is recommended.")

    else:
        print("⚠ Retrieval needs improvement before LLM integration.")

print("\nDONE")