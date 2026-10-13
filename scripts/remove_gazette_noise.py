import json
import re
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

INPUT_FILE = Path(
    "data/Knowledge_Dataset/income_tax_act_2023.json"
)

OUTPUT_FILE = Path(
    "data/Knowledge_Dataset/income_tax_act_2023_clean.json"
)


# ---------------------------------------------------------
# Gazette noise patterns
# ---------------------------------------------------------

def remove_gazette_noise(text):
    if not text:
        return ""

    # Remove long separator lines
    text = re.sub(
        r'={5,}',
        ' ',
        text
    )

    # Remove Bangladesh Gazette header
    text = re.sub(
        r'বাংলাদেশ গেজেট,\s*অতিরিক্ত,\s*অক্টোবর\s*১৬,\s*২০২৫',
        ' ',
        text
    )

    # Remove English Gazette header if present
    text = re.sub(
        r'Bangladesh Gazette,\s*Extraordinary,\s*October\s*16,\s*2025',
        ' ',
        text,
        flags=re.IGNORECASE
    )

    # Remove page numbers that appear as standalone numbers
    # Example:
    # 10458
    # 10459
    # 10460
    text = re.sub(
        r'(?m)^\s*\d{4,6}\s*$',
        ' ',
        text
    )

    # Remove PART headings that are PDF/Gazette navigation noise
    # Be careful: this removes only standalone PART headings.
    text = re.sub(
        r'(?m)^\s*PART\s+\d+\s*$',
        ' ',
        text,
        flags=re.IGNORECASE
    )

    # Remove repeated page separator artifacts
    text = re.sub(
        r'\s*={5,}\s*',
        '\n',
        text
    )

    # Normalize excessive blank lines
    text = re.sub(
        r'\n\s*\n\s*\n+',
        '\n\n',
        text
    )

    # Normalize excessive spaces
    text = re.sub(
        r'[ \t]+',
        ' ',
        text
    )

    return text.strip()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

print("=" * 70)
print("TaxAssist - Remove Gazette Noise")
print("=" * 70)

print(f"Input : {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")

print("\nLoading dataset...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Records loaded: {len(data)}")

cleaned_records = []

for record in data:

    new_record = record.copy()

    original_text = new_record.get("text", "")

    cleaned_text = remove_gazette_noise(original_text)

    new_record["text"] = cleaned_text

    cleaned_records.append(new_record)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        cleaned_records,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n" + "=" * 70)
print("CLEANING COMPLETED")
print("=" * 70)

print(f"Records: {len(cleaned_records)}")
print(f"Saved to: {OUTPUT_FILE}")