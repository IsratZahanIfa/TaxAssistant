from pathlib import Path
import json
import re

INPUT_FILE = Path("data/Knowledge_Dataset/income_tax_act_2023.json")

def contains_mojibake(text):
    bad_patterns = ["à¦", "à§", "â€™", "â€œ", "â€", "Ã", "ï‚¾"]
    return any(pattern in text for pattern in bad_patterns)

def contains_bangla(text):
    return bool(re.search(r"[\u0980-\u09FF]", text))

def contains_english(text):
    return bool(re.search(r"[A-Za-z]", text))

def main():
    print("=" * 70)
    print("TaxAssist - Dataset Validation")
    print("=" * 70)

    if not INPUT_FILE.exists():
        print(f"ERROR: Dataset not found: {INPUT_FILE}")
        return

    print(f"Loading dataset...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total records: {len(data):,}")

    missing_section = []
    missing_title = []
    missing_text = []
    missing_page = []
    duplicate_sections = []
    mojibake_records = []
    bangla_records = []
    english_records = []
    short_records = []
    section_numbers = []

    for i, record in enumerate(data):
        section = record.get("section")
        title = record.get("title")
        text = record.get("text")
        page = record.get("page")

        if not section:
            missing_section.append(i)
        else:
            section_numbers.append(str(section))

        if not title:
            missing_title.append(i)
        if not text or not text.strip():
            missing_text.append(i)
        if page is None:
            missing_page.append(i)

        combined_text = str(title) + " " + str(text)
        if contains_mojibake(combined_text):
            mojibake_records.append(section)
        if contains_bangla(combined_text):
            bangla_records.append(section)
        if contains_english(combined_text):
            english_records.append(section)
        if text and len(text.strip()) < 50:
            short_records.append(section)

    seen = set()
    for section in section_numbers:
        if section in seen:
            duplicate_sections.append(section)
        seen.add(section)
    duplicate_sections = sorted(set(duplicate_sections), key=lambda x: int(x) if x.isdigit() else 0)

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"Total records       : {len(data):,}")
    print(f"Missing section     : {len(missing_section):,}")
    print(f"Missing title       : {len(missing_title):,}")
    print(f"Missing text        : {len(missing_text):,}")
    print(f"Missing page        : {len(missing_page):,}")
    print(f"Duplicate sections  : {len(duplicate_sections):,}")
    print(f"Mojibake records    : {len(mojibake_records):,}")
    print(f"Bangla records      : {len(bangla_records):,}")
    print(f"English records     : {len(english_records):,}")
    print(f"Very short records  : {len(short_records):,}")

    if duplicate_sections:
        print(f"\nDuplicate sections: {', '.join(duplicate_sections[:50])}")
    if mojibake_records:
        print(f"\nPossible mojibake sections: {', '.join(map(str, mojibake_records[:50]))}")
    if short_records:
        print(f"\nVery short sections: {', '.join(map(str, short_records[:50]))}")

    problems = (len(missing_section) + len(missing_title) + len(missing_text) + 
                len(missing_page) + len(mojibake_records))
    print("\n" + "=" * 70)
    print("VALIDATION STATUS: PASSED" if problems == 0 else "VALIDATION STATUS: NEEDS REVIEW")
    print("=" * 70)

if __name__ == "__main__":
    main()