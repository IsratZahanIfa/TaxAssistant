import json
from pathlib import Path

INPUT_FILE = Path("data/QA_Dataset/retrieval_review.json")
OUTPUT_FILE = Path("data/QA_Dataset/retrieval_review_annotated.json")
LAW_FILE = Path("data/Knowledge_Dataset/income_tax_act_2023.json")

def load_law_data():
    if not LAW_FILE.exists():
        print(f"⚠️ Law file not found: {LAW_FILE}")
        return {}
    try:
        with open(LAW_FILE, "r", encoding="utf-8") as f:
            law_data = json.load(f)
        section_map = {}
        for item in law_data:
            section = str(item.get("section", "")).strip()
            if section:
                text = item.get("text", "")
                title = item.get("title", "")
                part = item.get("part", "")
                chapter = item.get("chapter", "")
                section_map[section] = {
                    "text": text[:800] + "..." if len(text) > 800 else text,
                    "title": title,
                    "part": part,
                    "chapter": chapter
                }
        print(f"✅ Loaded {len(section_map)} sections from law file")
        return section_map
    except Exception as e:
        print(f"⚠️ Error loading law file: {e}")
        return {}

def load_data():
    if not INPUT_FILE.exists():
        print("=" * 70)
        print("ERROR: File not found:", INPUT_FILE)
        return None
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("ERROR: Expected JSON list.")
            return None
        print("=" * 70)
        print("📚 TaxAssist - Retrieval Ground Truth Annotation")
        print("=" * 70)
        print(f"Questions loaded: {len(data)}")
        return data
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON file. {e}")
        return None

def display_question(item, law_data):
    print("\n" + "=" * 70)
    print(f"QUESTION {item.get('number')} ({item.get('question_id')})")
    print("=" * 70)
    print(f"\nQuestion:\n{item.get('question')}")
    print(f"\nLanguage: {item.get('language')}")
    
    print("\n" + "-" * 70)
    print("TOP CANDIDATES (with text preview)")
    print("-" * 70)
    
    for candidate in item.get("candidates", []):
        rank = candidate.get("rank")
        section = candidate.get("section")
        title = candidate.get("title")
        page = candidate.get("page")
        score = candidate.get("relevance_score")
        
        print(f"\nRank {rank} | Section {section} | Page {page} | Score {score}")
        print(f"Title: {title}")
        
        if section and str(section) in law_data:
            law_info = law_data[str(section)]
            print(f"Text preview: {law_info['text'][:300]}...")
        else:
            print("Text preview: [Not available in law file]")
    
    print("\n" + "-" * 70)

def parse_sections(user_input):
    user_input = user_input.strip()
    if not user_input:
        return []
    if user_input.lower() in ["skip", "s"]:
        return None
    if user_input.lower() in ["quit", "q", "exit"]:
        return "QUIT"
    user_input = user_input.replace("[", "").replace("]", "")
    parts = user_input.split(",")
    sections = []
    for part in parts:
        section = part.strip()
        if not section:
            continue
        if section.lower().startswith("section"):
            section = section[7:].strip()
        if not section.isdigit():
            print(f"Invalid section number: '{section}'")
            return "INVALID"
        sections.append(section)
    return list(dict.fromkeys(sections))

def save_data(data):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Annotation saved: {OUTPUT_FILE}")
    print(f"Total questions: {len(data)}")

def annotate():
    data = load_data()
    if data is None:
        return
    law_data = load_law_data()
    already_done = sum(1 for item in data if item.get("ground_truth_sections", []))
    print(f"Already annotated: {already_done}/{len(data)}")
    
    print("\n" + "=" * 70)
    print("INSTRUCTIONS")
    print("=" * 70)
    print("\nFor each question, enter the correct legal section(s).")
    print("\nExamples: 166")
    print("          166,169")
    print("          [166,169]")
    print("\nCommands:")
    print("  s / skip  -> skip this question")
    print("  q / quit  -> save and exit")
    print("\n" + "=" * 70)

    for index, item in enumerate(data):
        if item.get("ground_truth_sections", []):
            continue
        display_question(item, law_data)
        while True:
            try:
                user_input = input("Correct section(s): ")
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                save_data(data)
                return
            result = parse_sections(user_input)
            if result == "QUIT":
                save_data(data)
                return
            if result == "INVALID":
                continue
            if result is None:
                item["ground_truth_sections"] = []
                print(f"\nSkipped Q{item.get('number')}")
                break
            if not result:
                print("\nPlease enter at least one section.")
                continue
            candidate_sections = [str(c.get("section")) for c in item.get("candidates", [])]
            not_in_candidates = [s for s in result if s not in candidate_sections]
            if not_in_candidates:
                print(f"\n⚠️ WARNING: Section(s) {', '.join(not_in_candidates)} NOT in Top-5 candidates.")
                print("You can still enter them if verified from the Income Tax Act.")
                confirm = input("Are you sure? (y/n): ").strip().lower()
                if confirm != "y":
                    continue
            item["ground_truth_sections"] = result
            for candidate in item.get("candidates", []):
                candidate["correct"] = str(candidate.get("section")) in result
            print(f"\n✅ Q{item.get('number')} annotated: {result}")
            save_data(data)
            break

    total = sum(1 for item in data if item.get("ground_truth_sections", []))
    print(f"\nAnnotated questions: {total}/{len(data)}")
    print("DONE")

if __name__ == "__main__":
    annotate()