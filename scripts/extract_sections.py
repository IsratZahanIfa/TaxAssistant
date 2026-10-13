"""
extract_sections.py (FIXED)
============================
Two critical fixes vs the original:

1. DEDUPLICATE the source text. The raw "clean" text file contains the
   full Act repeated multiple times back-to-back (verified: "1. Short
   title and commencement" appears 4 separate times in the source).
   We detect repeats of the very first section's title line and only
   keep the text up through the END of the first complete copy
   (main Act body + its Schedules), discarding the rest.

2. STOP Schedule paragraphs from colliding with main Act section
   numbers. Third/Fourth/Fifth Schedule etc. each have their OWN
   internal numbering that restarts at 1, 2, 3... Previously these
   were being extracted as if they were "Section 1", "Section 2" of
   the main Act (verified: "Section 1" matched 122 times, "Section 2"
   124 times, when the real Act has exactly one of each). We now
   track whether we're inside the main Act body or inside a Schedule,
   and give Schedule paragraphs a clearly separate id/section label
   (schedule_no + para_no) so they can never be confused with a main
   Act section during retrieval or evaluation matching.
"""

from pathlib import Path
import json
import re

INPUT_FILE = Path("data/Clean_Text/Income_Tax_Act_2023_clean.txt")
OUTPUT_FILE = Path("data/Knowledge_Dataset/income_tax_act_2023.json")

SCHEDULE_HEADING_RE = re.compile(
    r"^(?:THE\s+)?(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|"
    r"NINTH|TENTH)\s+SCHEDULE\s*$",
    re.IGNORECASE,
)

SECTION_ONE_TITLE_MARKER = "short title"  # used to detect repeated copies


def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def deduplicate_source(raw_text):
    """
    The source file contains the whole Act repeated back-to-back.
    Keep only the first complete copy: from the start up to (but not
    including) the second occurrence of the Section-1 heading line.
    """
    lines = raw_text.splitlines()
    marker_hits = []
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped.startswith("1.") and SECTION_ONE_TITLE_MARKER in stripped:
            marker_hits.append(i)

    if len(marker_hits) <= 1:
        print("No duplicate copies detected in source text.")
        return raw_text

    print(
        f"WARNING: source text contains {len(marker_hits)} copies of the Act "
        f"(Section-1 heading found at lines {marker_hits}). "
        f"Keeping only the first copy."
    )
    second_copy_start = marker_hits[1]
    deduped = "\n".join(lines[:second_copy_start])
    return deduped


def is_section_heading(line):
    line = line.strip()
    pattern = r"^(\d{1,3})\.\s+(.+)$"
    match = re.match(pattern, line)
    if not match:
        return None
    section_number = int(match.group(1))
    title = match.group(2).strip()
    if not (1 <= section_number <= 345):
        return None
    if title.startswith("("):
        return None
    return section_number, title


def is_schedule_heading(line):
    line = line.strip()
    m = SCHEDULE_HEADING_RE.match(line)
    return m.group(1).upper() if m else None


def is_real_chapter_heading(line):
    line = line.strip()
    return bool(re.match(r"^CHAPTER\s+[IVXLCDM]+$", line, re.IGNORECASE))


def is_real_part_heading(line):
    line = line.strip()
    return bool(re.match(r"^PART\s+[IVXLCDM0-9]+$", line, re.IGNORECASE))


def is_page_marker(line):
    return bool(re.match(r"^PAGE\s+\d+$", line, re.IGNORECASE))


def extract_sections(text):
    lines = text.splitlines()
    sections = []

    current_section = None
    current_page = None
    current_chapter = None
    current_part = None
    buffer = []

    # NEW: track whether we're inside the main Act body or a Schedule
    in_schedule = False
    current_schedule_name = None
    schedule_para_counter = 0

    def flush_main_section():
        nonlocal current_section, buffer
        if current_section is not None:
            section_text = clean_text("\n".join(buffer))
            if section_text:
                sections.append({
                    "section": str(current_section["section"]),
                    "record_type": "act_section",
                    "title": current_section["title"],
                    "chapter": current_section["chapter"],
                    "part": current_section["part"],
                    "page": current_section["page"],
                    "source": "Income Tax Act, 2023",
                    "text": section_text,
                })
        current_section = None
        buffer = []

    def flush_schedule_para(section_info, title):
        nonlocal schedule_para_counter, buffer
        para_text = clean_text("\n".join(buffer))
        if para_text:
            schedule_para_counter += 1
            sections.append({
                # IMPORTANT: distinct namespace, never collides with
                # main Act section numbers like "82".
                "section": f"sch_{current_schedule_name}_{section_info}",
                "record_type": "schedule_paragraph",
                "title": title,
                "chapter": current_chapter,
                "part": current_part,
                "page": current_page,
                "schedule": current_schedule_name,
                "source": "Income Tax Act, 2023",
                "text": para_text,
            })
        buffer = []

    pending_schedule_para = None  # (number, title)

    for line in lines:
        line = line.strip()

        if not line:
            if current_section is not None or in_schedule:
                buffer.append("")
            continue

        if is_page_marker(line):
            current_page = int(re.search(r"\d+", line).group())
            continue

        sched_name = is_schedule_heading(line)
        if sched_name:
            # entering (or moving to a new) Schedule: flush whatever
            # main-Act section or previous schedule paragraph was open
            if not in_schedule:
                flush_main_section()
            elif pending_schedule_para is not None:
                flush_schedule_para(*pending_schedule_para)
                pending_schedule_para = None
            in_schedule = True
            current_schedule_name = sched_name
            continue

        if is_real_chapter_heading(line):
            current_chapter = line
            if current_section is not None:
                buffer.append(line)
            continue

        if is_real_part_heading(line):
            current_part = line
            if current_section is not None:
                buffer.append(line)
            continue

        if not in_schedule:
            section_info = is_section_heading(line)
            if section_info:
                section_number, title = section_info
                flush_main_section()
                current_section = {
                    "section": section_number,
                    "title": title,
                    "chapter": current_chapter,
                    "part": current_part,
                    "page": current_page,
                }
                buffer = []
                continue
            if current_section is not None:
                buffer.append(line)
        else:
            # inside a Schedule: numbered paragraphs get their OWN
            # namespace (schedule + local number), never "section".
            para_info = is_section_heading(line)  # same "N. Title" shape
            if para_info:
                if pending_schedule_para is not None:
                    flush_schedule_para(*pending_schedule_para)
                para_number, title = para_info
                pending_schedule_para = (para_number, title)
                buffer = []
                continue
            buffer.append(line)

    # flush whatever is left open at EOF
    if not in_schedule:
        flush_main_section()
    elif pending_schedule_para is not None:
        flush_schedule_para(*pending_schedule_para)

    return sections


def remove_duplicate_sections(sections):
    unique = []
    seen = set()
    for item in sections:
        key = (item["section"], item["title"], item.get("page"), item["text"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def add_ids(sections):
    for item in sections:
        if item["record_type"] == "act_section":
            item["id"] = f"ita_2023_sec_{item['section']}_page_{item['page']}"
        else:
            item["id"] = f"ita_2023_{item['section']}_page_{item['page']}"
    return sections


def main():
    print("=" * 70)
    print("TaxAssist - Final Income Tax Act Dataset Builder (FIXED)")
    print("=" * 70)

    if not INPUT_FILE.exists():
        print("ERROR: Input file not found:")
        print(INPUT_FILE)
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    print("Reading document...")
    text = INPUT_FILE.read_text(encoding="utf-8")
    print(f"Characters (raw)  : {len(text):,}")

    text = deduplicate_source(text)
    print(f"Characters (deduped): {len(text):,}")

    print()
    print("Extracting genuine Act sections and Schedule paragraphs...")
    sections = extract_sections(text)
    print(f"Extracted records: {len(sections):,}")

    act_secs = [s for s in sections if s["record_type"] == "act_section"]
    sched_paras = [s for s in sections if s["record_type"] == "schedule_paragraph"]
    print(f"  - Act sections      : {len(act_secs):,} (should be <= 345, ideally close to it)")
    print(f"  - Schedule paragraphs: {len(sched_paras):,}")

    unique_act_sec_numbers = sorted(set(s["section"] for s in act_secs), key=lambda x: int(x))
    print(f"  - Unique Act section numbers: {len(unique_act_sec_numbers):,}")
    dup_check = {}
    for s in act_secs:
        dup_check[s["section"]] = dup_check.get(s["section"], 0) + 1
    dupes = {k: v for k, v in dup_check.items() if v > 1}
    if dupes:
        print(f"  - WARNING: still-duplicated section numbers: {dupes}")
    else:
        print("  - No duplicate main-Act section numbers. ✔")

    before = len(sections)
    sections = remove_duplicate_sections(sections)
    after = len(sections)
    print(f"\nExact duplicates removed: {before - after}")

    sections = add_ids(sections)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print("DATASET CREATED")
    print("=" * 70)
    print(f"Total records : {len(sections):,}")
    print(f"Saved to      : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()