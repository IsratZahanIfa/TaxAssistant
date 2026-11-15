"""
build_ita2023_en_dataset.py - rebuild the Income Tax Act, 2023 dataset (Authentic English Text,
S.R.O. No. 404-Law/2025, 8 October 2025) from the extracted text file.

Replaces the old income_tax_act_2023.json, which had these problems (checked on the real file):
  * sections 48-345 were labelled 'sch_FIFTH_..' (a 'Fifth Schedule' line inside section 47 was
    taken as the start of the schedules)
  * 32 sections were missing - e.g. '3[6. Delegation of powers' (a section that starts with an
    amendment mark) was not recognised
  * gazette header 'বাংলাদেশ গেজেট, অতিরিক্ত, অক্টোবর ১৬, ২০২৫' + page numbers glued to the law text,
    '=====' / 'PAGE n' lines, footnotes inside the law text, one-word lines
  * titles cut at the end of the PDF line; section 2 = one 40,000-character record

What this script does
  1. removes page furniture (separators, gazette header, page numbers, signature block)
  2. separates the FOOTNOTES at the bottom of each page (amendment history) from the law text
  3. finds PART / CHAPTER headings, the 345 sections (also those starting with '3[') and,
     only after section 345, the eight SCHEDULES with their parts and paragraphs
  4. joins broken lines into paragraphs; keeps the text with amendment marks ('text_marked')
     and without them ('text')
  5. splits every section into sub-sections / clauses
  6. writes
       data/Knowledge_Dataset/income_tax_act_2023.json          records (one per section / schedule paragraph)
       data/Knowledge_Dataset/income_tax_act_2023_chunks.json   chunks (>10 sub-sections -> one chunk each)
       data/Knowledge_Dataset/income_tax_act_2023_report.txt    checks: missing sections, sizes

Usage
  python build_ita2023_en_dataset.py --input Income_Tax_Act_2023_clean.txt
  (Income_Tax_Act_2023_unicode.txt or _raw.txt also work; do NOT use _unicode_fixed.txt - it is corrupted)
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

OUT_DIR = Path("data/Knowledge_Dataset")
SOURCE = (
    "Income Tax Act, 2023 (Act No. XII of 2023) - Authentic English Text, "
    "S.R.O. No. 404-Law/2025, Bangladesh Gazette Extraordinary, 16 October 2025"
)
LAST_SECTION = 345
MAX_SUBS = 10

PAGE_SEP = re.compile(r"^=+$")
PAGE_LINE = re.compile(r"^PAGE\s+(\d+)$")
GAZETTE_HEAD = re.compile(
    r"^বাংলাদেশ গেজেট,?\s*অতিরিক্ত,?\s*অক্টোবর\s*১৬,?\s*২০২৫\s*(\d{5})?\s*"
)
PAGE_NUMBER = re.compile(r"^\(?\d{5}\)?$")  # 10457, (10455)
FN_START = re.compile(r"^(\d{1,2}|\*{1,3})\s+(\S.*)$")
FN_WORDS = re.compile(
    r"\b(substituted|inserted|omitted|added|renumbered|re-numbered|deleted|replaced)\b",
    re.I,
)
SECTION_START = re.compile(r"^((?:\d{1,2}\s*\[\s*)*)(\d{1,3}[A-Z]{0,2})\.(?:\s+(.*))?$")
PART_LINE = re.compile(r"^PART[\s\-]+([IVXLC]+|\d{1,2})$")
CHAPTER_LINE = re.compile(r"^CHAPTER\s+([IVXLC]+|\d{1,2})$")
SCHEDULE_LINE = re.compile(
    r"^(?:THE\s+)?(FIRST|SECOND|THIRD|FOURT|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH)\s+SCHEDULE$"
)
SCHED_NAME = {
    "FIRST": "First",
    "SECOND": "Second",
    "THIRD": "Third",
    "FOURT": "Fourth",
    "FOURTH": "Fourth",
    "FIFTH": "Fifth",
    "SIXTH": "Sixth",
    "SEVENTH": "Seventh",
    "EIGHTH": "Eighth",
}
SCHED_NO = {
    "First": 1,
    "Second": 2,
    "Third": 3,
    "Fourth": 4,
    "Fifth": 5,
    "Sixth": 6,
    "Seventh": 7,
    "Eighth": 8,
}
PARA_START = re.compile(r"^((?:\d{1,2}\s*\[\s*)*)\(?(\d{1,3}[A-Z]?)[.)]\s+(.*)$")
HEAD_DASH = re.compile(r"^(.{2,220}?)[.:]?\s*[—–]\s*(.*)$", re.S)
CLAUSE = re.compile(
    r"^(?:\d{1,2}\[)?(\(\d{1,3}[A-Z]{0,2}\)|\([a-z]{1,4}\)|\([ivxl]{1,6}\)|Provided\b|Explanation\b)"
)
END_BLOCK = re.compile(r"^(By the Order of President|By order of the President)", re.I)
ROMAN = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
    "XIII": 13,
    "XIV": 14,
    "XV": 15,
    "XVI": 16,
    "XVII": 17,
    "XVIII": 18,
    "XIX": 19,
    "XX": 20,
    "XXI": 21,
    "XXII": 22,
    "XXIII": 23,
    "XXIV": 24,
    "XXV": 25,
}
AMENDING_LAW = re.compile(
    r"Finance Act,\s*\d{4}|Finance Ordinance,\s*\d{4}|Income Tax \(Amendment\) "
    r"(?:Act|Ordinance),\s*\d{4}"
)


def to_int(label):
    return int(label) if label.isdigit() else ROMAN.get(label, 0)


def num_key(no):
    m = re.match(r"(\d+)([A-Z]*)", no)
    return (int(m.group(1)), m.group(2)) if m else (0, "")


# ------------------------------------------------------------------ 1-2. pages, furniture, footnotes
def read_pages(path):
    pages, cur, no = [], [], 0
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        m = PAGE_LINE.match(line)
        if m:
            if cur or no:
                pages.append((no, cur))
            no, cur = int(m.group(1)), []
            continue
        if PAGE_SEP.match(line):
            continue
        cur.append(line)
    pages.append((no, cur))
    return [p for p in pages if p[0]]


def clean_page(lines):
    out = []
    for l in lines:
        l = l.replace("\uf0be", "—")  # Symbol-font em dash (51 times in the PDF)
        l = GAZETTE_HEAD.sub("", l).strip()
        if (
            not l
            or PAGE_NUMBER.match(l)
            or re.match(r"^মূল্য\s*:", l)
            or re.match(r"^রেজিস্টার্ড", l)
        ):
            continue
        if re.search(r"[ঀ-৿]", l) and not re.search(r"[A-Za-z]", l):
            continue  # Bangla gazette furniture / printer line
        out.append(l)
    return out


def split_footnotes(lines):
    """Footnotes are at the bottom of the page: '1 The word “Joint” was omitted by section 19 of the
    Finance Act, 2024 ...'. The first footnote line is '1 ...' or '* ...' and the block mentions an
    amendment word."""
    for i, l in enumerate(lines):
        m = FN_START.match(l)
        if not m or not (m.group(1) in ("1", "*") or i > len(lines) - 12):
            continue
        block = " ".join(lines[i:])
        if FN_WORDS.search(block) and re.search(r"\bAct\b|Ordinance", block):
            notes, cur = [], None
            for x in lines[i:]:
                mm = FN_START.match(x)
                if mm and (
                    cur is None
                    or mm.group(1) == "*" * len(mm.group(1))
                    or (
                        mm.group(1).isdigit()
                        and cur["no"].isdigit()
                        and int(mm.group(1)) == int(cur["no"]) + 1
                    )
                ):
                    cur = {"no": mm.group(1), "text": mm.group(2)}
                    notes.append(cur)
                elif cur is not None:
                    cur["text"] += (" " if not cur["text"].endswith("-") else "") + x
            return lines[:i], notes
    return lines, []


# ------------------------------------------------------------------ 3. structure
def strip_marks(text):
    """'2[(4) Where ...]' -> '(4) Where ...'; '1[***]' -> ''; normal brackets like '[See section 24]' kept."""
    out, stack, i = [], [], 0
    while i < len(text):
        m = (
            re.match(r"(?<![\w.])(\d{1,2})\s*\[", text[i:])
            if (i == 0 or not text[i - 1].isalnum())
            else None
        )
        if m:
            i += m.end()
            star = re.match(r"\s*\*\s*\*\s*\*\s*\]", text[i:])
            if star:
                i += star.end()
            else:
                stack.append("A")
            continue
        c = text[i]
        if c == "[":
            stack.append("N")
            out.append(c)
        elif c == "]" and stack:
            if stack.pop() == "N":
                out.append(c)
        else:
            out.append(c)
        i += 1
    s = re.sub(r"[ \t]{2,}", " ", "".join(out))
    s = re.sub(r" +([,;:.)])", r"\1", s)
    return "\n".join(l.strip() for l in s.split("\n") if l.strip())


def join_lines(lines):
    """PDF lines -> paragraphs. A new paragraph starts at a clause label '(1)', '(a)', '(i)',
    'Provided', 'Explanation'; a hyphen at the line end joins without a space."""
    paras = []
    for l in lines:
        if not paras or CLAUSE.match(l):
            paras.append(l)
        elif paras[-1].endswith("-") and not paras[-1].endswith(" -"):
            paras[-1] += l
        else:
            paras[-1] += " " + l
    return "\n".join(re.sub(r"\s+", " ", p).strip() for p in paras)


def heading_and_body(first, following, loose=False):
    """'12. Exercise of assessment functions of the Inspecting Additional' + 'Commissioner of Taxes.—(1) ...'
    -> (heading, body, extra lines used). The heading may run over several lines (justified text
    sometimes gives one word per line: '337.' / 'Proceeding' / 'against' ...).
    loose=True also accepts a heading that ends in '.' without the dash ('Officer.Any person') -
    used only when the section number is exactly the next one."""
    cand = first
    for used in range(0, 13):
        if used:
            if (
                used > len(following)
                or CLAUSE.match(following[used - 1])
                or SECTION_START.match(following[used - 1])
            ):
                break
            cand = (cand + " " + following[used - 1]).strip()
        m = HEAD_DASH.match(cand)
        if m and "(" not in m.group(1)[:2] and len(m.group(1).split()) <= 30:
            return m.group(1).strip().rstrip("."), m.group(2).strip(), used
        if loose:
            m = re.match(r"^([A-Z][^.()]{2,150}?)\.\s*(?=[A-Z(])(.*)$", cand, re.S)
            if m and len(m.group(1).split()) <= 20:
                return m.group(1).strip(), m.group(2).strip(), used
    return None


def build(pages):
    flat = []  # (page, line)
    footnotes = {}  # page -> [notes]
    for no, lines in pages:
        body, notes = split_footnotes(clean_page(lines))
        footnotes[no] = notes
        for l in body:
            if END_BLOCK.match(l):
                break
            flat.append((no, l))

    records, preamble = [], []
    part = part_title = chapter = chapter_title = ""
    schedule = sch_part = sch_part_title = sch_title = ""
    cur, expect, last_sec, in_sched = None, None, 0, False
    i = 0
    while i < len(flat):
        pg, line = flat[i]
        nxt = flat[i + 1][1] if i + 1 < len(flat) else ""

        m = SCHEDULE_LINE.match(line)
        if m and last_sec >= LAST_SECTION - 2:
            in_sched, schedule = True, SCHED_NAME[m.group(1)] + " Schedule"
            sch_part = sch_part_title = sch_title = ""
            cur = None
            expect = "sched_title"
            i += 1
            continue
        if expect in ("part_title", "chapter_title", "sched_part_title"):
            if expect == "part_title":
                part_title = line
            elif expect == "chapter_title":
                chapter_title = line
            else:
                sch_part_title = line
            expect = None
            i += 1
            continue
        if expect == "sched_title":
            if (line.isupper() or line.startswith("[")) and not PART_LINE.match(line):
                sch_title = (sch_title + " " + line).strip()
                i += 1
                continue
            expect = None

        m = PART_LINE.match(line)
        if m:
            if in_sched:
                sch_part, expect = f"Part {to_int(m.group(1))}", "sched_part_title"
                cur = None
            else:
                part, chapter, chapter_title, expect = (
                    f"Part {to_int(m.group(1))}",
                    "",
                    "",
                    "part_title",
                )
            i += 1
            continue
        m = CHAPTER_LINE.match(line)
        if m and not in_sched:
            chapter, expect = f"Chapter {to_int(m.group(1))}", "chapter_title"
            i += 1
            continue

        if not in_sched:
            m = SECTION_START.match(line)
            if m:
                n = num_key(m.group(2))
                ok = (
                    (n[0] == last_sec + 1 and not n[1])
                    or (n[0] == last_sec and n[1])
                    or (last_sec < n[0] <= last_sec + 3)
                )
                following = [x for _, x in flat[i + 1 : i + 14]]
                hb = (
                    heading_and_body(
                        m.group(3) or "", following, loose=(n[0] == last_sec + 1)
                    )
                    if ok
                    else None
                )
                if hb:
                    title, body, used = hb
                    cur = {
                        "record_type": "section",
                        "section": m.group(2),
                        "title": title,
                        "part": part,
                        "part_title": part_title,
                        "chapter": chapter,
                        "chapter_title": chapter_title,
                        "lines": [m.group(1) + body] if body or m.group(1) else [],
                        "pages": [pg],
                    }
                    records.append(cur)
                    last_sec = n[0]
                    i += 1 + used
                    continue
        else:
            m = PARA_START.match(line)
            if m:
                hb = heading_and_body(m.group(3), [x for _, x in flat[i + 1 : i + 14]])
                last = (
                    cur["paragraph"]
                    if cur
                    and cur.get("schedule") == schedule
                    and cur.get("schedule_part") == sch_part
                    and cur["paragraph"]
                    else "0"
                )
                if hb and 0 < num_key(m.group(2))[0] <= num_key(last)[0] + 3:
                    title, body, used = hb
                    cur = {
                        "record_type": "schedule_paragraph",
                        "schedule": schedule,
                        "schedule_title": sch_title,
                        "schedule_part": sch_part,
                        "schedule_part_title": sch_part_title,
                        "paragraph": m.group(2),
                        "title": title,
                        "lines": [m.group(1) + body] if body or m.group(1) else [],
                        "pages": [pg],
                    }
                    records.append(cur)
                    i += 1 + used
                    continue
            if cur is None:  # schedule part without numbered paragraphs
                cur = {
                    "record_type": "schedule_paragraph",
                    "schedule": schedule,
                    "schedule_title": sch_title,
                    "schedule_part": sch_part,
                    "schedule_part_title": sch_part_title,
                    "paragraph": "",
                    "title": sch_part_title or sch_title,
                    "lines": [],
                    "pages": [pg],
                }
                records.append(cur)

        if cur is None:
            preamble.append(line)
        else:
            cur["lines"].append(line)
            if cur["pages"][-1] != pg:
                cur["pages"].append(pg)
        i += 1
    return preamble, records, footnotes


# ------------------------------------------------------------------ 4-5. records and chunks
def split_subsections(text):
    """'(1) ... (2) ...' at the start of paragraphs -> lead-in + numbered parts (sequence checked)."""
    parts, last = [], 0
    starts = []
    for m in re.finditer(r"(?m)^(?:\d{1,2}\[)?\((\d{1,3})([A-Z]{0,2})\)\s", text):
        n = int(m.group(1))
        if last < n <= last + 2 or (n == last and m.group(2)):
            starts.append((m.start(), f"({m.group(1)}{m.group(2)})"))
            last = n
    if len(starts) < 2:
        return text.strip(), []
    lead = text[: starts[0][0]].strip()
    for k, (pos, label) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(text)
        parts.append({"label": label, "text": text[pos:end].strip()})
    return lead, parts


def finish(records, footnotes):
    out, ids = [], Counter()
    for r in records:
        marked = join_lines(r["lines"])
        clean = strip_marks(marked)
        marks = set(re.findall(r"(?<![\w.])(\d{1,2})\s*\[", marked)) | (
            {"*"} if "*" in marked else set()
        )
        notes = [
            {"page": p, "no": f["no"], "text": f["text"]}
            for p in r["pages"]
            for f in footnotes.get(p, [])
            if f["no"] in marks
        ]
        lead, subs = split_subsections(clean)
        if r["record_type"] == "section":
            rid = f"ita2023_s{r['section']}"
            ref = f"Section {r['section']}"
        else:
            sno = SCHED_NO.get(r["schedule"].split()[0], 0)
            rid = (
                f"ita2023_sch{sno}"
                + (
                    f"_{r['schedule_part'].replace(' ', '').lower()}"
                    if r["schedule_part"]
                    else ""
                )
                + (f"_p{r['paragraph']}" if r["paragraph"] else "_text")
            )
            ref = ", ".join(
                x
                for x in (
                    r["schedule"],
                    r["schedule_part"],
                    f"paragraph {r['paragraph']}" if r["paragraph"] else "",
                )
                if x
            )
        ids[rid] += 1
        if ids[rid] > 1:
            rid += f"_{ids[rid]}"
        rec = {
            "id": rid,
            "record_type": r["record_type"],
            "reference": ref,
            "section": r.get("section", ""),
            "title": r["title"],
            "part": r.get("part", ""),
            "part_title": r.get("part_title", ""),
            "chapter": r.get("chapter", ""),
            "chapter_title": r.get("chapter_title", ""),
            "schedule": r.get("schedule", ""),
            "schedule_title": r.get("schedule_title", ""),
            "schedule_part": r.get("schedule_part", ""),
            "schedule_part_title": r.get("schedule_part_title", ""),
            "paragraph": r.get("paragraph", ""),
            "text": clean,
            "text_marked": marked,
            "lead_in": lead,
            "subsections": subs,
            "subsection_count": len(subs),
            "footnotes": notes,
            "amended_by": sorted(
                {a for n in notes for a in AMENDING_LAW.findall(n["text"])}
            ),
            "page": r["pages"][0],
            "page_end": r["pages"][-1],
            "language": "en",
            "source": SOURCE,
        }
        out.append(rec)
    return out


def finish_order(records):
    return records


def make_chunks(records):
    chunks = []
    for r in records:
        if r.get("is_omitted"):
            continue
        if r["record_type"] == "section":
            ctx = " > ".join(
                x
                for x in (
                    "Income Tax Act, 2023",
                    f"{r['part']} {r['part_title']}".strip(),
                    f"{r['chapter']} {r['chapter_title']}".strip(),
                    f"Section {r['section']}",
                    r["title"],
                )
                if x
            )
            sub_word = "Clause" if r["section"] == "2" else "Sub-section"
        else:
            ctx = " > ".join(
                x
                for x in (
                    "Income Tax Act, 2023",
                    r["schedule"],
                    r["schedule_title"],
                    f"{r['schedule_part']} {r['schedule_part_title']}".strip(),
                    f"Paragraph {r['paragraph']}" if r["paragraph"] else "",
                    r["title"],
                )
                if x
            )
            sub_word = "Sub-paragraph"
        base = {
            k: r[k]
            for k in (
                "record_type",
                "reference",
                "section",
                "title",
                "part",
                "chapter",
                "schedule",
                "schedule_part",
                "paragraph",
                "page",
                "page_end",
                "amended_by",
                "source",
            )
        }
        if r["subsection_count"] <= MAX_SUBS:
            pieces = [("", r["text"])]
        else:
            pieces = ([("intro", r["lead_in"])] if r["lead_in"] else []) + [
                (s["label"], s["text"]) for s in r["subsections"]
            ]
        for label, text in pieces:
            c_ctx = ctx + (
                f" > {sub_word} {label}" if label and label != "intro" else ""
            )
            cid = r["id"] + (f"_{re.sub(r'[^0-9A-Za-z]', '', label)}" if label else "")
            chunks.append(
                {
                    "chunk_id": cid,
                    **base,
                    "subsection": label,
                    "context": c_ctx,
                    "text": text,
                    "context_and_text": f"{c_ctx}\n{text}",
                    "n_chars": len(text),
                }
            )
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        help="Income_Tax_Act_2023_clean.txt (or _unicode.txt / _raw.txt)",
    )
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    if "fixed" in Path(args.input).name:
        print(
            "WARNING: *_unicode_fixed.txt is corrupted (English letters turned into Bangla). Use _clean.txt."
        )

    pages = read_pages(args.input)
    preamble, recs, footnotes = build(pages)
    records = finish(recs, footnotes)
    # sections removed by a Finance Act: keep an entry saying so (from the footnote)
    have = {num_key(r["section"])[0] for r in records if r["record_type"] == "section"}
    for p, notes in footnotes.items():
        for f in notes:
            m = re.match(r"Section\s+(\d{1,3})\s+was\s+omitted", f["text"])
            if m and int(m.group(1)) not in have and int(m.group(1)) <= LAST_SECTION:
                n = m.group(1)
                records.append(
                    {
                        "id": f"ita2023_s{n}",
                        "record_type": "section",
                        "reference": f"Section {n}",
                        "section": n,
                        "title": "[Omitted]",
                        "part": "",
                        "part_title": "",
                        "chapter": "",
                        "chapter_title": "",
                        "schedule": "",
                        "schedule_title": "",
                        "schedule_part": "",
                        "schedule_part_title": "",
                        "paragraph": "",
                        "text": f"{n}. [Omitted.]",
                        "text_marked": f"{n}. [***]",
                        "lead_in": "",
                        "subsections": [],
                        "subsection_count": 0,
                        "is_omitted": True,
                        "footnotes": [{"page": p, **f}],
                        "amended_by": AMENDING_LAW.findall(f["text"]),
                        "page": p,
                        "page_end": p,
                        "language": "en",
                        "source": SOURCE,
                    }
                )
                have.add(int(n))
    records.sort(
        key=lambda r: (
            (
                r["record_type"] != "section",
                num_key(r["section"]) if r["section"] else (0, ""),
                r["id"],
            )
            if r["record_type"] == "section"
            else (True, (0, ""), "")
        )
    )
    records = [r for r in records if r["record_type"] == "section"] + [
        r for r in finish_order(records) if r["record_type"] != "section"
    ]
    for r in records:
        r.setdefault("is_omitted", False)
    chunks = make_chunks(records)

    secs = [r for r in records if r["record_type"] == "section"]
    nums = {num_key(r["section"])[0] for r in secs}
    missing = [n for n in range(1, LAST_SECTION + 1) if n not in nums]
    no_title = [r["section"] for r in secs if not r["title"]]
    sched = Counter(
        r["schedule"] for r in records if r["record_type"] == "schedule_paragraph"
    )
    big = sorted(chunks, key=lambda c: -c["n_chars"])[:5]
    report = [
        f"Pages read: {len(pages)}",
        f"Sections: {len(secs)} (expected {LAST_SECTION}); missing: {missing}",
        f"Sections without title: {no_title}",
        f"Schedule records: {dict(sched)}",
        f"Footnotes found: {sum(len(v) for v in footnotes.values())}; "
        f"records with amendment notes: {sum(1 for r in records if r['footnotes'])}",
        f"Records: {len(records)}  Chunks: {len(chunks)}",
        f"Largest chunks (chars): {[(c['chunk_id'], c['n_chars']) for c in big]}",
    ]
    print("\n".join(report))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "income_tax_act_2023.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "income_tax_act_2023_chunks.json").write_text(
        json.dumps(
            {
                "dataset": "income_tax_act_2023_en",
                "source": SOURCE,
                "rule": f"one chunk per section/paragraph; more than {MAX_SUBS} sub-sections -> one chunk each",
                "chunk_count": len(chunks),
                "chunks": chunks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "income_tax_act_2023_preamble.txt").write_text(
        "\n".join(preamble), encoding="utf-8"
    )
    (out / "income_tax_act_2023_report.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(
        f"Saved -> {out / 'income_tax_act_2023.json'}\n         {out / 'income_tax_act_2023_chunks.json'}"
    )


if __name__ == "__main__":
    main()
