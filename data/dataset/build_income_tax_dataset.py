# -*- coding: utf-8 -*-
"""
build_income_tax_dataset.py   (version 2.2)
-----------------------------------------
Turns the bdcode.gov.bd income-tax PDF into an easy-to-read JSON dataset.

NOTE: the PDF at the bdcode link is the "Income-tax Ordinance, 1984" (as amended),
not the "Income Tax Act, 2023". Names below are set for the Ordinance.
Use --law-name / --law-code if you run this on another law.

The contents list (printed pages 128-143 = PDF pages 1-16) is skipped.
The law is read from printed page 144 = PDF page 17 (change with --start-page).

One record per section, with:
  - section number (also with letters: 4A, 16BBB, 19DD), title, chapter
  - clean text and sub-sections labelled   4. (A)   (B)   (C) ...
  - amendment notes (the small footnotes at the bottom of the PDF pages)
  - page numbers, source link
  - optional easy-language explanation from your local Ollama (--simplify)

Install once:
    pip install pymupdf requests

Run (PowerShell):
    python -u build_income_tax_dataset.py --pdf-path "E:\\TaxAssist\\data\\legal\\income_tax_act_2023\\income_tax_act_2023.pdf"
    python -u build_income_tax_dataset.py --debug
    python -u build_income_tax_dataset.py --simplify --simple-lang en
"""

import argparse
import base64
import json
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path

import requests

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
PDF_URL = "http://bdcode.gov.bd/upload/bdcodeact/2023-12-24-11-21-06-12.-Income-tax-final.pdf"
DEFAULT_OUT_DIR = r"E:\TaxAssist\data\legal\income_tax_ordinance_1984"
DEFAULT_LAW_NAME = "Income-tax Ordinance, 1984"
DEFAULT_LAW_CODE = "ITO1984"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"

# How sub-sections are labelled in the JSON:
#   "letters"  -> 4. (A), 4. (B), 4. (C) ...   (original (1),(2) kept in "original_label")
#   "original" -> keep the law's own labels (1), (2) ...
SUBSECTION_STYLE = "letters"

# This PDF: the contents list is printed pages 128-143 = PDF pages 1-16 (printed number = PDF page + 127).
# The law itself starts at printed page 144 = PDF page 17.
DEFAULT_START_PAGE = 17

# PDFs you may already have (checked before downloading)
OLD_PDF_LOCATIONS = [r"E:\TaxAssist\data\legal\income_tax_act_2023\income_tax_act_2023.pdf"]

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹"


# --------------------------------------------------------------------------
# 1. Download
# --------------------------------------------------------------------------
def download_pdf(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"[1/5] PDF already downloaded: {dest}")
        return dest
    print("[1/5] Downloading PDF ...")
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=180)
    except requests.exceptions.SSLError:
        r = requests.get(url, headers=headers, timeout=180, verify=False)
    r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise RuntimeError("The link did not return a PDF. Download it by hand and use --pdf-path.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    print(f"      saved {len(r.content)/1_000_000:.1f} MB -> {dest}")
    return dest


def pick_pdf(pdf_path_arg, url: str, out_dir: Path) -> Path:
    """--pdf-path if given; else a PDF you already have; else download it."""
    if pdf_path_arg:
        return Path(pdf_path_arg)
    for c in [out_dir / "income_tax_law.pdf"] + [Path(x) for x in OLD_PDF_LOCATIONS]:
        if c.exists() and c.stat().st_size > 10_000:
            print(f"[1/5] Using PDF you already have: {c}")
            return c
    return download_pdf(url, out_dir / "income_tax_law.pdf")


# --------------------------------------------------------------------------
# 2. Read the PDF (normal text with font sizes; OCR only when a page is broken)
# --------------------------------------------------------------------------
def looks_broken(text: str) -> bool:
    t = text.strip()
    if len(t) < 40:
        return True
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return True
    good = sum(1 for c in letters if ("A" <= c <= "z") or ("\u0980" <= c <= "\u09FF"))
    odd = sum(1 for c in t if "\u0080" <= c <= "\u00FF" or c in "‡¤†•")
    return good / len(letters) < 0.85 or odd / len(t) > 0.02


def ollama_ocr(png_bytes: bytes, model: str) -> str:
    prompt = (
        "Transcribe all the text on this page exactly as written. Keep the original "
        "language (Bangla or English), keep section numbers and line breaks. "
        "Do not translate, summarise or add anything. Output only the text."
    )
    payload = {"model": model, "prompt": prompt, "images": [base64.b64encode(png_bytes).decode()],
               "stream": False, "options": {"temperature": 0}}
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json().get("response", "")


def ollama_ready(model: str):
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        names = [m.get("name", "") for m in r.json().get("models", [])]
        return any(n == model or n == model + ":latest" for n in names), True, names
    except Exception:
        return False, False, []


def page_rows(page):
    """Lines of one page with font size and vertical position (0 = top, 1 = bottom)."""
    d = page.get_text("dict", sort=True)
    height = page.rect.height or 1
    rows = []
    for block in d["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            rows.append({
                "text": "".join(s["text"] for s in spans),
                "size": statistics.median([s["size"] for s in spans]),
                "y": line["bbox"][1] / height,
            })
    return rows


def printed_page_number(rows):
    """Page number printed at the top/bottom edge of a page (to double-check the start page)."""
    for r in rows or []:
        t = r["text"].strip()
        if (r["y"] < 0.09 or r["y"] > 0.94) and re.fullmatch(r"[0-9]{1,4}", t):
            return t
    return None


def extract_pages(pdf_path: Path, force_ocr: bool, model: str, cache_dir: Path,
                  no_ocr: bool = False, dpi: int = 200, start_page: int = 1, end_page=None):
    import fitz  # PyMuPDF

    cache_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages, ocr_count, skipped = [], 0, 0
    model_ok, server_ok, _ = ollama_ready(model)
    last = min(end_page or len(doc), len(doc))
    print(f"[2/5] PDF has {len(doc)} pages. Reading PDF pages {start_page} to {last} "
          f"(earlier pages, e.g. the contents list, are skipped) ...")
    for i, page in enumerate(doc, start=1):
        if i < start_page or i > last:
            continue
        rows = page_rows(page)
        text = "\n".join(r["text"] for r in rows)
        method = "text"
        if (force_ocr or looks_broken(text)) and not no_ocr:
            cache_file = cache_dir / f"page_{i:04d}.txt"
            if cache_file.exists():
                text, rows, method = cache_file.read_text(encoding="utf-8"), None, "ocr-cache"
            else:
                if not (server_ok and model_ok):
                    print("\nERROR: This PDF needs OCR, but Ollama is not ready.")
                    print("  - Start Ollama (app, or: ollama serve) and run: ollama pull " + model)
                    print(f"  - Finished OCR pages are saved in: {cache_dir}")
                    print("  - Or run with --no-ocr to use only the normal text of the PDF.")
                    sys.exit(1)
                text = ollama_ocr(page.get_pixmap(dpi=dpi).tobytes("png"), model)
                cache_file.write_text(text, encoding="utf-8")
                rows, method = None, "ocr"
                ocr_count += 1
                print(f"      page {i}/{len(doc)} OCR done")
        elif looks_broken(text):
            skipped += 1
        pages.append({"page": i, "text": text, "rows": rows, "method": method})
        if len(pages) == 1:
            pn = printed_page_number(rows)
            print(f"      first page used: PDF page {i}, printed page number: {pn or 'not found'}")
    print(f"      done. New OCR pages: {ocr_count}. Pages with poor text left as is: {skipped}.")
    if not pages:
        print("ERROR: no pages in the chosen range. Check --start-page / --end-page.")
        sys.exit(1)
    return pages


# --------------------------------------------------------------------------
# 3. Clean text   (lines are tuples: (page, text, is_footnote))
# --------------------------------------------------------------------------
def tidy_line(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("\u00ad", "")
    # this PDF's font prints opening quote as ― and closing quote as ‖
    s = s.replace("‖", "”")
    s = re.sub(r"(^|[\s(\[])―(?=\S)", lambda m: m.group(1) + "“", s)
    s = s.replace("―", "—")
    return re.sub(r"[ \t]+", " ", s).strip()


def estimate_body_font(pages):
    sizes = Counter()
    for p in pages:
        for r in p["rows"] or []:
            if 0.08 < r["y"] < 0.82 and len(r["text"]) > 22:
                sizes[round(r["size"] * 2) / 2] += 1
    return sizes.most_common(1)[0][0] if sizes else None


def furniture_key(s):
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", s.lower())).strip()


def clean_pages(pages, body_font, footnote_ratio=0.86):
    n_pages = max(1, len(pages))
    edge = lambda y: y < 0.09 or y > 0.94
    # headers/footers = lines at the page edge that repeat on many pages
    edge_count = Counter()
    for p in pages:
        if p["rows"]:
            edge_count.update({furniture_key(tidy_line(r["text"])) for r in p["rows"] if edge(r["y"])})
    repeated = {k for k, c in edge_count.items() if c >= max(4, int(n_pages * 0.015)) and len(k) < 125}
    # for OCR pages (no positions): short lines repeated on many pages
    ocr_count = Counter()
    for p in pages:
        if not p["rows"]:
            ocr_count.update({furniture_key(tidy_line(l)) for l in p["text"].splitlines() if 0 < len(l) < 80})
    ocr_repeated = {k for k, c in ocr_count.items() if c >= max(5, int(n_pages * 0.30))}

    out = []
    for p in pages:
        if p["rows"]:
            items = [(tidy_line(r["text"]), r) for r in p["rows"]]
        else:
            items = [(tidy_line(l), None) for l in p["text"].splitlines()]
        for text, r in items:
            if not text:
                continue
            note = False
            if r is not None:
                if edge(r["y"]) and (furniture_key(text) in repeated or re.fullmatch(r"[–—-]?\s*\d{1,4}\s*[–—-]?", text)):
                    continue
                note = bool(body_font and r["size"] < body_font * footnote_ratio and r["y"] > 0.5)
                if note and re.fullmatch(r"\d{1,3}", text):          # footnote marker alone
                    continue
            else:
                if furniture_key(text) in ocr_repeated or re.fullmatch(r"[0-9০-৯]{1,4}", text):
                    continue
                if re.fullmatch(r"(?i)page\s*\d+( of \d+)?", text):
                    continue
            out.append((p["page"], text, note))
    return merge_orphan_numbers(out)


ORPHAN_RE = re.compile(r"^[0-9০-৯]{1,3}[A-Z]{0,6}\s*[।.]$")


def merge_orphan_numbers(lines):
    """'16A.' alone on a line + title on next line  ->  '16A. Title'"""
    out, i = [], 0
    while i < len(lines):
        pg, t, note = lines[i]
        if (ORPHAN_RE.match(t) and i + 1 < len(lines) and lines[i + 1][0] == pg
                and not note and not lines[i + 1][2]):
            out.append((pg, t + " " + lines[i + 1][1], False))
            i += 2
        else:
            out.append((pg, t, note))
            i += 1
    return out


def detect_language(lines) -> str:
    sample = " ".join(t for _, t, _ in lines[:2000])
    bn = sum(1 for c in sample if "\u0980" <= c <= "\u09FF")
    en = sum(1 for c in sample if "a" <= c.lower() <= "z")
    return "bn" if bn > en else "en"


# --------------------------------------------------------------------------
# 4. Find chapters, sections, sub-sections
# --------------------------------------------------------------------------
# optional footnote marker "1[" in front, then 16A. / 4. / ১২।  then the rest
SEC_RE = re.compile(r"^(?:[0-9" + SUPERSCRIPTS + r"]{1,3}\s*\[\s*)?([0-9০-৯]{1,3}[A-Z]{0,6})\s*[।.]\s*(\S.*)$")
CHAPTER_RE = re.compile(
    r"^(?:CHAPTER\s+(?:CHAPTER\s+)?([IVXLCDM]+[A-Z]{0,2}|\d+)|(\S{2,20})\s+অধ্যায়)\s*$")
SCHEDULE_RE = re.compile(
    r"^(?:(?:প্রথম|দ্বিতীয়|তৃতীয়|চতুর্থ|পঞ্চম|ষষ্ঠ|সপ্তম|অষ্টম|নবম|দশম|একাদশ|দ্বাদশ)\s+তফসিল"
    r"|(?:THE\s+)?[A-Z]{3,12}\s+SCHEDULE)\s*$")


def sec_key(ident: str):
    m = re.fullmatch(r"([0-9০-৯]+)([A-Z]*)", ident)
    return (int(m.group(1).translate(BN_DIGITS)), m.group(2)) if m else (-1, "")


def split_title(first: str, nxt: str = ""):
    """'Short title.—(1) This ...' -> (title, rest, used_next_line).  Only a '.—' ends a title."""
    cands = [(first, False)]
    wrapped_title = (nxt and not nxt.startswith("(") and not first.rstrip().endswith((".", "]", "।"))
                     and not SEC_RE.match(nxt) and not CHAPTER_RE.match(nxt) and not SCHEDULE_RE.match(nxt))
    if wrapped_title:                         # long title that continues on the next line
        cands.append((first + " " + nxt, True))
    for text, used in cands:
        m = re.search(r"[।.]\s*[—–]+\s*", text)
        if m and m.start() <= 200:
            return text[:m.start()].strip(), text[m.end():].strip(), used
    return first.strip().rstrip(".।").strip(), "", False


def find_body_start(lines):
    """Skip title page + contents list: the real section 1 has real text after it."""
    cands = []
    for i, (_, t, note) in enumerate(lines):
        m = SEC_RE.match(t)
        if m and not note:
            cands.append((i, sec_key(m.group(1))))
    for idx, (i, key) in enumerate(cands):
        if key != (1, ""):
            continue
        nxt = next((j for j, k in cands[idx + 1:] if k == (2, "")), None)
        end = nxt if nxt is not None else min(len(lines), i + 60)
        if sum(len(t) for _, t, _ in lines[i:end]) >= 90:
            for back in (2, 1):                      # chapter heading (+ its title) just above
                if i - back >= 0 and CHAPTER_RE.match(lines[i - back][1]):
                    return i - back
            return i
    return 0


def parse_structure(lines):
    start = find_body_start(lines)
    chapters, items = [], []
    debug = {"start_index": start, "schedule_hits": [], "rejected": []}
    cur, chapter, prev = None, {"no": "", "title": ""}, (0, "")
    markers, note_target = {}, None          # (page, n) -> section that has the marker "n["

    def close():
        nonlocal cur
        if cur:
            items.append(cur)
            cur = None

    def register_markers(text, page):
        if cur is not None:
            for n in re.findall(r"(?<![0-9])([0-9]{1,2})\s*\[", text):
                markers[(page, int(n))] = cur

    i = start
    while i < len(lines):
        page, line, note = lines[i]
        nxt = lines[i + 1][1] if i + 1 < len(lines) and not lines[i + 1][2] else ""

        if note:                                     # footnote / amendment note
            m = re.match(r"^([0-9]{1,2})\s+\S", line)
            if m:                                    # a new footnote starts: find its section
                note_target = markers.get((page, int(m.group(1))), cur)
            target = note_target or cur
            if target is not None:
                target["notes"].append(line)
            i += 1
            continue
        note_target = None

        m = CHAPTER_RE.match(line)
        if m:
            close()
            no = (m.group(1) or m.group(2) or "").strip()
            title = nxt if nxt and not SEC_RE.match(nxt) and (nxt.isupper() or nxt.startswith("[")
                                                              or re.search("[\u0980-\u09FF]", nxt)) else ""
            chapter = {"no": no, "title": title}
            chapters.append({"chapter_no": no, "chapter_title": title, "sections": []})
            i += 2 if title else 1
            continue

        m = SCHEDULE_RE.match(line)
        if m and prev[0] > 0:
            close()
            debug["schedule_hits"].append((page, line, "".join(map(str, prev))))
            cur = {"type": "schedule", "id": line, "title": line, "chapter": {"no": "", "title": ""},
                   "page_start": page, "page_end": page, "lines": [], "notes": []}
            i += 1
            continue

        m = SEC_RE.match(line)
        if m:
            key = sec_key(m.group(1))
            ok = (key == (1, "") and prev == (0, "")) or (key > prev and key[0] - prev[0] <= 12 and prev[0] > 0)
            if ok:
                close()
                title, rest, used_next = split_title(m.group(2), nxt)
                cur = {"type": "section", "id": m.group(1), "title": title, "chapter": dict(chapter),
                       "page_start": page, "page_end": page, "lines": [rest] if rest else [], "notes": []}
                if chapters:
                    chapters[-1]["sections"].append(m.group(1))
                prev = key
                register_markers(line, page)
                i += 2 if used_next else 1
                continue
            debug["rejected"].append((page, line, "".join(map(str, prev))))

        if cur:
            cur["lines"].append(line)
            cur["page_end"] = page
            register_markers(line, page)
        i += 1
    close()
    return items, chapters, debug


# ---- sub-sections ---------------------------------------------------------
SUB_START = r"(?:[0-9" + SUPERSCRIPTS + r"]{1,3}\s*\[\s*)?"
NUM_SUB = re.compile(r"(?m)^" + SUB_START + r"\(([0-9০-৯]{1,3}[A-Za-z]{0,2})\)\s*")
LET_SUB = re.compile(r"(?m)^" + SUB_START + r"\(([a-z]{1,3})\)\s*")
NEWLINE_BEFORE = re.compile(r"^" + SUB_START + r"\(([0-9০-৯]{1,3}[A-Za-z]{0,2}|[a-zA-Z]{1,4})\)")


def join_lines(lines):
    """Join PDF line-breaks into paragraphs; (1), (2), (a) start on a new line."""
    text = ""
    for l in lines:
        if not text:
            text = l
        elif NEWLINE_BEFORE.match(l):
            text += "\n" + l
        elif text.endswith("-") and re.match(r"[a-z]", l):
            text = text[:-1] + l
        else:
            text += " " + l
    return text.strip()


def letters_label(index: int) -> str:
    """0 -> A, 1 -> B, ... 25 -> Z, 26 -> AA"""
    s, n = "", index + 1
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _accept_numeric(matches):
    ok, prev = [], None
    for m in matches:
        lab = m.group(1).translate(BN_DIGITS)
        mm = re.fullmatch(r"(\d+)([A-Za-z]*)", lab)
        n, suf = int(mm.group(1)), mm.group(2)
        if prev is None or (n == prev[0] + 1 and not suf) or (n == prev[0] and suf > prev[1]) \
                or (n == prev[0] + 1 and suf):
            ok.append(m)
            prev = (n, suf)
    return ok


def _accept_letters(matches):
    ok, prev = [], None
    for m in matches:
        lab = m.group(1)
        if prev is None:
            good = lab == "a"
        else:
            base = prev[0]
            good = (len(lab) == 1 and ord(lab) == ord(base) + 1) or (len(set(lab)) == 1 and lab[0] == base and len(lab) > len(prev))
        if good:
            ok.append(m)
            prev = lab
    return ok


def split_subsections(text: str, sec_no: str, style: str = SUBSECTION_STYLE):
    """Return sub-sections labelled '4. (A)', '4. (B)' ... (or the original labels)."""
    ms = _accept_numeric(list(NUM_SUB.finditer(text)))
    if not ms:                                       # no (1),(2): use clauses (a),(b)
        ms = _accept_letters(list(LET_SUB.finditer(text)))
    if not ms:
        return []
    subs = []
    intro = text[:ms[0].start()].strip()
    if intro:
        subs.append({"label": "intro", "full_label": f"{sec_no}. (intro)", "original_label": "", "text": intro})
    for k, m in enumerate(ms):
        end = ms[k + 1].start() if k + 1 < len(ms) else len(text)
        original = f"({m.group(1)})"
        label = f"({letters_label(k)})" if style == "letters" else original
        subs.append({"label": label, "full_label": f"{sec_no}. {label}",
                     "original_label": original, "text": text[m.end():end].strip()})
    return subs


def readable_text(sec_no, title, text, subs):
    head = f"{sec_no}. {title}".strip()
    if not subs:
        return f"{head}\n{text}".strip()
    body = [s["text"] if s["label"] == "intro" else f"{s['label']} {s['text']}" for s in subs]
    return head + "\n" + "\n".join(body)


# --------------------------------------------------------------------------
# 5. Build final records
# --------------------------------------------------------------------------
def build_records(items, lang, source_url, law_name, law_code, style=SUBSECTION_STYLE):
    best = {}
    for it in items:
        text = join_lines(it["lines"])
        key = (it["type"], it["id"])
        if key in best and len(best[key]["text"]) >= len(text):
            continue
        best[key] = {**it, "text": text}

    records, sch = [], 0
    for (typ, ident), it in best.items():
        if typ == "section":
            rec_id, sec_no = f"{law_code}-S{ident}", ident
        else:
            sch += 1
            rec_id, sec_no = f"{law_code}-SCH{sch}", ""
        subs = split_subsections(it["text"], sec_no, style) if typ == "section" else []
        records.append({
            "id": rec_id,
            "type": typ,
            "law_name": law_name,
            "language": lang,
            "chapter_no": it["chapter"]["no"],
            "chapter_title": it["chapter"]["title"],
            "section_no": sec_no,
            "section_title": it["title"],
            "text": it["text"],
            "subsections": subs,
            "text_readable": readable_text(sec_no, it["title"], it["text"], subs) if typ == "section" else it["text"],
            "amendment_notes": join_lines(it["notes"]) if it["notes"] else "",
            "page_start": it["page_start"],
            "page_end": it["page_end"],
            "source_url": source_url,
            "easy_explanation": None,
        })
    records.sort(key=lambda r: (r["type"] != "section", sec_key(r["section_no"]) if r["section_no"] else (0, ""), r["id"]))
    return records


def check_gaps(records):
    ids = [r["section_no"] for r in records if r["type"] == "section"]
    if not ids:
        print("      WARNING: no sections found. Run with --debug and check cleaned_lines.txt.")
        return
    nums = sorted({sec_key(i)[0] for i in ids})
    missing = [n for n in range(nums[0], nums[-1] + 1) if n not in set(nums)]
    print(f"      sections found: {len(ids)} (numbers {nums[0]} to {nums[-1]})")
    if missing:
        print(f"      WARNING: missing section numbers: {missing[:40]}{' ...' if len(missing) > 40 else ''}")
        print("      (some may be repealed; check the others in the PDF)")


def print_debug(lines, dbg, records, out_dir, verbose):
    secs = [r for r in records if r["type"] == "section"]
    if dbg["schedule_hits"]:
        pg, ln, last = dbg["schedule_hits"][0]
        print(f"      first schedule heading: page {pg}, after section {last}: {ln!r}")
    if verbose:
        path = out_dir / "cleaned_lines.txt"
        path.write_text("\n".join(f"[p{p}{' NOTE' if n else ''}] {t}" for p, t, n in lines), encoding="utf-8")
        print(f"      saved cleaned text: {path}")
        print(f"      body starts at cleaned line #{dbg['start_index']}")
        if secs:
            last = max(secs, key=lambda r: sec_key(r["section_no"]))
            print(f"      last section: {last['section_no']} (page {last['page_start']}) {last['section_title'][:70]!r}")
        print(f"      numbered lines NOT accepted as sections: {len(dbg['rejected'])}")
        for pg, ln, last in dbg["rejected"][:15]:
            print(f"        p{pg} (last accepted={last}): {ln[:90]}")


# --------------------------------------------------------------------------
# 6. Easy-language explanation with local Ollama (optional)
# --------------------------------------------------------------------------
def explain_simple(rec, model, lang):
    language = "Bangla (বাংলা)" if lang == "bn" else "English"
    prompt = f"""You help ordinary people in Bangladesh understand tax law.
Explain this section of the {rec['law_name']} in very simple {language}.

Rules:
- Short sentences. No difficult legal words. A school student should understand.
- Keep every number, percentage, amount and date exactly as in the law.
- Use only what is written in the text. Do not add anything new.
- Return JSON only, like:
  {{"summary": "one simple sentence", "key_points": ["point 1", "point 2", "point 3"]}}

Section {rec['section_no']}: {rec['section_title']}
Text:
{rec['text'][:6000]}
"""
    payload = {"model": model, "prompt": prompt, "stream": False, "format": "json",
               "options": {"temperature": 0.2}}
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    data = json.loads(r.json()["response"])
    return {
        "summary": str(data.get("summary", "")).strip(),
        "key_points": [str(x).strip() for x in data.get("key_points", [])][:6],
        "language": lang,
        "made_by": f"AI ({model}) - please verify with the law text",
    }


def add_explanations(records, model, lang, save_fn):
    model_ok, server_ok, _ = ollama_ready(model)
    if not (server_ok and model_ok):
        print("\nERROR: --simplify needs Ollama running with the model installed.")
        print(f"  Start Ollama and run: ollama pull {model}")
        return
    todo = [r for r in records if r["type"] == "section" and len(r["text"]) > 30 and not r["easy_explanation"]
            and "[Omitted]" not in r["text"][:40]]
    print(f"[5/5] Writing easy explanations for {len(todo)} sections (this takes a while) ...")
    for k, rec in enumerate(todo, start=1):
        try:
            rec["easy_explanation"] = explain_simple(rec, model, lang)
        except Exception as e:
            print(f"      section {rec['section_no']}: failed ({e})")
            time.sleep(2)
        if k % 10 == 0:
            save_fn()
            print(f"      ... {k}/{len(todo)}")
    save_fn()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build income-tax law JSON dataset")
    ap.add_argument("--pdf-url", default=PDF_URL)
    ap.add_argument("--pdf-path", help="use a PDF you already downloaded")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--law-name", default=DEFAULT_LAW_NAME)
    ap.add_argument("--law-code", default=DEFAULT_LAW_CODE)
    ap.add_argument("--subsection-style", choices=["letters", "original"], default=SUBSECTION_STYLE)
    ap.add_argument("--start-page", type=int, default=DEFAULT_START_PAGE,
                    help="first PDF page to read (default 17 = printed page 144; contents list is skipped)")
    ap.add_argument("--end-page", type=int, default=None, help="last PDF page to read (default: end of PDF)")
    ap.add_argument("--force-ocr", action="store_true", help="OCR every page with Ollama")
    ap.add_argument("--no-ocr", action="store_true", help="never use OCR (normal PDF text only)")
    ap.add_argument("--simplify", action="store_true", help="add easy-language explanations")
    ap.add_argument("--simple-lang", choices=["bn", "en"], default="en")
    ap.add_argument("--model", default=OLLAMA_MODEL)
    ap.add_argument("--debug", action="store_true", help="save cleaned_lines.txt and print parser details")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pick_pdf(args.pdf_path, args.pdf_url, out_dir)
    json_path = out_dir / f"{args.law_code.lower()}_sections.json"

    pages = extract_pages(pdf_path, args.force_ocr, args.model, out_dir / "ocr_cache", args.no_ocr,
                          start_page=args.start_page, end_page=args.end_page)

    print("[3/5] Cleaning text ...")
    body_font = estimate_body_font(pages)
    lines = clean_pages(pages, body_font)
    lang = detect_language(lines)
    print(f"      language: {lang}, body font: {body_font}, lines kept: {len(lines)}, "
          f"footnote lines: {sum(1 for _, _, n in lines if n)}")

    print("[4/5] Finding chapters and sections ...")
    items, chapters, dbg = parse_structure(lines)
    records = build_records(items, lang, args.pdf_url, args.law_name, args.law_code, args.subsection_style)
    check_gaps(records)
    print_debug(lines, dbg, records, out_dir, args.debug)

    if json_path.exists():                           # keep old easy explanations on re-run
        try:
            old = {r["id"]: r.get("easy_explanation")
                   for r in json.loads(json_path.read_text(encoding="utf-8"))["sections"]}
            for r in records:
                r["easy_explanation"] = old.get(r["id"])
        except Exception:
            pass

    def save():
        data = {
            "metadata": {
                "law_name": args.law_name,
                "language": lang,
                "source_url": args.pdf_url,
                "created_on": date.today().isoformat(),
                "total_sections": sum(1 for r in records if r["type"] == "section"),
                "subsection_style": args.subsection_style,
                "note": "'text' is the law text (ground truth). 'text_readable' shows sub-sections as "
                        "4. (A), (B) ... 'easy_explanation' is AI-made and only helps reading.",
            },
            "chapters": chapters,
            "sections": records,
        }
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    if args.simplify:
        add_explanations(records, args.model, args.simple_lang, save)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()