"""
13_enhance_dataset.py
Adds, to the Income Tax Act 2023 dataset made by build_ita2023_en_dataset.py:

  1. Cross-reference graph  - every section/schedule a record refers to ("section 76",
     "sections 77 and 78", "sections 84 to 89", "Sixth Schedule"), and the reverse links.
     Idea: Siddig, Tugeev & Georges 2026 (arXiv 2610.13695, RAGE-KG @ ISWC 2026);
           FourCorners, ACL 2026 Industry Track (cross-reference following).
  2. Temporal / version metadata from the amendment footnotes: amending Act, Act number,
     amending section, kind (inserted / substituted / omitted), effective date; plus status
     (in_force / omitted). Section 344 repeals the 1984 Ordinance, which is recorded too.
     Idea: Cymbler, Guez & Fabre 2026, "Temporal Misgrounding in Legal RAG" (arXiv 2608.09393).
  3. Bangla text aligned by section number from 11_extract_ita2023_gazette.py
     (section 345(2) of the Act: where Bangla and English differ, the Bangla text prevails).
  4. Clean chunk context (amendment marks such as "2[" left in schedule titles).
  5. Oversized chunks split at sub-section / sentence boundaries with the context repeated,
     so no chunk is cut by the 512-token limit of bge-m3 / bge-reranker in retrieval.
     Idea: StrucChunk, Singh et al. 2026 (arXiv 2610.26313).
  6. Chunk-quality report (StrucChunk-style boundary metrics).

Input files are only read; new files are written next to them with "_enhanced" in the name.

Run:
  python 13_enhance_dataset.py
  python 13_enhance_dataset.py --bangla data/ita2023_gazette/income_tax_act_2023_gazette.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

DATA_DIR = Path("data/Knowledge_Dataset")
RECORDS = DATA_DIR / "income_tax_act_2023.json"
CHUNKS = DATA_DIR / "income_tax_act_2023_chunks.json"
BANGLA = Path("data/ita2023_gazette/income_tax_act_2023_gazette.json")
LAST_SECTION = 345
MAX_CHARS = 2000  # about 450-500 English tokens: fits a 512-token encoder/reranker

ORDINALS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth"]
MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        1,
    )
}

# "section 76", "sections 77 and 78", "sections 76, 77 and 78", "sections 84 to 89", "section 9(2)"
REF = re.compile(
    r"\bsections?\s+(\d{1,3}(?:\s*\([^)]{1,6}\))*"
    r"(?:\s*(?:,|and|or|to|-|–)\s*\d{1,3}(?:\s*\([^)]{1,6}\))*)*)",
    re.I,
)
# "... of the Finance Act, 2024", "of the repealed Ordinance", "of the Companies Act" -> another law
EXTERNAL = re.compile(
    r"^\s*(?:\([^)]{1,6}\)\s*)*(?:,\s*)?of\s+the\s+(?!this\b)"
    r"(?:(?!\bthis\b)[^.;\n]){0,90}?\b(?:Act|Ordinance|Order|Rules|Code)\b"
    r"|^\s*(?:\([^)]{1,6}\)\s*)*of\s+the\s+repealed\b"
    r"|^\s*(?:\([^)]{1,6}\)\s*)*of\s+the\s+said\s+(?:Act|Ordinance)\b",
    re.I,
)
SRO = re.compile(
    r"\bby\s+(S\.R\.O\.\s*No\.?\s*[^,]+),\s*dated\s+([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})",
    re.I,
)
SCHED = re.compile(r"\b(" + "|".join(ORDINALS) + r"|Fourt)\s+Schedule\b", re.I)
KIND = re.compile(
    r"\b(inserted|substituted|omitted|added|renumbered|re-numbered|deleted|replaced)\b",
    re.I,
)
ACT = re.compile(r"\b((?:[A-Z][a-z]+\s+){1,4}(?:Act|Ordinance),?\s+\d{4})")
ACT_NO = re.compile(r"\((?:Act|Ordinance)\s+No\.?\s*([^)]+)\)", re.I)
BY_SEC = re.compile(r"\bby\s+section\s+([\w()]+)", re.I)
EFFECT = re.compile(
    r"with\s+effect\s+from\s+(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),?\s+(\d{4})", re.I
)
MARK = re.compile(r"\d{1,2}\[|\[\*+\]|\*{3}|(?<=\w)\](?=[\s,.;:]|$)")
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
EN_TO_BN = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


# ============================================================ helpers
def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"  wrote {path}")


def numbers_in(group):
    """'77 and 78' -> [77, 78]; '84 to 89' -> 84..89; '9(2)' -> [9]."""
    group = re.sub(r"\([^)]*\)", " ", group)
    out, tokens = [], re.findall(r"\d{1,3}|to|-|–", group)
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.isdigit():
            if (
                i + 2 < len(tokens)
                and tokens[i + 1] in ("to", "-", "–")
                and tokens[i + 2].isdigit()
            ):
                a, b = int(t), int(tokens[i + 2])
                out += list(range(a, b + 1)) if 0 < b - a <= 30 else [a, b]
                i += 3
                continue
            out.append(int(t))
        i += 1
    return out


def cross_refs(text, own_section=None):
    """Section numbers of THIS Act that the text refers to (other laws are skipped)."""
    found = []
    for m in REF.finditer(text):
        if EXTERNAL.match(text[m.end() : m.end() + 90]):
            continue
        for n in numbers_in(m.group(1)):
            if (
                1 <= n <= LAST_SECTION
                and str(n) != str(own_section)
                and str(n) not in found
            ):
                found.append(str(n))
    return found


def schedule_refs(text):
    out = []
    for m in SCHED.finditer(text):
        name = (
            m.group(1)
            .capitalize()
            .replace("Fourt", "Fourth")
            .replace("Fourthh", "Fourth")
        )
        if name not in out:
            out.append(name)
    return out


def parse_footnote(note):
    """Amendment footnote -> structured event (only what the footnote itself says)."""
    t = note.get("text", "")
    ev = {"footnote": note.get("no", ""), "page": note.get("page", ""), "text": t}
    if m := KIND.search(t):
        ev["kind"] = m.group(1).lower().replace("re-numbered", "renumbered")
    if m := ACT.search(t):
        ev["amending_act"] = re.sub(r"\s+", " ", m.group(1)).replace(" ,", ",")
    if m := ACT_NO.search(t):
        ev["amending_act_no"] = m.group(1).strip()
    if m := BY_SEC.search(t):
        ev["amending_section"] = m.group(1)
    if m := SRO.search(t):  # amended by a notification, not by an Act
        ev["amending_instrument"] = re.sub(r"\s+", " ", m.group(1)).strip()
        mon, d, y = m.group(2), m.group(3), m.group(4)
        if mon.lower() in MONTHS:
            try:  # the S.R.O. date; the footnote gives no other date
                ev["instrument_date"] = date(
                    int(y), MONTHS[mon.lower()], int(d)
                ).isoformat()
            except ValueError:
                pass
    if m := EFFECT.search(t):
        d, mon, y = m.groups()
        if mon.lower() in MONTHS:
            try:
                ev["effective_from"] = date(
                    int(y), MONTHS[mon.lower()], int(d)
                ).isoformat()
            except ValueError:
                pass
    return ev


def drop_unmatched_brackets(text):
    out, depth = [], 0
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            if depth == 0:
                continue
            depth -= 1
        out.append(ch)
    return "".join(out)


def clean_context(text):
    text = drop_unmatched_brackets(MARK.sub("", text))
    text = re.sub(r"\[\s*See\s+", "[See ", text)
    text = re.sub(r"\s+([,\]])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def split_long(body, limit):
    """Split at line starts (sub-sections, clauses, table rows) first, then at sentence ends."""
    units = []
    for line in body.split("\n"):
        if len(line) <= limit:
            units.append(line)
            continue
        buf = ""
        sents = []
        for sent in re.split(r"(?<=[.;:])\s+(?=[(A-Z“\"\d])", line):
            while len(sent) > limit:  # last resort: cut at a space
                cut = sent.rfind(" ", 0, limit)
                cut = cut if cut > limit // 2 else limit
                sents.append(sent[:cut])
                sent = sent[cut:].lstrip()
            sents.append(sent)
        for sent in sents:
            if buf and len(buf) + len(sent) + 1 > limit:
                units.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            units.append(buf)
    pieces, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) + 1 > limit:
            pieces.append(cur)
            cur = u
        else:
            cur = f"{cur}\n{u}" if cur else u
    if cur:
        pieces.append(cur)
    return pieces


def bangla_index(path):
    """section number -> Bangla section from 11_extract_ita2023_gazette.py output."""
    if not path or not Path(path).exists():
        return {}
    g = load(path)
    out = {}
    for s in g.get("sections", []):
        no = str(s.get("section_no", "")).translate(BN_DIGITS).strip()
        if no:
            out[no] = s
    return out


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(RECORDS))
    ap.add_argument("--chunks", default=str(CHUNKS))
    ap.add_argument(
        "--bangla",
        default=str(BANGLA),
        help="JSON from 11_extract_ita2023_gazette.py (skipped if the file is missing)",
    )
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument(
        "--bangla-chunks",
        action="store_true",
        help="also add one Bangla chunk per section (bilingual index)",
    )
    args = ap.parse_args()

    records = load(args.records)
    chunk_file = load(args.chunks)
    chunks = chunk_file["chunks"] if isinstance(chunk_file, dict) else chunk_file
    bn = bangla_index(args.bangla)
    print(f"{len(records)} records, {len(chunks)} chunks, {len(bn)} Bangla sections")

    # ---------------------------------------------------------------- 1-3 records
    by_id = {r["id"]: r for r in records}
    referenced_by = defaultdict(set)
    sched_referenced_by = defaultdict(set)
    for r in records:
        own = r.get("section") if r.get("record_type") == "section" else None
        r["cross_refs"] = cross_refs(r.get("text", ""), own)
        r["schedule_refs"] = schedule_refs(r.get("text", ""))
        for n in r["cross_refs"]:
            referenced_by[n].add(r.get("reference") or r["id"])
        for s in r["schedule_refs"]:
            sched_referenced_by[s].add(r.get("reference") or r["id"])

        events = [parse_footnote(f) for f in r.get("footnotes", [])]
        r["amendments"] = events
        dates = sorted(
            e.get("effective_from") or e.get("instrument_date")
            for e in events
            if e.get("effective_from") or e.get("instrument_date")
        )
        r["last_amended_on"] = dates[-1] if dates else ""
        r["status"] = "omitted" if r.get("is_omitted") else "in_force"
        r["supersedes"] = (
            "Income-tax Ordinance, 1984 (Ordinance No. XXXVI of 1984), repealed by "
            "section 344 of this Act"
        )

        if r.get("record_type") == "section" and str(r.get("section")) in bn:
            b = bn[str(r["section"])]
            r["title_bn"] = b.get("heading", "")
            r["text_bn"] = b.get("text", "")
            r["bn_section_number_inferred"] = bool(b.get("number_inferred"))
        else:
            r.setdefault("title_bn", "")
            r.setdefault("text_bn", "")
    for r in records:
        if r.get("record_type") == "section":
            r["referenced_by"] = sorted(
                referenced_by.get(str(r["section"]), set()), key=lambda x: (len(x), x)
            )
        else:
            r["referenced_by"] = sorted(
                sched_referenced_by.get(r.get("schedule", ""), set()),
                key=lambda x: (len(x), x),
            )

    # ---------------------------------------------------------------- 4-5 chunks
    rec_of_chunk = {}
    for c in chunks:
        base = c["chunk_id"].split("__")[0]
        rec = by_id.get(base)
        if rec is None:  # sub-section chunk ids: ita2023_s2_c4 ...
            m = re.match(r"(ita2023_s\d+)", c["chunk_id"])
            rec = by_id.get(m.group(1)) if m else None
        if rec is None and c.get("section"):
            rec = by_id.get(f"ita2023_s{c['section']}")
        rec_of_chunk[c["chunk_id"]] = rec

    out_chunks, n_split, n_ctx = [], 0, 0
    for c in chunks:
        rec = rec_of_chunk.get(c["chunk_id"]) or {}
        ctx = clean_context(c["context"])
        n_ctx += ctx != c["context"]
        own = c.get("section") if c.get("record_type") == "section" else None
        base = {
            k: v
            for k, v in c.items()
            if k not in ("context", "text", "context_and_text", "n_chars")
        }
        base.update(
            {
                "parent_id": rec.get("id", ""),
                "language": "en",
                "status": rec.get("status", "in_force"),
                "last_amended_on": rec.get("last_amended_on", ""),
                "schedule_refs": schedule_refs(c["text"]),
            }
        )
        pieces = (
            split_long(c["text"], max(400, args.max_chars - len(ctx) - 20))
            if len(c["text"]) + len(ctx) + 1 > args.max_chars
            else [c["text"]]
        )
        n_split += len(pieces) > 1
        for k, body in enumerate(pieces, 1):
            ch = dict(base)
            if len(pieces) > 1:
                ch["chunk_id"] = f"{c['chunk_id']}_part{k}"
                ch["part_of_chunk"] = c["chunk_id"]
                ch["part_no"], ch["part_total"] = k, len(pieces)
            ch["cross_refs"] = cross_refs(body, own)
            ch["context"] = ctx + (
                f" (part {k} of {len(pieces)})" if len(pieces) > 1 else ""
            )
            ch["text"] = body
            ch["context_and_text"] = f"{ch['context']}\n{body}"
            ch["n_chars"] = len(ch["context_and_text"])
            out_chunks.append(ch)

    n_bn = 0
    if args.bangla_chunks:
        for r in records:
            if not r.get("text_bn"):
                continue
            no_bn = str(r["section"]).translate(EN_TO_BN)
            ctx = f"আয়কর আইন, ২০২৩ > ধারা {no_bn} > {r.get('title_bn', '')}".strip()
            for k, body in enumerate(split_long(r["text_bn"], args.max_chars // 2), 1):
                n_bn += 1
                out_chunks.append(
                    {
                        "chunk_id": f"{r['id']}_bn{k}",
                        "record_type": "section",
                        "reference": r["reference"],
                        "section": r["section"],
                        "title": r.get("title_bn", ""),
                        "parent_id": r["id"],
                        "language": "bn",
                        "status": r["status"],
                        "last_amended_on": r["last_amended_on"],
                        "source": "আয়কর আইন, ২০২৩ (২০২৩ সনের ১২ নং আইন), বাংলাদেশ গেজেট",
                        "cross_refs": r["cross_refs"],
                        "schedule_refs": r["schedule_refs"],
                        "context": ctx,
                        "text": body,
                        "context_and_text": f"{ctx}\n{body}",
                        "n_chars": len(ctx) + 1 + len(body),
                    }
                )

    # ---------------------------------------------------------------- write
    rp, cp = Path(args.records), Path(args.chunks)
    rec_out = rp.with_name(rp.stem + "_enhanced.json")
    chk_out = cp.with_name(cp.stem.replace("_chunks", "") + "_enhanced_chunks.json")
    save(rec_out, records)
    meta = dict(chunk_file) if isinstance(chunk_file, dict) else {}
    meta.pop("chunks", None)
    meta.update(
        {
            "enhanced_by": "13_enhance_dataset.py",
            "max_chars": args.max_chars,
            "chunk_count": len(out_chunks),
            "chunks": out_chunks,
        }
    )
    save(chk_out, meta)

    # ---------------------------------------------------------------- report
    sec = [r for r in records if r.get("record_type") == "section"]
    edges = sum(len(r["cross_refs"]) for r in records)
    lens = [c["n_chars"] for c in out_chunks if c["language"] == "en"]
    cross_boundary = sum(
        1
        for c in out_chunks
        if c.get("section")
        and re.search(rf"^\s*{int(c['section']) + 1}\.\s", c["text"], re.M)
    )
    kinds = Counter(e.get("kind", "?") for r in records for e in r["amendments"])
    eff = Counter(
        e.get("effective_from")
        or ("S.R.O. " + e["instrument_date"] if e.get("instrument_date") else "none")
        for r in records
        for e in r["amendments"]
    )
    top_in = sorted(sec, key=lambda r: -len(r["referenced_by"]))[:10]
    lines = [
        f"Records: {len(records)}  (sections {len(sec)})",
        f"Cross-reference edges (this Act): {edges}; records with at least one: "
        f"{sum(1 for r in records if r['cross_refs'])}",
        "Most referenced sections: "
        + ", ".join(f"{r['section']} ({len(r['referenced_by'])})" for r in top_in),
        f"Schedule references: {sum(len(r['schedule_refs']) for r in records)}",
        f"Amendment events: {sum(len(r['amendments']) for r in records)}  kinds {dict(kinds)}",
        f"Effective dates: {dict(eff)}",
        f"Bangla text aligned: {sum(1 for r in sec if r['text_bn'])}/{len(sec)} sections "
        f"(number inferred by OCR repair: {sum(1 for r in sec if r.get('bn_section_number_inferred'))})",
        f"Chunks in: {len(chunks)}  out: {len(out_chunks)}  (English {len(lens)}, Bangla {n_bn})",
        f"Chunks split because longer than {args.max_chars} chars: {n_split}",
        f"Contexts cleaned of amendment marks: {n_ctx}",
        f"English chunk length: median {sorted(lens)[len(lens) // 2]}, max {max(lens)}",
        f"Chunks that contain the start of the next section (boundary errors): {cross_boundary}",
    ]
    rep = rec_out.with_name(rp.stem + "_enhanced_report.txt")
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"  wrote {rep}")


if __name__ == "__main__":
    main()
