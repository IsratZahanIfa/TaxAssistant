"""
build_integrated_corpus.py
Builds the preprocessing variants for the 3 experiments (same Act JSON, same ground truth):

  A  baseline    : one chunk per RAW record, raw text and raw labels (what you have today)
  B  integrated  : cleaned corpus, correct section ids, section-aware chunks + identity header
  C  hierarchical: same corpus, one child chunk per subsection + identity header, parent = whole section

What it repairs while building B/C (all seen in your retrieval files):
  - gazette header / ==== / page numbers / amendment footnotes removed (footnotes kept in "notes")
  - title/text split fixed ("Submission of Return.—(1) Every person ..." -> title + text)
  - body sections wrongly labelled sch_FIFTH_166 get their real number (labels are decided by order)
  - record that swallowed the next section ("154" containing "155. Amount of advance tax ...") is split
  - numbered list items of a section ("1. applying for a loan ...", sections 1/7/13/29/32 on pages 189-191)
    are merged back into their section instead of becoming fake sections 1, 7, 13 ...
  - schedule / rule paragraphs (numbering restarts after the last section) get ids that cannot collide
    with body sections

Run (PowerShell, from E:\\TaxAssist):
    python scripts\\build_integrated_corpus.py
Outputs go to data\\processed\\integrated_corpus\\ : chunks_A.json, chunks_B.json, chunks_C.json, parents.json
"""
import collections
import json
import re
from pathlib import Path

ROOT = Path(r"E:\TaxAssist")
CANDIDATE_INPUTS = [                                   # first one that exists is used
    ROOT / "data/legal/income_tax_act_2023/income_tax_act_2023.json",
    ROOT / "data/Knowledge_Dataset/income_tax_act_2023.json",
]
OUT_DIR = ROOT / "data/processed/integrated_corpus"
LAW_KEY, LAW_NAME = "ita_2023", "Income Tax Act, 2023"
MAX_CHARS_B = 1800        # section-aware chunk size (variant B)
MAX_CHARS_C = 1200        # max child size (variant C)
MIN_CHILD = 120           # tiny subsections are merged into the previous child
SCHEDULE_RESTART_AFTER = 200   # numbering that restarts (<=3) after section 200+ = schedules / rules

# ------------------------------------------------------------------ cleaning helpers
GAZETTE = re.compile(r"বাংলাদেশ গেজেট,[^\n]*?[০-৯]{4}")
SEP = re.compile(r"^=+\s*$", re.M)
PAGENO = re.compile(r"^\s*\d{4,6}\s*$", re.M)
FOOT_START = re.compile(
    r"^\d{1,2}\s+(The|Clause|Sub-?section|Section|Chapter|Part|Schedule|Paragraph|Table|Explanation|Heading|Rules?|"
    r"Omitted|Substituted|Inserted|Proviso|Words?)\b")
MARKER = re.compile(r"^(PART|CHAPTER)\s+([0-9IVXLC]+)[ \t]*\n(?:([A-Z][A-Z ,\-–—'’()&/]{3,})[ \t]*\n?)?", re.M)
HEADING = re.compile(
    r"^(\d{1,3}[A-Za-z]{0,2})\.[ \t]+([A-Z0-9\[][^\n]{2,160}?(?:\n[^\n]{1,100}?)?)\.[ \t]?[—–-]", re.M)


def split_footnotes(text):
    body, notes, cur = [], [], None
    for line in text.split("\n"):
        s = line.strip()
        if cur is None and FOOT_START.match(s):
            cur = [s]
        elif cur is not None:
            cur.append(s)
        else:
            body.append(line)
            continue
        if cur[-1].endswith("."):
            notes.append(" ".join(cur))
            cur = None
        elif len(cur) > 4:
            body.extend(cur)
            cur = None
    if cur:
        notes.append(" ".join(cur))
    return "\n".join(body), notes


def first_alpha_lower(s):
    for ch in s:
        if ch.isalpha():
            return ch.islower()
    return False


def parse_section(sec):
    m = re.match(r"sch_([A-Za-z]+)_(.+)$", sec or "")
    tail = m.group(2) if m else re.sub(r"^sec_", "", sec or "")
    n = re.search(r"\d+[A-Za-z]*", tail)
    return n.group(0) if n else tail


def num_parts(no):
    m = re.match(r"(\d+)([A-Za-z]*)", str(no))
    return (int(m.group(1)), m.group(2)) if m else (None, "")


def prepare(rec):
    """Cleaned (title, text, notes) of one raw record. Text still has its original line breaks."""
    text = rec.get("text", "")
    for pat in (GAZETTE, SEP, PAGENO):
        text = pat.sub("", text)
    text, notes = split_footnotes(text)
    text = MARKER.sub(lambda m: f"§{m.group(1)} {m.group(2)}|{m.group(3) or ''}§\n", text)
    title = (rec.get("title") or "").strip()
    head, sep, rest = title.partition(".—")
    if sep:
        title = head.strip()
        text = rest.strip() + ("" if rest.rstrip().endswith("-") else " ") + text.lstrip()
    else:                                   # title was cut at ~70 chars: the end of it is at the start of the text
        m = re.search(r"\.[ \t]?[—–-]", text[:220])
        if m:
            tail = text[:m.start()].strip()
            title = (title + ("" if title.endswith("-") else " ") + tail).strip()
            text = text[m.end():].lstrip()
    return title, text, notes


def tidy(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1-\2", text)
    text = re.sub(r"\n(?!\s*\([A-Za-z0-9]{1,5}\))", " ", text)       # join wrapped lines, keep (a) (1) starts
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def take_markers(seg):
    """Remove §PART 18|TITLE§ sentinels. Returns (text, (chapter, part) update, applies_to_this_segment)."""
    upd, here = {}, False
    for m in re.finditer(r"§(PART|CHAPTER) ([0-9IVXLC]+)\|[^§]*§\n?", seg):
        upd["part" if m.group(1) == "PART" else "chapter"] = f"{m.group(1)} {m.group(2)}"
        here = here or m.start() < 60
    return re.sub(r"§(?:PART|CHAPTER) [0-9IVXLC]+\|[^§]*§\n?", "", seg), upd, here


def split_bleed(text, cur_int):
    """Cut a record at headings of the NEXT sections it swallowed. Returns [(no|None, title|None, text)]."""
    cuts, last = [], cur_int
    for m in HEADING.finditer(text):
        n_int, _ = num_parts(m.group(1))
        if n_int is not None and last < n_int <= last + 5:
            cuts.append((m.start(), m.end(), m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()))
            last = n_int
    if not cuts:
        return [(None, None, text)]
    out = [(None, None, text[:cuts[0][0]])]
    for i, (s, e, no, title) in enumerate(cuts):
        nxt = cuts[i + 1][0] if i + 1 < len(cuts) else len(text)
        out.append((no, title, text[e:nxt].lstrip()))
    return out


# ------------------------------------------------------------------ corpus (parents)
def build_parents(raw):
    parents, st = [], collections.Counter()
    mode, prev_int, prev_suf = "body", 0, ""
    ctx = {"chapter": None, "part": None}
    last_pg = 0

    def add(no, title, text, notes, rec, ptype, split_from=None):
        pid = f"{LAW_KEY}_" + (f"sec_{no}" if ptype == "section" else f"sch_{rec.get('page')}_{no}")
        parents.append(dict(
            id=pid, law_key=LAW_KEY, law_name=LAW_NAME, record_type=ptype, section_no=str(no), title=title,
            chapter=ctx["chapter"] or rec.get("chapter"), part=ctx["part"] or rec.get("part"),
            page_start=rec.get("page") or 0, page_end=rec.get("page") or 0, text=text, notes=list(notes),
            source_ids=[rec.get("id")], split_from=split_from))

    for rec in raw:
        label = parse_section(rec.get("section", ""))
        n_int, n_suf = num_parts(label)
        page = rec.get("page") or 0
        if page < last_pg:
            st["page_order_warnings"] += 1
        last_pg = max(last_pg, page)
        title0 = (rec.get("title") or "").strip()
        lower = first_alpha_lower(title0)

        # 1) list item / continuation of the previous parent
        is_cont = bool(parents) and (
            lower or (mode == "body" and n_int is not None and (n_int < prev_int or (n_int == prev_int and n_suf == prev_suf))))
        has_title_mark = bool(re.search(r"\.[ \t]?[—–-]", title0 + " " + rec.get("text", "")[:220]))
        restart = (mode == "body" and n_int is not None and n_int <= 3 and prev_int >= SCHEDULE_RESTART_AFTER
                   and not lower and has_title_mark)
        if restart:
            mode = "schedule"
            is_cont = False
            st["schedule_start_page"] = page
        if is_cont and not restart:
            extra = "\n".join(x for x in (title0, rec.get("text", "")) if x)
            for pat in (GAZETTE, SEP, PAGENO):
                extra = pat.sub("", extra)
            extra, notes = split_footnotes(extra)
            parents[-1]["text"] += "\n" + extra
            parents[-1]["notes"] += notes
            parents[-1]["page_end"] = max(parents[-1]["page_end"], page)
            parents[-1]["source_ids"].append(rec.get("id"))
            st["merged_list_items_or_continuations"] += 1
            continue

        # 2) a real new section / schedule paragraph
        title, text, notes = prepare(rec)
        segs = split_bleed(text, n_int or 0) if mode == "body" else [(None, None, text)]
        cur_no = label
        for j, (no, t, seg) in enumerate(segs):
            if j:
                cur_no, title = no, t
                st["split_from_swallowed_record"] += 1
            seg, upd, here = take_markers(seg)
            if here:
                ctx.update(upd)
            ptype = "section" if mode == "body" else "schedule_paragraph"
            add(cur_no, title, seg, notes if j == 0 else [], rec, ptype, split_from=None if j == 0 else rec.get("id"))
            if not here:
                ctx.update(upd)
            if mode == "body":
                prev_int, prev_suf = num_parts(cur_no)[0] or prev_int, num_parts(cur_no)[1]
            else:
                prev_int = num_parts(cur_no)[0] or prev_int
    for p in parents:
        p["text"] = tidy(p["text"])
        p["title"] = re.sub(r"\s+", " ", p["title"]).strip(" .—")
    # unique ids
    seen = collections.Counter()
    for p in parents:
        seen[p["id"]] += 1
        if seen[p["id"]] > 1:
            p["id"] += f"_{seen[p['id']]}"
    return parents, st


# ------------------------------------------------------------------ chunking
def section_key(p):
    return f"{LAW_KEY}:{p['section_no']}" if p["record_type"] == "section" else \
        f"{LAW_KEY}:sch_{p['page_start']}_{p['section_no']}"


def header(p, sub=None):
    lines = [f"Source: {p['law_name']}"]
    if p.get("chapter"):
        lines.append(f"Chapter: {p['chapter']}")
    if p.get("part"):
        lines.append(f"Part: {p['part']}")
    lines.append(("Section: " if p["record_type"] == "section" else "Schedule paragraph: ") + p["section_no"])
    lines.append(f"Title: {p['title']}")
    if sub:
        lines.append(f"Subsection: {sub}")
    return "\n".join(lines)


def subsections(text):
    idx = [m.start() for m in re.finditer(r"(?m)^(?=\(\d{1,3}[A-Z]?\)\s)", text)]
    if not idx:
        return [("", text)]
    blocks = []
    pre = text[:idx[0]].strip()
    for a, b in zip(idx, idx[1:] + [len(text)]):
        blk = text[a:b].strip()
        blocks.append([re.match(r"\(\d{1,3}[A-Z]?\)", blk).group(0), blk])
    if pre:
        blocks[0][1] = pre + "\n" + blocks[0][1]            # keep the lead-in ("For the purposes of ...")
    return [tuple(b) for b in blocks]


def pieces(block, limit):
    if len(block) <= limit:
        return [block]
    out, cur = [], ""
    for line in block.split("\n"):
        while len(line) > limit:                            # very long single line: cut at a sentence end
            cut = max(line.rfind(". ", 0, limit), limit // 2)
            out.append((cur + "\n" + line[:cut + 1]).strip()) if cur else out.append(line[:cut + 1])
            cur, line = "", line[cut + 1:].strip()
        if cur and len(cur) + len(line) > limit:
            out.append(cur)
            cur = ""
        cur = (cur + "\n" + line).strip()
    if cur:
        out.append(cur)
    return out


def chunk(p, cid, parent_id, text, sub, variant):
    return dict(id=cid, parent_id=parent_id, variant=variant, key=section_key(p), law_key=LAW_KEY,
                section_no=p["section_no"], record_type=p["record_type"], title=p["title"], chapter=p["chapter"],
                part=p["part"], subsection=sub, page_start=p["page_start"], page_end=p["page_end"],
                language="en", effective_from=None, text=text, embed_text=f"{header(p, sub)}\n\n{text}")


def chunks_B(p):
    out, cur, size = [], [], 0
    for sub, blk in subsections(p["text"]):
        for piece in pieces(blk, MAX_CHARS_B):
            if cur and size + len(piece) > MAX_CHARS_B:
                out.append("\n".join(cur))
                cur, size = [], 0
            cur.append(piece)
            size += len(piece)
    if cur:
        out.append("\n".join(cur))
    return [chunk(p, f"{p['id']}_B{i}", p["id"], t, None, "B") for i, t in enumerate(out)]


def chunks_C(p):
    kids = []
    for sub, blk in subsections(p["text"]):
        if kids and len(blk) < MIN_CHILD:
            kids[-1][1] += "\n" + blk
        else:
            kids.append([sub, blk])
    out = []
    for sub, blk in kids:
        for k, piece in enumerate(pieces(blk, MAX_CHARS_C)):
            out.append((sub + (f" part {k + 1}" if k else ""), piece))
    return [chunk(p, f"{p['id']}_C{i}", p["id"], t, s or None, "C") for i, (s, t) in enumerate(out)]


def chunks_A(rec, i):
    pid = rec.get("id") or f"raw_{i}"
    title, text = (rec.get("title") or ""), rec.get("text", "")
    return dict(id=f"{pid}_A", parent_id=pid, variant="A", key=f"{LAW_KEY}:{rec.get('section')}",
                law_key=LAW_KEY, section_no=str(rec.get("section")), record_type=rec.get("record_type"),
                title=title, chapter=rec.get("chapter"), part=rec.get("part"), subsection=None,
                page_start=rec.get("page"), page_end=rec.get("page"), language="en", effective_from=None,
                text=text, embed_text=f"{title}\n{text}")


# ------------------------------------------------------------------ main
def main():
    src = next((p for p in CANDIDATE_INPUTS if p.exists()), None)
    if src is None:
        raise SystemExit("Act JSON not found. Edit CANDIDATE_INPUTS at the top of this file.")
    raw = json.load(open(src, encoding="utf-8"))
    if isinstance(raw, dict):
        raw = next(v for v in raw.values() if isinstance(v, list))
    print(f"input: {src}  ({len(raw)} raw records)")
    print("raw record_type counts:", dict(collections.Counter(r.get("record_type") for r in raw)))

    parents, st = build_parents(raw)
    body = [p for p in parents if p["record_type"] == "section"]
    sched = [p for p in parents if p["record_type"] != "section"]
    nums = sorted({num_parts(p["section_no"])[0] for p in body if num_parts(p["section_no"])[0] is not None})
    missing = [n for n in range(1, (nums[-1] if nums else 0) + 1) if n not in nums]
    print(f"parents: {len(body)} body sections, {len(sched)} schedule/rule paragraphs")
    print("repairs:", dict(st))
    print(f"highest section number: {nums[-1] if nums else None}; numbers with no record "
          f"({len(missing)}): {missing[:60]}{' ...' if len(missing) > 60 else ''}")

    A = [chunks_A(r, i) for i, r in enumerate(raw)]
    B = [c for p in parents for c in chunks_B(p)]
    C = [c for p in parents for c in chunks_C(p)]
    print(f"chunks: A={len(A)}  B={len(B)}  C={len(C)}")
    noisy = sum(1 for c in B if "গেজেট" in c["text"] or "====" in c["text"])
    print("B chunks still containing gazette/separator noise:", noisy)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in (("chunks_A", A), ("chunks_B", B), ("chunks_C", C)):
        json.dump(data, open(OUT_DIR / f"{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({p["id"]: p for p in parents}, open(OUT_DIR / "parents.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("saved ->", OUT_DIR)


if __name__ == "__main__":
    main()