"""
evaluate_retrieval4.py - layered legal RAG retrieval for the Income Tax Act 2023 + evaluation.

Result of v3 on the 140 questions: Hit@5 47.79 %, Hit@1 14.71 %, pool recall 83.09 %,
23 retriever failures, 48 reranker failures. Problems found, the layer that fixes each, and
the 2025-2026 work it comes from:

 P1  Ground-truth parser read "Section 45–54" as section 45 only and missed "Schedule 6",
     "6th Schedule" (18 + 7 questions); schedule records were stored as "Sixth Schedule" but
     compared with "Sixth", so no schedule could ever be a hit, be boosted or be reached by a
     cross-reference.                                           -> L0 parser + schedule fix
 P2  70 Bangla questions are searched against an English-only index (0 Bangla chunks), and
     lay words (TIN, AIT, FDR, e-return, freelancing) do not occur in the statute
     (vocabulary gap).                                          -> L1 query translation + legal
     rewrite + glossary   [Decompose-and-Refine, Lee et al. 2026, arXiv 2605.24454;
     GuRE, Kim et al. NLLP 2025; translation-centric Bengali RAG, Hossain et al. 2026,
     arXiv 2601.02065]
 P3  The right section is in the pool but ranked low: chunks alone carry little of
     "what this section is about".                              -> L2 section-card index
     (title + Part/Chapter path + opening text, one per section)
     [Chunking German Legal Code 2026, arXiv 2605.19806: section-level units;
      StrucChunk 2026, arXiv 2610.26313: hierarchical context]
 P4  Questions are asked by lay people, the corpus is statute text.
                                                                -> L3 synthetic-question
     index (doc2query: questions generated from every section, made once and cached)
     [Synthetic legal queries: Pham et al. 2024 (Vietnamese law); Stammbach et al. 2026,
      arXiv 2601.14348 (curated synthetic data and query expansion help)]
 P5  Reranker failures.                                         -> L4 reranker reads the
     English legal rewrite + original question; reranker/fusion weight is a parameter
 kept from v3: hybrid BM25 + BGE-M3, HyDE, RRF, cross-reference expansion, query routing,
 retriever-vs-reranker failure split, bootstrap CI, ablation.

All LLM outputs (rewrite, HyDE, doc2query) are cached in data/*.json, so the second run
needs no LLM. The LLM never sees the evaluation answers or ground truth.

Usage (from the project folder):
  python scripts\\evaluate_retrieval4.py --questions data\\QA_Dataset\\Income_Tax_audited.xlsx --sheet Corrected_GT
  python scripts\\evaluate_retrieval4.py --questions ... --ablation
  python scripts\\evaluate_retrieval4.py --no-d2q          # skip the synthetic-question layer
"""

import argparse
import gc
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ============================================================ CONFIG
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"
RECORDS_PATH = Path("data/Knowledge_Dataset/income_tax_act_2023_enhanced.json")
CHUNKS_PATH = Path("data/Knowledge_Dataset/income_tax_act_2023_enhanced_chunks.json")
GOOGLE_SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/17ew_umfBKtAeTUEBFtBknNoyQR5KbWRSgceh47DNeYY"
    "/export?format=csv&gid=0"
)
HYDE_CACHE = Path("data/hyde_cache.json")
REWRITE_CACHE = Path("data/rewrite_cache.json")
D2Q_CACHE = Path("data/doc2query_cache.json")
OUT_XLSX = "retrieval_v4_results.xlsx"

TOP_CHUNKS_PER_LIST = 200
RRF_K = 60
POOL = 30
XREF_FROM, XREF_MAX, XREF_PER_SECTION = 10, 15, 3
CHUNKS_PER_SECTION = 3
TOP_K_FINAL = 10
RERANK_WEIGHT = 0.6  # v3 used 0.8; most v3 failures were in the reranker
MAX_LEN_EMBED = 512
MAX_LEN_RERANK = 512
BATCH_EMBED = 16
BATCH_RERANK = 8
BOOTSTRAP = 10000
D2Q_PER_DOC = 6
CARD_CHARS = 600

BN_TO_EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
BANGLA = re.compile(r"[ঀ-৿]")
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
BN_ORDINALS = [
    "প্রথম",
    "দ্বিতীয়",
    "তৃতীয়",
    "চতুর্থ",
    "পঞ্চম",
    "ষষ্ঠ",
    "সপ্তম",
    "অষ্টম",
]
NUM_SUFFIX = {
    "1st": 1,
    "2nd": 2,
    "3rd": 3,
    "4th": 4,
    "5th": 5,
    "6th": 6,
    "7th": 7,
    "8th": 8,
}

# lay word -> words used in the Act (general tax vocabulary, written without looking at the test set)
GLOSSARY = {
    r"\btin\b|e-?tin|টিআইএন|টিন": "taxpayer's identification number registration of assessee",
    r"\bait\b|advance income tax|অগ্রিম": "advance tax payment instalments",
    r"\btds\b|withholding|উৎসে কর": "deduction of tax at source",
    r"\bfdr\b|fixed deposit|এফডিআর": "fixed deposit savings deposit interest income from financial assets",
    r"\bdps\b|ডিপিএস": "deposit pension scheme investment tax rebate",
    r"e-?return|online|ই-রিটার্ন|অনলাইন": "electronic tax management electronic return",
    r"freelanc|youtube|facebook|e-?commerce|ফ্রিল্যান্স|ইউটিউব": "income from business profession",
    r"\bhra\b|house rent allowance|বাড়ি ভাড়া ভাতা": "income from employment perquisite allowance",
    r"rent(?:al)?\b|ভাড়া": "income from rent gross rental value",
    r"agricultur|কৃষি": "income from agriculture",
    r"capital gain|মূলধনী": "capital gains transfer of capital asset",
    r"refund|ফেরত": "refund of excess tax right of refund claim of refund",
    r"late|delay|দেরি|বিলম্ব": "after the tax day failure to file return",
    r"appeal|আপিল": "appeal to appellate income-tax authority appellate tribunal",
    r"rebate|রেয়াত": "tax rebate in respect of certain investment",
    r"exempt|অব্যাহতি|করমুক্ত": "tax exemptions exclusion from total income sixth schedule",
    r"resident|অনিবাসী|আবাসিক": "resident non-resident scope of total income",
    r"wealth statement|it-?10b|সম্পদ ও দায়": "statement of assets and liabilities",
    r"dividend|লভ্যাংশ": "dividend deduction of tax from dividend",
    r"slab|tax rate|করের হার": "rate of tax",
}


# ============================================================ text helpers
def norm(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return str(text).lower().translate(BN_TO_EN)


def section_id(value):
    m = re.search(r"\d{1,3}[a-z]{0,2}", norm(value))
    return m.group(0) if m else ""


SECTION_REF = re.compile(
    r"(?:\bsections?\b|\bsec\.?|\bs\.|\bu/s\.?|ধারা)\s*((?:\d{1,3}[a-z]{0,2}(?:\([^)]{1,6}\))*"
    r"(?:\s*(?:,|and|or|&|ও|এবং|to|–|—|-)\s*(?:sec(?:tion)?s?\.?\s*)?)?)+)",
    re.I,
)


def sections_named(text):
    """'Section 45–54' -> 45..54, 'Section 62 & Sec 102' -> {62, 102}, 'ধারা ৭৬' -> {76}."""
    found = set()
    t = re.sub(r"\b\d(?:st|nd|rd|th)\s+schedule|\bschedule\s+\d\b", " ", norm(text))
    for m in SECTION_REF.finditer(t):
        g = re.sub(r"\([^)]*\)", "", m.group(1))
        tokens = re.findall(r"\d{1,3}[a-z]{0,2}|to|–|—|-", g)
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t[0].isdigit():
                if (
                    i + 2 < len(tokens)
                    and tokens[i + 1] in ("to", "–", "—", "-")
                    and tokens[i + 2].isdigit()
                ):
                    a, b = int(re.sub(r"\D", "", t)), int(tokens[i + 2])
                    if 0 < b - a <= 60:
                        found.update(str(n) for n in range(a, b + 1))
                        i += 3
                        continue
                found.add(t)
            i += 1
    return found


def schedules_named(text):
    t = norm(text)
    found = {
        m.group(1).capitalize()
        for m in re.finditer(r"\b(" + "|".join(ORDINALS) + r")\s+schedule\b", t)
    }
    for m in re.finditer(r"\b(\d)(?:st|nd|rd|th)\s+schedule\b|\bschedule\s+(\d)\b", t):
        n = int(m.group(1) or m.group(2))
        if 1 <= n <= 8:
            found.add(ORDINALS[n - 1].capitalize())
    for i, w in enumerate(BN_ORDINALS):
        if w + " তফসিল" in str(text):
            found.add(ORDINALS[i].capitalize())
    for m in re.finditer(r"([০-৯\d])(?:ম|য়|র্থ|ষ্ঠ)?\s*তফসিল", str(text)):
        n = int(m.group(1).translate(BN_TO_EN))
        if 1 <= n <= 8:
            found.add(ORDINALS[n - 1].capitalize())
    return found


def tokenize(text):
    return re.findall(r"[a-z0-9]+|[ঀ-৿]+", norm(text))


def glossary_terms(query):
    q = norm(query)
    return " ".join(v for k, v in GLOSSARY.items() if re.search(k, q, re.I))


ROUTES = [
    (
        "definition",
        r"\bwhat (?:is|are|does)\b.*\b(?:mean|meaning|defin)|\bdefin|\bmeaning of\b|সংজ্ঞা|বলতে কী|বলিতে",
    ),
    (
        "penalty",
        r"\bpenalt|\bfine\b|\bpunish|\boffen[cs]e|\bimprison|জরিমানা|দণ্ড|শাস্তি",
    ),
    (
        "rate_amount",
        r"\brate\b|\bpercent|%|\bhow much\b|\bthreshold\b|\blimit\b|\bslab\b|হার|শতাংশ|কত টাকা|সীমা",
    ),
    (
        "deadline_procedure",
        r"\bwhen\b|\bdeadline\b|\bdue date\b|\bwithin\b|\bprocedure\b|\bhow (?:to|can|do|should)\b"
        r"|\bfile\b|\bsubmit\b|সময়|মেয়াদ|দাখিল|পদ্ধতি|কীভাবে",
    ),
    (
        "multi_condition",
        r"\bif\b.*\b(?:and|but|while)\b|\bboth\b|\bcompare|\bdifference\b|যদি.*(?:এবং|কিন্তু)",
    ),
]
ROUTE_POOL = {
    "definition": 30,
    "rate_amount": 40,
    "deadline_procedure": 30,
    "penalty": 35,
    "multi_condition": 45,
    "general": 30,
}


def route(query):
    q = norm(query)
    for name, pat in ROUTES:
        if re.search(pat, q, re.I):
            return name
    return "general"


# ============================================================ LLM layers (cached)
def call_ollama(prompt, num_predict=200):
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": num_predict},
            },
            timeout=180,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        print(f"Ollama error {r.status_code}")
    except Exception as e:  # noqa: BLE001
        print(f"Ollama request failed: {e}")
    return ""


def load_cache(path):
    try:
        return (
            json.loads(Path(path).read_text(encoding="utf-8"))
            if Path(path).exists()
            else {}
        )
    except json.JSONDecodeError:
        return {}


def save_cache(path, cache):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def strip_numbers(text):
    """LLM output must not decide the section: remove section / schedule numbers and disclaimers."""
    text = re.split(r"\*{0,2}(?:Disclaimer|Note:)", text, flags=re.I)[0]
    text = re.sub(r"(?m)^#+\s.*$", "", text)
    text = re.sub(
        r"\b(?:[Ss]ections?|[Ss]ec\.?|[Ss]chedule|[Cc]hapter|[Pp]art|ধারা|তফসিল)\s*[\dA-Z০-৯]+"
        r"(?:\([\w]+\))*(?:\s*(?:,|and|to|ও)\s*[\d০-৯]+[A-Z]*(?:\([\w]+\))*)*",
        "",
        text,
    )
    text = re.sub(r"\[[^\]]*\]|\*+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def rewrite_query(query, cache):
    """L1: Bangla -> English, lay words -> words of the Act (one short query)."""
    if query not in cache:
        prompt = (
            "You help search the Bangladesh Income Tax Act, 2023 (English text). Rewrite the user's "
            "question as ONE short English search query that uses the legal terms the Act itself would "
            "use (for example 'taxpayer's identification number', 'deduction of tax at source', "
            "'advance tax', 'income from rent'). If the question is in Bangla, translate it. "
            "Do not answer. Do not write section numbers. Output only the query.\n\n"
            f"Question: {query}\nSearch query:"
        )
        cache[query] = call_ollama(prompt, 60)
    return strip_numbers(cache[query]).split("\n")[0][:300]


def hyde_document(query, cache):
    if query not in cache:
        prompt = (
            "You are an expert on the Bangladesh Income Tax Act, 2023. Write a short passage, in the "
            "style of the Act itself and in English, that answers the question. Use the legal terms "
            "the Act would use. Do not write section numbers.\n\n"
            f"Question: {query}\n\nPassage:"
        )
        text = call_ollama(prompt, 200)
        cache[query] = text if len(text) > 10 else ""
    return strip_numbers(cache[query])


def doc2query(doc, cache):
    """L3: lay questions that this section answers (generated once per section, cached)."""
    key = doc["id"]
    if key not in cache:
        prompt = (
            f"Below is a provision of the Bangladesh Income Tax Act, 2023.\n\n{doc['section']} - "
            f"{doc['title']}\n{doc['text'][:1500]}\n\n"
            f"Write {D2Q_PER_DOC} different questions that an ordinary taxpayer in Bangladesh, who has "
            "never read the Act, could ask and that this provision answers. Use everyday words "
            "(e.g. TIN, AIT, TDS, FDR, salary, rent, late return). One question per line, English, "
            "no numbering, no section numbers."
        )
        cache[key] = call_ollama(prompt, 220)
    lines = [
        strip_numbers(re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", l))
        for l in cache[key].split("\n")
    ]
    return [l for l in lines if len(l) > 8][:D2Q_PER_DOC]


# ============================================================ corpus
def load_corpus(records_path, chunks_path):
    records = json.loads(Path(records_path).read_text(encoding="utf-8"))
    docs, by_rec = [], {}
    for r in records:
        if r.get("is_omitted") or not str(r.get("text", "")).strip():
            continue
        if r.get("record_type") == "section":
            sid, label = section_id(r["section"]), f"Section {r['section']}"
            path = " > ".join(
                x
                for x in (
                    f"{r.get('part', '')} {r.get('part_title', '')}".strip(),
                    f"{r.get('chapter', '')} {r.get('chapter_title', '')}".strip(),
                )
                if x
            )
        else:
            sid, label = "sch:" + r["id"].lower(), r.get("reference") or r["id"]
            path = " > ".join(
                x
                for x in (
                    f"{r.get('schedule', '')} {r.get('schedule_title', '')}".strip(),
                    f"{r.get('schedule_part', '')} {r.get('schedule_part_title', '')}".strip(),
                )
                if x.strip()
            )
        path = re.sub(r"\d{1,2}\[|\]|\[\s*See[^>]*?(?=>|$)", "", path).strip()
        by_rec[r["id"]] = len(docs)
        sched = re.sub(
            r"\s*Schedule\s*$", "", r.get("schedule", "") or ""
        ).strip()  # 'Sixth Schedule' -> 'Sixth'
        docs.append(
            {
                "id": r["id"],
                "section": label,
                "section_id": sid,
                "schedule": sched,
                "title": re.sub(r"\d{1,2}\[|\]", "", r.get("title", "")),
                "path": path,
                "cross_refs": r.get("cross_refs", []),
                "schedule_refs": r.get("schedule_refs", []),
                "last_amended_on": r.get("last_amended_on", ""),
                "amended_by": ", ".join(r.get("amended_by", [])),
                "text": r["text"],
                "text_bn": r.get("text_bn", ""),
            }
        )
    cf = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
    raw = cf["chunks"] if isinstance(cf, dict) else cf
    chunks = []
    for c in raw:
        d = by_rec.get(c.get("parent_id", ""))
        if d is None:
            m = re.match(
                r"(ita2023_(?:s\d+|sch\d+[\w]*?_p\d+|sch\d+_part\d+_text|sch\d+[\w]*))",
                c["chunk_id"],
            )
            d = by_rec.get(m.group(1)) if m else None
        if d is not None:
            chunks.append(
                {
                    "doc": d,
                    "lang": c.get("language", "en"),
                    "text": c["context_and_text"],
                }
            )
    by_section = {}
    for i, d in enumerate(docs):
        by_section.setdefault(d["section_id"], i)
    for d in docs:
        d["xref_docs"] = [by_section[s] for s in d["cross_refs"] if s in by_section]
        d["xsched_docs"] = [
            i
            for i, e in enumerate(docs)
            if e["schedule"] and e["schedule"] in d["schedule_refs"]
        ]
        d["card"] = (
            f"{d['section']}: {d['title']}\n{d['path']}\n{d['text'][:CARD_CHARS]}"
        )
    return docs, chunks, by_section


# ============================================================ retrieval
def to_sections(order, owner, limit=TOP_CHUNKS_PER_LIST):
    seen, out = set(), []
    for i in order[:limit]:
        d = owner[i]
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def rrf(rankings, k=RRF_K):
    scores = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + rank + 1)
    return scores


def minmax(x):
    x = np.asarray(x, dtype=float)
    return (
        np.ones_like(x)
        if x.max() - x.min() < 1e-9
        else (x - x.min()) / (x.max() - x.min())
    )


class Index:
    """One searchable list of texts (chunks, cards or synthetic questions), each pointing to a doc."""

    def __init__(self, name, texts, owner, dense, bm25_cls, lang=None):
        self.name, self.owner = name, owner
        self.lang = np.array(lang if lang is not None else ["en"] * len(texts))
        print(f"Encoding {len(texts)} {name} ...")
        self.emb = dense.encode(
            texts,
            batch_size=BATCH_EMBED,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self.bm25 = bm25_cls([tokenize(t) for t in texts])

    def rank(self, q_emb, q_text, use_bangla=True):
        mask = np.ones(len(self.owner), bool) if use_bangla else (self.lang == "en")
        out = []
        if q_emb is not None:
            out.append(
                to_sections(
                    np.argsort(-np.where(mask, self.emb @ q_emb, -1e9)), self.owner
                )
            )
        if q_text:
            out.append(
                to_sections(
                    np.argsort(
                        -np.where(mask, self.bm25.get_scores(tokenize(q_text)), -1e9)
                    ),
                    self.owner,
                )
            )
        return out


class Retriever:
    def __init__(self, docs, chunks, by_section, dense, reranker, bm25_cls, d2q=None):
        self.docs, self.chunks, self.by_section = docs, chunks, by_section
        self.dense, self.reranker = dense, reranker
        self.chunk_index = Index(
            "chunks",
            [c["text"] for c in chunks],
            [c["doc"] for c in chunks],
            dense,
            bm25_cls,
            [c["lang"] for c in chunks],
        )
        self.card_index = Index(
            "section cards",
            [d["card"] for d in docs],
            list(range(len(docs))),
            dense,
            bm25_cls,
        )
        self.d2q_index = None
        if d2q:
            texts, owner = [], []
            for i, qs in d2q.items():
                for q in qs:
                    texts.append(q)
                    owner.append(i)
            if texts:
                self.d2q_index = Index(
                    "synthetic questions", texts, owner, dense, bm25_cls
                )
        self.chunks_of = defaultdict(list)
        for i, c in enumerate(chunks):
            self.chunks_of[c["doc"]].append(i)
        self.def_doc = by_section.get("2")
        self.schedule_docs = [
            i for i, d in enumerate(docs) if d["section_id"].startswith("sch:")
        ]

    def enc(self, text):
        return (
            self.dense.encode([text], normalize_embeddings=True, convert_to_numpy=True)[
                0
            ]
            if text
            else None
        )

    def search(self, query, rewrite="", hypo="", cfg=None):
        cfg = cfg or {}
        gloss = glossary_terms(query) if cfg.get("rewrite", True) else ""
        q_en = " ".join(
            x for x in (rewrite, gloss) if x
        )  # English legal-vocabulary query
        q_emb, r_emb = self.enc(query), self.enc(q_en)
        h_emb = None
        if hypo:
            h_emb = (q_emb if r_emb is None else r_emb) + self.enc(hypo)
            h_emb /= np.linalg.norm(h_emb) + 1e-9
        ub = cfg.get("use_bangla", True)
        rankings = []
        for emb, txt in ((q_emb, query), (r_emb, q_en), (h_emb, hypo)):
            if emb is None and not txt:
                continue
            rankings += self.chunk_index.rank(emb, txt, ub)
            if cfg.get("cards", True):
                rankings += self.card_index.rank(emb, txt)
            if cfg.get("d2q", True) and self.d2q_index is not None and emb is not h_emb:
                rankings += self.d2q_index.rank(emb, txt)
        fused = rrf(rankings)
        top = max(fused.values(), default=0.0)

        named = sections_named(query)
        for sid in named:
            if sid in self.by_section:
                fused[self.by_section[sid]] = top + 1.0
        for sch in schedules_named(query):
            for d in self.schedule_docs:
                if self.docs[d]["schedule"] == sch:
                    fused[d] = max(fused.get(d, 0.0), top * 0.9)
        qtype = route(query) if cfg.get("route", True) else "general"
        if qtype == "definition" and self.def_doc is not None:
            fused[self.def_doc] = max(fused.get(self.def_doc, 0.0), top * 0.9)

        ranked = sorted(fused, key=fused.get, reverse=True)
        pool = ranked[: ROUTE_POOL[qtype] if cfg.get("route", True) else POOL]
        added = []
        if cfg.get("xref", True):
            for d in ranked[:XREF_FROM]:
                links = self.docs[d]["xref_docs"] + (
                    self.docs[d]["xsched_docs"] if qtype == "rate_amount" else []
                )
                taken = 0
                for x in links:
                    if taken >= XREF_PER_SECTION:
                        break
                    if x not in pool and x not in added and len(added) < XREF_MAX:
                        taken += 1
                        added.append(x)
                        fused[x] = max(fused.get(x, 0.0), fused[d] * 0.5)
        pool = pool + added

        # L4: reranker reads the question + its English legal rewrite; candidates = best chunks + card
        rq = f"{query}\n{rewrite}".strip() if rewrite else query
        best_emb = (
            h_emb if h_emb is not None else (r_emb if r_emb is not None else q_emb)
        )
        ci = self.chunk_index
        sim = ci.emb @ best_emb
        pairs, owner = [], []
        for d in pool:
            idx = [i for i in self.chunks_of[d] if ub or ci.lang[i] == "en"]
            idx = sorted(idx, key=lambda i: -sim[i])[:CHUNKS_PER_SECTION]
            texts = [self.chunks[i]["text"] for i in idx]
            if cfg.get("cards", True):
                texts.append(self.docs[d]["card"])
            for t in texts:
                pairs.append([rq, t])
                owner.append(d)
        scores = self.reranker.predict(
            pairs, batch_size=BATCH_RERANK, show_progress_bar=False
        )
        rr = {}
        for d, s in zip(owner, scores):
            rr[d] = max(rr.get(d, -1e9), float(s))
        w = cfg.get("rerank_weight", RERANK_WEIGHT)
        final = w * minmax([rr[d] for d in pool]) + (1 - w) * minmax(
            [fused[d] for d in pool]
        )
        for j, d in enumerate(pool):
            if self.docs[d]["section_id"] in named:
                final[j] += 10
        order = np.argsort(-final)[:TOP_K_FINAL]
        return [pool[j] for j in order], pool, qtype, added


# ============================================================ evaluation
BASE = dict(
    hyde=True,
    rewrite=True,
    cards=True,
    d2q=True,
    xref=True,
    route=True,
    use_bangla=True,
)
CONFIGS = {
    "full": BASE,
    "no_rewrite": {**BASE, "rewrite": False},
    "no_cards": {**BASE, "cards": False},
    "no_d2q": {**BASE, "d2q": False},
    "no_hyde": {**BASE, "hyde": False},
    "no_xref": {**BASE, "xref": False},
    "v3_like": {**BASE, "rewrite": False, "cards": False, "d2q": False},
}


def is_hit(doc, gs, gsch):
    return doc["section_id"] in gs or (
        bool(doc["schedule"]) and doc["schedule"] in gsch
    )


def bootstrap_ci(values, n=BOOTSTRAP, seed=0):
    v = np.asarray(values, float)
    if len(v) == 0:
        return (0.0, 0.0)
    means = v[np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))].mean(
        axis=1
    )
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(retriever, docs, questions, caches, cfg):
    rows = []
    for q, gt, lang in tqdm(questions, desc="Evaluating", leave=False):
        rw = rewrite_query(q, caches["rewrite"]) if cfg["rewrite"] else ""
        hy = hyde_document(q, caches["hyde"]) if cfg["hyde"] else ""
        top, pool, qtype, added = retriever.search(q, rw, hy, cfg)
        gs, gsch = sections_named(gt), schedules_named(gt)
        if re.search(r"finance act", gt, re.I) and not re.search(
            r"\bITA\b|income tax act|section|ধারা", gt, re.I
        ):
            gs, gsch = (
                set(),
                set(),
            )  # e.g. 'Finance Act 2024 / Schedule 1': not in this corpus
        m = [1 if is_hit(docs[d], gs, gsch) else 0 for d in top]
        in_pool = any(is_hit(docs[d], gs, gsch) for d in pool)
        first = next((i + 1 for i, v in enumerate(m) if v), 0)
        rows.append(
            {
                "Query": q,
                "Language": lang,
                "Rewrite": rw,
                "Ground_Truth": gt,
                "Type": qtype,
                "GT_sections": ", ".join(
                    sorted(gs, key=lambda s: int(re.sub(r"\D", "", s) or 0))
                    + sorted(f"{s} Schedule" for s in gsch)
                ),
                "Hit@1": int(sum(m[:1]) > 0),
                "Hit@3": int(sum(m[:3]) > 0),
                "Hit@5": int(sum(m[:5]) > 0),
                "Hit@10": int(sum(m[:10]) > 0),
                "RR": 1.0 / first if first else 0.0,
                "In_pool": int(in_pool),
                "Rank": first or "",
                "Failure": (
                    ""
                    if sum(m[:5])
                    else (
                        "retriever: not in reranker pool"
                        if not in_pool
                        else "reranker: in pool, ranked below 5"
                    )
                ),
                "Retrieved": ", ".join(docs[d]["section"] for d in top[:5]),
                "Top1_title": docs[top[0]]["title"] if top else "",
                "Added_by_xref": ", ".join(docs[d]["section"] for d in added),
                "Top1_amended_by": docs[top[0]]["amended_by"] if top else "",
                "GT_has_no_section": int(not gs and not gsch),
            }
        )
    return pd.DataFrame(rows)


def summarize(df):
    s = df[df["GT_has_no_section"] == 0]
    n = len(s)

    def pct(x):
        return f"{100 * x:.2f}%"

    if not n:
        return {"Queries": len(df), "Scored": 0}
    lo, hi = bootstrap_ci(s["Hit@5"])
    return {
        "Queries": len(df),
        "Scored (ground truth names a section/schedule)": n,
        "Hit@1": pct(s["Hit@1"].mean()),
        "Hit@3": pct(s["Hit@3"].mean()),
        "Hit@5": pct(s["Hit@5"].mean()),
        "Hit@5 95% CI": f"{pct(lo)} - {pct(hi)}",
        "Hit@10": pct(s["Hit@10"].mean()),
        "MRR": f"{s['RR'].mean():.4f}",
        "Pool recall": pct(s["In_pool"].mean()),
        "Failures: retriever": int(s["Failure"].str.startswith("retriever").sum()),
        "Failures: reranker": int(s["Failure"].str.startswith("reranker").sum()),
    }


def read_questions(path, sheet):
    if str(path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet or 0)
    else:
        df = pd.read_csv(path)
    if "Question" not in df.columns or "Ground Truth Legal Basis" not in df.columns:
        raise ValueError("Need the columns 'Question' and 'Ground Truth Legal Basis'.")
    out = []
    for _, r in df.iterrows():
        q, g = str(r["Question"]).strip(), str(r["Ground Truth Legal Basis"]).strip()
        if not q or q.lower() == "nan":
            continue
        g = "" if g.lower() == "nan" else g
        lang = str(r.get("Language", "")) or (
            "Bangla" if BANGLA.search(q) else "English"
        )
        if lang.lower() == "nan" or not lang:
            lang = "Bangla" if BANGLA.search(q) else "English"
        out.append((q, g, lang))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(RECORDS_PATH))
    ap.add_argument("--chunks", default=str(CHUNKS_PATH))
    ap.add_argument(
        "--questions", default=GOOGLE_SHEET_CSV_URL, help=".xlsx / .csv path or CSV URL"
    )
    ap.add_argument(
        "--sheet", default="", help="sheet name for .xlsx (e.g. Corrected_GT)"
    )
    ap.add_argument("--no-hyde", action="store_true")
    ap.add_argument(
        "--no-d2q", action="store_true", help="skip the synthetic-question index"
    )
    ap.add_argument("--rerank-weight", type=float, default=RERANK_WEIGHT)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder, SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    docs, chunks, by_section = load_corpus(args.records, args.chunks)
    print(
        f"{len(docs)} records, {len(chunks)} chunks ({sum(c['lang'] == 'bn' for c in chunks)} Bangla), "
        f"device {device}"
    )

    caches = {"rewrite": load_cache(REWRITE_CACHE), "hyde": load_cache(HYDE_CACHE)}
    d2q = {}
    if not args.no_d2q:
        cache = load_cache(D2Q_CACHE)
        todo = sum(1 for d in docs if d["id"] not in cache)
        if todo:
            print(
                f"Generating synthetic questions for {todo} sections (once; cached in {D2Q_CACHE}) ..."
            )
        for n, (i, d) in enumerate(tqdm(list(enumerate(docs)), desc="doc2query")):
            d2q[i] = doc2query(d, cache)
            if n % 10 == 0:
                save_cache(D2Q_CACHE, cache)
        save_cache(D2Q_CACHE, cache)
        print(f"  {sum(len(v) for v in d2q.values())} synthetic questions")

    dense = SentenceTransformer("BAAI/bge-m3", device=device)
    dense.max_seq_length = MAX_LEN_EMBED
    reranker = CrossEncoder(
        "BAAI/bge-reranker-v2-m3", device=device, max_length=MAX_LEN_RERANK
    )
    retriever = Retriever(docs, chunks, by_section, dense, reranker, BM25Okapi, d2q)
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    questions = read_questions(args.questions, args.sheet)
    if args.limit:
        questions = questions[: args.limit]
    names = list(CONFIGS) if args.ablation else ["full"]
    results, details = {}, None
    for name in names:
        cfg = {**CONFIGS[name], "rerank_weight": args.rerank_weight}
        if args.no_hyde:
            cfg["hyde"] = False
        if args.no_d2q:
            cfg["d2q"] = False
        print(f"\n== {name}")
        df = evaluate(retriever, docs, questions, caches, cfg)
        results[name] = summarize(df)
        if name == "full":
            details = df
        save_cache(REWRITE_CACHE, caches["rewrite"])
        save_cache(HYDE_CACHE, caches["hyde"])

    summary = pd.DataFrame(
        {"Metric": list(results["full"]), "Value": list(results["full"].values())}
    )
    scored = details[details["GT_has_no_section"] == 0]

    def group(col):
        return (
            scored.groupby(col)
            .agg(
                Questions=("Query", "size"),
                Hit1=("Hit@1", "mean"),
                Hit5=("Hit@5", "mean"),
                MRR=("RR", "mean"),
                Pool_recall=("In_pool", "mean"),
            )
            .reset_index()
        )

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        if args.ablation:
            pd.DataFrame(results).T.reset_index(names="Config").to_excel(
                w, sheet_name="Ablation", index=False
            )
        group("Language").to_excel(w, sheet_name="ByLanguage", index=False)
        group("Type").to_excel(w, sheet_name="ByType", index=False)
        details.to_excel(w, sheet_name="Details", index=False)
        details[(details["Hit@5"] == 0) & (details["GT_has_no_section"] == 0)].to_excel(
            w, sheet_name="Failures", index=False
        )
    print("\n" + summary.to_string(index=False))
    print("\n" + group("Language").to_string(index=False))
    if args.ablation:
        print(
            "\n"
            + pd.DataFrame(results)
            .T[["Hit@1", "Hit@5", "MRR", "Pool recall"]]
            .to_string()
        )
    print(f"\nSaved {OUT_XLSX}")


if __name__ == "__main__":
    main()
