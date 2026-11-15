"""
evaluate_retrieval2.py (v5) - layered legal RAG retrieval for the Income Tax Act 2023 + evaluation.

v4 layers (kept): L0 section/schedule parser, L1 query rewrite + glossary, L2 section cards,
L3 doc2query synthetic questions, L4 cross-encoder reranker, HyDE, RRF, routing.

v5 changes, aimed at the v3/v4 failure split (most misses were RERANKER failures):
 R1  BGE-M3 all three modes from ONE forward pass: dense + learned sparse + ColBERT
     multi-vector (BM25 kept as an extra list). BM25 alone cannot match inflected Bangla.
 R2  Query decomposition: multi-issue questions -> up to 3 sub-queries, each retrieved,
     all fused by RRF (ground truth often names several sections).
 R3  Cross-reference graph: Personalized PageRank from the top hits over cross_refs
     (and schedule refs for rate questions) replaces the fixed 0.5 boost.
 R4  LLM listwise rerank (RankGPT-style) of the top LISTWISE_K sections, fused with the
     cross-encoder and fusion scores.
 R5  Fine-tuned models: --reranker / --embedder accept the folders written by
     scripts/train_retrieval_models.py (trained on doc2query questions only).
 R6  Evaluation hygiene: fixed dev/test split (tune on dev, report test), Recall@k for
     multi-section answers, paired bootstrap + McNemar test in the ablation.
 fix doc2query parsing: only lines that are questions are indexed (the cache also holds
     the LLM's preamble and explanations).

All LLM outputs (rewrite, HyDE, doc2query, decomposition, listwise ranking) are cached in
data/*.json. The LLM never sees the ground truth.

Usage (from the project folder):
  python scripts\\evaluate_retrieval2.py --questions data\\QA_Dataset\\Income_Tax_audited.xlsx --sheet Corrected_GT
  python scripts\\evaluate_retrieval2.py ... --reranker models\\reranker-ita2023 --embedder models\\bge-m3-ita2023
  python scripts\\evaluate_retrieval2.py ... --ablation            # LLM-heavy the first time
  python scripts\\evaluate_retrieval2.py ... --no-listwise --no-decompose   # no new LLM calls
"""

import argparse
import gc
import hashlib
import json
import re
from collections import defaultdict, namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ============================================================ CONFIG
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RECORDS_PATH = Path("data/Knowledge_Dataset/income_tax_act_2023_enhanced.json")
CHUNKS_PATH = Path("data/Knowledge_Dataset/income_tax_act_2023_enhanced_chunks.json")
GOOGLE_SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/17ew_umfBKtAeTUEBFtBknNoyQR5KbWRSgceh47DNeYY"
    "/export?format=csv&gid=0"
)
HYDE_CACHE = Path("data/hyde_cache.json")
REWRITE_CACHE = Path("data/rewrite_cache.json")
D2Q_CACHE = Path("data/doc2query_cache.json")
DECOMPOSE_CACHE = Path("data/decompose_cache.json")
LISTWISE_CACHE = Path("data/listwise_cache.json")
CACHE_PATHS = {
    "rewrite": REWRITE_CACHE,
    "hyde": HYDE_CACHE,
    "decompose": DECOMPOSE_CACHE,
    "listwise": LISTWISE_CACHE,
}
OUT_XLSX = "retrieval_v5_results.xlsx"

TOP_CHUNKS_PER_LIST = 200
RRF_K = 60
POOL = 30
XREF_FROM, XREF_MAX, XREF_PER_SECTION = 10, 15, 3
PPR_ALPHA, PPR_ITERS = 0.5, 30  # restart probability: high = stay close to the seed hits
CHUNKS_PER_SECTION = 3
TOP_K_FINAL = 10
RERANK_WEIGHT = 0.6
LISTWISE_K = 15
LISTWISE_WEIGHT = 0.3
LISTWISE_CHARS = 350
MAX_SUBQUERIES = 3
MAX_LEN_EMBED = 512
MAX_LEN_RERANK = 512
BATCH_EMBED = 16
BATCH_RERANK = 8
BOOTSTRAP = 10000
D2Q_PER_DOC = 6
CARD_CHARS = 600
DEV_PERCENT = 30  # fixed split by question hash: tune on dev, report test

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
            tok = tokens[i]
            if tok[0].isdigit():
                if (
                    i + 2 < len(tokens)
                    and tokens[i + 1] in ("to", "–", "—", "-")
                    and tokens[i + 2].isdigit()
                ):
                    a, b = int(re.sub(r"\D", "", tok)), int(tokens[i + 2])
                    if 0 < b - a <= 60:
                        found.update(str(n) for n in range(a, b + 1))
                        i += 3
                        continue
                found.add(tok)
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
def call_ollama(prompt, num_predict=200, num_ctx=4096):
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": num_predict,
                    "num_ctx": num_ctx,
                },
            },
            timeout=300,
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


def save_caches(caches):
    for name, path in CACHE_PATHS.items():
        if name in caches:
            save_cache(path, caches[name])


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


def clean_line(line):
    line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
    line = re.sub(r"^\**\s*(?:Q(?:uestion)?\s*\d+\s*[:.)-])\s*", "", line, flags=re.I)
    return strip_numbers(line)


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


def decompose_query(query, cache):
    """R2: split a multi-issue question into sub-queries ([] when it has one issue)."""
    if query not in cache:
        prompt = (
            "Split the user's tax question into the separate legal issues it raises, written as short "
            "English search queries that use the legal terms of the Bangladesh Income Tax Act, 2023. "
            f"If it raises only one issue, output one line. At most {MAX_SUBQUERIES} lines, one query "
            "per line, no numbering, no section numbers. Do not answer.\n\n"
            f"Question: {query}\nQueries:"
        )
        cache[query] = call_ollama(prompt, 120)
    lines = [clean_line(l) for l in cache[query].split("\n")]
    lines = [l for l in lines if len(l) > 8][:MAX_SUBQUERIES]
    return lines if len(lines) > 1 else []


def parse_questions(raw):
    """Keep only lines that are questions (the cache also holds preambles and explanations)."""
    out = []
    for line in str(raw).split("\n"):
        line = clean_line(line)
        if line.endswith("?") and len(line) > 12 and line not in out:
            out.append(line)
    return out[:D2Q_PER_DOC]


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
    return parse_questions(cache[key])


def listwise_rank(query, docs, cand, cache):
    """R4: the LLM orders the candidate provisions (RankGPT-style). Returns doc indices, best first.
    Candidates are shown in a fixed (corpus) order so the cache is reused across configurations
    and the LLM is not biased by our own ranking."""
    cand = sorted(cand)
    key = query + "||" + ",".join(docs[d]["id"] for d in cand)
    if key not in cache:
        lines = []
        for n, d in enumerate(cand, 1):
            doc = docs[d]
            text = re.sub(r"\s+", " ", doc["text"])[:LISTWISE_CHARS]
            lines.append(f"[{n}] {doc['title']} ({doc['path']}): {text}")
        prompt = (
            "You rank provisions of the Bangladesh Income Tax Act, 2023 for a user's question.\n\n"
            f"Question: {query}\n\nProvisions:\n" + "\n".join(lines) + "\n\n"
            "List the numbers of the provisions that are needed to answer the question, most relevant "
            "first, separated by commas. Output only the numbers. If none is relevant, output 0."
        )
        cache[key] = call_ollama(prompt, 60, num_ctx=8192)
    order = []
    for x in re.findall(r"\d+", cache[key]):
        n = int(x)
        if 1 <= n <= len(cand) and cand[n - 1] not in order:
            order.append(cand[n - 1])
    return order


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


# ============================================================ BGE-M3: dense + sparse + ColBERT
QVec = namedtuple("QVec", "dense lex colbert")


class M3Encoder:
    """Runs the BGE-M3 transformer once and derives all three BGE-M3 outputs:
    dense = normalised [CLS]; sparse = relu(sparse_linear(h)) max-pooled per token id;
    ColBERT = normalised colbert_linear(h[1:]). The two heads are loaded from the model folder
    (fine-tuned model) or from the BAAI/bge-m3 repository."""

    def __init__(self, st_model, device, model_dir=None):
        import torch
        from huggingface_hub import hf_hub_download

        self.torch, self.st, self.device = torch, st_model, device
        hidden = st_model.get_sentence_embedding_dimension()
        dtype = next(st_model.parameters()).dtype

        def head(name, out_dim):
            local = Path(model_dir) / name if model_dir else None
            path = (
                str(local)
                if local is not None and local.exists()
                else hf_hub_download(EMBED_MODEL, name)
            )
            lin = torch.nn.Linear(hidden, out_dim)
            lin.load_state_dict(torch.load(path, map_location="cpu"))
            return lin.to(device=device, dtype=dtype).eval()

        self.sparse_head = head("sparse_linear.pt", 1)
        self.colbert_head = head("colbert_linear.pt", hidden)
        tok = st_model.tokenizer
        self.skip = {
            tok.cls_token_id,
            tok.eos_token_id,
            tok.pad_token_id,
            tok.unk_token_id,
        }
        self.cache = {}

    def encode(self, texts, colbert=False, progress=False):
        torch = self.torch
        res = [None] * len(texts)
        order = np.argsort([-len(t) for t in texts])
        steps = range(0, len(texts), BATCH_EMBED)
        if progress:
            steps = tqdm(steps, desc="encode")
        with torch.inference_mode():
            for s in steps:
                idx = order[s : s + BATCH_EMBED]
                feats = self.st.tokenize([texts[i] for i in idx])
                ids_t = feats["input_ids"].to(self.device)
                mask_t = feats["attention_mask"].to(self.device)
                h = self.st[0].auto_model(
                    input_ids=ids_t, attention_mask=mask_t
                ).last_hidden_state
                dense = torch.nn.functional.normalize(h[:, 0].float(), dim=-1)
                weights = torch.relu(self.sparse_head(h)).squeeze(-1).float()
                cvec = None
                if colbert:
                    cvec = self.colbert_head(h[:, 1:]).float() * mask_t[:, 1:, None]
                    cvec = torch.nn.functional.normalize(cvec, dim=-1).cpu().numpy()
                dense, weights = dense.cpu().numpy(), weights.cpu().numpy()
                ids, mask = ids_t.cpu().numpy(), mask_t.cpu().numpy()
                for j, i in enumerate(idx):
                    lex = {}
                    for t, w, m in zip(ids[j], weights[j], mask[j]):
                        t = int(t)
                        if m and w > 0 and t not in self.skip and w > lex.get(t, 0.0):
                            lex[t] = float(w)
                    n = int(mask[j].sum()) - 1
                    res[i] = QVec(
                        dense[j],
                        lex,
                        cvec[j, :n].astype(np.float16) if colbert else None,
                    )
        return res

    def query(self, text):
        if not text:
            return None
        if text not in self.cache:
            self.cache[text] = self.encode([text], colbert=True)[0]
        return self.cache[text]


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
    """One searchable list of texts (chunks, cards or synthetic questions), each pointing to a doc.
    Ranks by BGE-M3 dense, BGE-M3 sparse, BM25 and (optionally) ColBERT late interaction."""

    def __init__(
        self, name, texts, owner, m3, bm25_cls, lang=None, colbert=False, colbert_device="cpu"
    ):
        import torch

        self.torch, self.name, self.owner = torch, name, owner
        self.lang = np.array(lang if lang is not None else ["en"] * len(texts))
        print(f"Encoding {len(texts)} {name} ...")
        vecs = m3.encode(texts, colbert=colbert, progress=True)
        self.emb = np.stack([v.dense for v in vecs])
        inv = defaultdict(lambda: ([], []))
        for i, v in enumerate(vecs):
            for t, w in v.lex.items():
                inv[t][0].append(i)
                inv[t][1].append(w)
        self.inv = {
            t: (np.array(a), np.array(b, dtype=np.float32)) for t, (a, b) in inv.items()
        }
        self.bm25 = bm25_cls([tokenize(t) for t in texts])
        self.colbert = None
        if colbert:
            length = max(len(v.colbert) for v in vecs)
            dtype = torch.float16 if str(colbert_device).startswith("cuda") else torch.float32
            mat = torch.zeros(len(vecs), length, self.emb.shape[1], dtype=dtype)
            msk = torch.zeros(len(vecs), length, dtype=torch.bool)
            for i, v in enumerate(vecs):
                mat[i, : len(v.colbert)] = torch.from_numpy(v.colbert.astype(np.float32))
                msk[i, : len(v.colbert)] = True
            self.colbert = (mat.to(colbert_device), msk.to(colbert_device))

    def sparse_scores(self, lex):
        s = np.zeros(len(self.owner), np.float32)
        for t, w in lex.items():
            if t in self.inv:
                idx, ws = self.inv[t]
                np.add.at(s, idx, w * ws)
        return s

    def colbert_scores(self, qc):
        torch = self.torch
        mat, msk = self.colbert
        q = torch.from_numpy(qc.astype(np.float32)).to(mat.device, mat.dtype)
        out = []
        with torch.inference_mode():
            for s in range(0, len(mat), 128):
                sim = torch.einsum("qh,nlh->nql", q, mat[s : s + 128])
                sim = sim.masked_fill(~msk[s : s + 128, None, :], -1e4)
                out.append(sim.max(-1).values.float().mean(-1))
        return torch.cat(out).cpu().numpy()

    def rank(self, q, q_text, use_bangla=True, cfg=None):
        cfg = cfg or {}
        mask = np.ones(len(self.owner), bool) if use_bangla else (self.lang == "en")

        def ordered(scores, positive_only=False):
            scores = np.where(mask, scores, -1e9)
            order = np.argsort(-scores)
            if positive_only:
                order = order[scores[order] > 0]
            return to_sections(order, self.owner)

        out = []
        if q is not None:
            out.append(ordered(self.emb @ q.dense))
            if cfg.get("sparse", True) and q.lex:
                out.append(ordered(self.sparse_scores(q.lex), positive_only=True))
            if cfg.get("colbert", True) and self.colbert is not None and q.colbert is not None:
                out.append(ordered(self.colbert_scores(q.colbert)))
        if q_text and cfg.get("bm25", True):
            out.append(
                ordered(self.bm25.get_scores(tokenize(q_text)), positive_only=True)
            )
        return [r for r in out if r]


class Retriever:
    def __init__(
        self, docs, chunks, by_section, m3, reranker, bm25_cls, d2q=None, colbert_device="cpu"
    ):
        self.docs, self.chunks, self.by_section = docs, chunks, by_section
        self.m3, self.reranker = m3, reranker
        self.chunk_index = Index(
            "chunks",
            [c["text"] for c in chunks],
            [c["doc"] for c in chunks],
            m3,
            bm25_cls,
            [c["lang"] for c in chunks],
            colbert=True,
            colbert_device=colbert_device,
        )
        self.card_index = Index(
            "section cards", [d["card"] for d in docs], list(range(len(docs))), m3, bm25_cls
        )
        self.d2q_index = None
        if d2q:
            texts, owner = [], []
            for i, qs in d2q.items():
                for q in qs:
                    texts.append(q)
                    owner.append(i)
            if texts:
                self.d2q_index = Index("synthetic questions", texts, owner, m3, bm25_cls)
        self.chunks_of = defaultdict(list)
        for i, c in enumerate(chunks):
            self.chunks_of[c["doc"]].append(i)
        self.def_doc = by_section.get("2")
        self.schedule_docs = [
            i for i, d in enumerate(docs) if d["section_id"].startswith("sch:")
        ]
        self.trans = self._transition(with_schedules=False)
        self.trans_sched = self._transition(with_schedules=True)

    # ---------------------------------------------------- R3: cross-reference graph
    def _transition(self, with_schedules):
        n = len(self.docs)
        a = np.zeros((n, n), np.float32)
        for i, d in enumerate(self.docs):
            for j in d["xref_docs"] + (d["xsched_docs"] if with_schedules else []):
                if j != i:
                    a[i, j] = 1.0
        rows = a.sum(1, keepdims=True)
        return np.divide(a, rows, out=np.zeros_like(a), where=rows > 0)

    def ppr(self, seeds, weights, with_schedules):
        m = self.trans_sched if with_schedules else self.trans
        s = np.zeros(len(self.docs), np.float32)
        for d, w in zip(seeds, weights):
            s[d] = w
        s /= s.sum() + 1e-12
        dangling = m.sum(1) == 0
        p = s.copy()
        for _ in range(PPR_ITERS):
            p = PPR_ALPHA * s + (1 - PPR_ALPHA) * (p @ m + p[dangling].sum() * s)
        return p

    def expand(self, fused, ranked, pool, qtype, cfg):
        """Add / boost sections linked to the top hits. Returns the sections added to the pool."""
        added = []
        seeds = ranked[:XREF_FROM]
        if not seeds:
            return added
        top = fused[seeds[0]]
        if cfg.get("ppr", True):
            p = self.ppr(seeds, [fused[d] for d in seeds], qtype == "rate_amount")
            p[seeds] = 0.0
            if p.max() <= 0:
                return added
            pn = p / p.max()
            for x in np.argsort(-pn):
                if pn[x] <= 0.05:
                    break
                fused[x] = max(fused.get(x, 0.0), 0.5 * top * float(pn[x]))
                if x not in pool and len(added) < XREF_MAX:
                    added.append(int(x))
            return added
        for d in seeds:  # v4 behaviour: fixed boost of direct references
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
        return added

    # ---------------------------------------------------- main search
    def search(self, query, rewrite="", hypo="", subqueries=(), cfg=None, listwise_cache=None):
        cfg = cfg or {}
        gloss = glossary_terms(query) if cfg.get("rewrite", True) else ""
        q_en = " ".join(x for x in (rewrite, gloss) if x)  # English legal-vocabulary query
        qv, rv = self.m3.query(query), self.m3.query(q_en)
        hv = None
        if hypo:
            h = self.m3.query(hypo)
            base = rv if rv is not None else qv
            dense = base.dense + h.dense
            hv = QVec(dense / (np.linalg.norm(dense) + 1e-9), h.lex, None)
        ub = cfg.get("use_bangla", True)

        rankings = []
        variants = [(qv, query, True), (rv, q_en, True), (hv, hypo, False)]
        variants += [(self.m3.query(s), s, True) for s in subqueries]
        for vec, txt, use_d2q in variants:
            if vec is None and not txt:
                continue
            rankings += self.chunk_index.rank(vec, txt, ub, cfg)
            if cfg.get("cards", True):
                rankings += self.card_index.rank(vec, txt, True, cfg)
            if cfg.get("d2q", True) and self.d2q_index is not None and use_d2q:
                rankings += self.d2q_index.rank(vec, txt, True, cfg)
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
        added = self.expand(fused, ranked, pool, qtype, cfg) if cfg.get("xref", True) else []
        pool = pool + added

        # L4: cross-encoder reads the question + its English legal rewrite
        rq = f"{query}\n{rewrite}".strip() if rewrite else query
        best = hv if hv is not None else (rv if rv is not None else qv)
        ci = self.chunk_index
        sim = ci.emb @ best.dense
        pairs, owner = [], []
        for d in pool:
            idx = [i for i in self.chunks_of[d] if ub or ci.lang[i] == "en"]
            idx = sorted(idx, key=lambda i: -sim[i])[:CHUNKS_PER_SECTION]
            texts = [self.chunks[i]["text"] for i in idx]
            if cfg.get("cards", True) or not texts:
                texts.append(self.docs[d]["card"])
            for t in texts:
                pairs.append([rq, t])
                owner.append(d)
        scores = self.reranker.predict(pairs, batch_size=BATCH_RERANK, show_progress_bar=False)
        rr = {}
        for d, s in zip(owner, scores):
            rr[d] = max(rr.get(d, -1e9), float(s))
        w = cfg.get("rerank_weight", RERANK_WEIGHT)
        final = w * minmax([rr[d] for d in pool]) + (1 - w) * minmax([fused[d] for d in pool])

        # R4: LLM listwise rerank of the current top LISTWISE_K
        if cfg.get("listwise", True) and listwise_cache is not None:
            lw = cfg.get("listwise_weight", LISTWISE_WEIGHT)
            head = [pool[j] for j in np.argsort(-final)[:LISTWISE_K]]
            order = listwise_rank(query if not rewrite else rq, self.docs, head, listwise_cache)
            llm = {d: (LISTWISE_K - r) / LISTWISE_K for r, d in enumerate(order)}
            final = (1 - lw) * final + lw * np.array([llm.get(d, 0.0) for d in pool])

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
    sparse=True,
    colbert=True,
    bm25=True,
    decompose=True,
    ppr=True,
    listwise=True,
)
CONFIGS = {
    "full": BASE,
    "no_listwise": {**BASE, "listwise": False},
    "no_decompose": {**BASE, "decompose": False},
    "no_ppr": {**BASE, "ppr": False},
    "no_colbert": {**BASE, "colbert": False},
    "no_sparse": {**BASE, "sparse": False},
    "no_d2q": {**BASE, "d2q": False},
    "no_cards": {**BASE, "cards": False},
    "no_rewrite": {**BASE, "rewrite": False},
    "no_hyde": {**BASE, "hyde": False},
    "v4_like": {
        **BASE,
        "sparse": False,
        "colbert": False,
        "decompose": False,
        "ppr": False,
        "listwise": False,
    },
}


def split_of(question):
    h = int(hashlib.md5(question.encode("utf-8")).hexdigest()[:8], 16)
    return "dev" if h % 100 < DEV_PERCENT else "test"


def is_hit(doc, gs, gsch):
    return doc["section_id"] in gs or (bool(doc["schedule"]) and doc["schedule"] in gsch)


def coverage(docs, dlist, gs, gsch):
    secs = {docs[d]["section_id"] for d in dlist}
    schs = {docs[d]["schedule"] for d in dlist if docs[d]["schedule"]}
    return len(gs & secs) + len(gsch & schs)


def bootstrap_ci(values, n=BOOTSTRAP, seed=0):
    v = np.asarray(values, float)
    if len(v) == 0:
        return (0.0, 0.0)
    means = v[np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_p(a, b):
    """Exact McNemar test on paired 0/1 outcomes."""
    from scipy.stats import binomtest

    a, b = np.asarray(a), np.asarray(b)
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    if n10 + n01 == 0:
        return 1.0
    return float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)


def evaluate(retriever, docs, questions, caches, cfg):
    rows = []
    for n, (q, gt, lang) in enumerate(tqdm(questions, desc="Evaluating", leave=False)):
        rw = rewrite_query(q, caches["rewrite"]) if cfg["rewrite"] else ""
        hy = hyde_document(q, caches["hyde"]) if cfg["hyde"] else ""
        sub = decompose_query(q, caches["decompose"]) if cfg["decompose"] else []
        top, pool, qtype, added = retriever.search(
            q, rw, hy, sub, cfg, caches["listwise"] if cfg["listwise"] else None
        )
        if n % 20 == 19:
            save_caches(caches)
        gs, gsch = sections_named(gt), schedules_named(gt)
        if re.search(r"finance act", gt, re.I) and not re.search(
            r"\bITA\b|income tax act|section|ধারা", gt, re.I
        ):
            gs, gsch = set(), set()  # e.g. 'Finance Act 2024 / Schedule 1': not in this corpus
        units = len(gs) + len(gsch)
        m = [1 if is_hit(docs[d], gs, gsch) else 0 for d in top]
        in_pool = any(is_hit(docs[d], gs, gsch) for d in pool)
        first = next((i + 1 for i, v in enumerate(m) if v), 0)
        rows.append(
            {
                "Query": q,
                "Language": lang,
                "Split": split_of(q),
                "Rewrite": rw,
                "Subqueries": " | ".join(sub),
                "Ground_Truth": gt,
                "Type": qtype,
                "GT_sections": ", ".join(
                    sorted(gs, key=lambda s: int(re.sub(r"\D", "", s) or 0))
                    + sorted(f"{s} Schedule" for s in gsch)
                ),
                "GT_units": units,
                "Hit@1": int(sum(m[:1]) > 0),
                "Hit@3": int(sum(m[:3]) > 0),
                "Hit@5": int(sum(m[:5]) > 0),
                "Hit@10": int(sum(m[:10]) > 0),
                "Recall@5": coverage(docs, top[:5], gs, gsch) / units if units else 0.0,
                "Recall@10": coverage(docs, top[:10], gs, gsch) / units if units else 0.0,
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
    multi = s[s["GT_units"] > 1]
    return {
        "Queries": len(df),
        "Scored (ground truth names a section/schedule)": n,
        "Hit@1": pct(s["Hit@1"].mean()),
        "Hit@3": pct(s["Hit@3"].mean()),
        "Hit@5": pct(s["Hit@5"].mean()),
        "Hit@5 95% CI": f"{pct(lo)} - {pct(hi)}",
        "Hit@10": pct(s["Hit@10"].mean()),
        "MRR": f"{s['RR'].mean():.4f}",
        "Recall@5": pct(s["Recall@5"].mean()),
        "Recall@10": pct(s["Recall@10"].mean()),
        "Recall@10 (multi-section GT)": pct(multi["Recall@10"].mean()) if len(multi) else "",
        "Pool recall": pct(s["In_pool"].mean()),
        "Failures: retriever": int(s["Failure"].str.startswith("retriever").sum()),
        "Failures: reranker": int(s["Failure"].str.startswith("reranker").sum()),
    }


def compare(full, other):
    """Paired comparison of Hit@5 on the same scored questions (full minus other)."""
    a = full[full["GT_has_no_section"] == 0]["Hit@5"].to_numpy()
    b = other[other["GT_has_no_section"] == 0]["Hit@5"].to_numpy()
    lo, hi = bootstrap_ci(a - b)
    return {
        "dHit@5 (full - config)": f"{100 * (a - b).mean():+.2f} pts",
        "dHit@5 95% CI": f"{100 * lo:+.2f} .. {100 * hi:+.2f}",
        "McNemar p": f"{mcnemar_p(a, b):.4f}",
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
        lang = str(r.get("Language", ""))
        if lang.lower() == "nan" or not lang:
            lang = "Bangla" if BANGLA.search(q) else "English"
        out.append((q, g, lang))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(RECORDS_PATH))
    ap.add_argument("--chunks", default=str(CHUNKS_PATH))
    ap.add_argument("--questions", default=GOOGLE_SHEET_CSV_URL, help=".xlsx / .csv path or CSV URL")
    ap.add_argument("--sheet", default="", help="sheet name for .xlsx (e.g. Corrected_GT)")
    ap.add_argument("--embedder", default=EMBED_MODEL, help="BGE-M3 name or fine-tuned folder")
    ap.add_argument("--reranker", default=RERANK_MODEL, help="reranker name or fine-tuned folder")
    ap.add_argument("--colbert-device", default="cpu", help="cpu (saves GPU memory for Ollama) or cuda")
    ap.add_argument("--no-hyde", action="store_true")
    ap.add_argument("--no-d2q", action="store_true", help="skip the synthetic-question index")
    ap.add_argument("--no-listwise", action="store_true", help="skip the LLM listwise reranker")
    ap.add_argument("--no-decompose", action="store_true", help="skip query decomposition")
    ap.add_argument("--rerank-weight", type=float, default=RERANK_WEIGHT)
    ap.add_argument("--listwise-weight", type=float, default=LISTWISE_WEIGHT)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT_XLSX)
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

    caches = {name: load_cache(path) for name, path in CACHE_PATHS.items()}
    d2q = {}
    if not args.no_d2q:
        cache = load_cache(D2Q_CACHE)
        todo = sum(1 for d in docs if d["id"] not in cache)
        if todo:
            print(f"Generating synthetic questions for {todo} sections (once; cached in {D2Q_CACHE}) ...")
        for n, (i, d) in enumerate(tqdm(list(enumerate(docs)), desc="doc2query")):
            d2q[i] = doc2query(d, cache)
            if n % 10 == 0 and todo:
                save_cache(D2Q_CACHE, cache)
        save_cache(D2Q_CACHE, cache)
        print(f"  {sum(len(v) for v in d2q.values())} synthetic questions")

    dense = SentenceTransformer(args.embedder, device=device)
    dense.max_seq_length = MAX_LEN_EMBED
    reranker = CrossEncoder(args.reranker, device=device, max_length=MAX_LEN_RERANK)
    if device == "cuda":  # fp16 halves GPU memory, leaving room for Ollama
        dense.half()
        reranker.half()
    model_dir = args.embedder if Path(args.embedder).exists() else None
    m3 = M3Encoder(dense, device, model_dir)
    retriever = Retriever(
        docs, chunks, by_section, m3, reranker, BM25Okapi, d2q, args.colbert_device
    )
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    questions = read_questions(args.questions, args.sheet)
    if args.limit:
        questions = questions[: args.limit]
    names = list(CONFIGS) if args.ablation else ["full"]
    results, frames = {}, {}
    for name in names:
        cfg = {
            **CONFIGS[name],
            "rerank_weight": args.rerank_weight,
            "listwise_weight": args.listwise_weight,
        }
        for flag, key in (
            (args.no_hyde, "hyde"),
            (args.no_d2q, "d2q"),
            (args.no_listwise, "listwise"),
            (args.no_decompose, "decompose"),
        ):
            if flag:
                cfg[key] = False
        print(f"\n== {name}")
        df = evaluate(retriever, docs, questions, caches, cfg)
        frames[name] = df
        results[name] = summarize(df)
        save_caches(caches)

    details = frames["full"]
    summary = pd.DataFrame(
        {
            "All": summarize(details),
            "Dev (tune here)": summarize(details[details["Split"] == "dev"]),
            "Test (report this)": summarize(details[details["Split"] == "test"]),
        }
    ).reset_index(names="Metric")
    scored = details[details["GT_has_no_section"] == 0]

    def group(col):
        return (
            scored.groupby(col)
            .agg(
                Questions=("Query", "size"),
                Hit1=("Hit@1", "mean"),
                Hit5=("Hit@5", "mean"),
                Recall10=("Recall@10", "mean"),
                MRR=("RR", "mean"),
                Pool_recall=("In_pool", "mean"),
            )
            .reset_index()
        )

    ablation = None
    if args.ablation:
        ablation = pd.DataFrame(
            {
                name: {
                    **results[name],
                    "Test Hit@5": summarize(frames[name][frames[name]["Split"] == "test"]).get("Hit@5", ""),
                    **(compare(details, frames[name]) if name != "full" else {}),
                }
                for name in names
            }
        ).T.reset_index(names="Config")

    with pd.ExcelWriter(args.out, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        if ablation is not None:
            ablation.to_excel(w, sheet_name="Ablation", index=False)
        group("Language").to_excel(w, sheet_name="ByLanguage", index=False)
        group("Type").to_excel(w, sheet_name="ByType", index=False)
        details.to_excel(w, sheet_name="Details", index=False)
        details[(details["Hit@5"] == 0) & (details["GT_has_no_section"] == 0)].to_excel(
            w, sheet_name="Failures", index=False
        )
    print("\n" + summary.to_string(index=False))
    print("\n" + group("Language").to_string(index=False))
    if ablation is not None:
        cols = ["Config", "Hit@1", "Hit@5", "Test Hit@5", "MRR", "Pool recall",
                "dHit@5 (full - config)", "McNemar p"]
        print("\n" + ablation[[c for c in cols if c in ablation.columns]].to_string(index=False))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
