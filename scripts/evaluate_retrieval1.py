"""
legal_retrieval.py  --  Adaptive, tax-year-aware legal retrieval with a corrective evidence gate.

Pipeline
  1. RuleRouter          : language, section/SRO/doc-id, tax year, topic, complexity -> QueryPlan
  2. AdaptiveRetriever   : exact lookup | BM25 | BGE-M3 dense | legal reference graph (only what the plan needs)
  3. TaxYearFilter       : drops provisions not in force on the reference date (only when dates are verified)
  4. Reranker            : bge-reranker-v2-m3 (+ OCR-noise penalty) + attach exceptions / amendments
  5. EvidenceGate        : relevance, source validity, completeness, citation -> accept or retry (escalating)
Output: VerifiedEvidence (sections + clauses + applicable tax year + gate report)

Data source order: chunks*.json / *.pkl (data_root, then project_root)  ->  ChromaDB index (collection bd_tax)

Install:  pip install sentence-transformers rank_bm25 numpy torch chromadb
Run:      python legal_retrieval.py "your query"
          python legal_retrieval.py "your query" --root E:\\TaxAssist\\legal_engine\\data_root
          python legal_retrieval.py "your query" --field-map "{\"section\":\"sec_no\",\"source_pdf\":\"file\"}"
          python legal_retrieval.py "your query" --lenient      (citation fields become warnings, not failures)
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import logging
import os
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("legal_rag")


# ----------------------------------------------------------------------------- config
@dataclass
class Config:
    project_root: Path = Path(r"E:\TaxAssist")
    data_root: Path = Path(r"E:\TaxAssist\legal_engine\data_root")
    chunks_path: Optional[Path] = None     # None -> auto-search data_root, then project_root, then chroma_db
    parents_path: Optional[Path] = None    # None -> auto-search data_root, then project_root
    chroma_path: Optional[Path] = None     # None -> data_root\chroma_db, then search for chroma.sqlite3
    chroma_collection: str = "bd_tax"
    cache_dir: Path = Path(r"E:\TaxAssist\legal_engine\data_root\cache")
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # logical field -> your metadata key, e.g. {"section": "sec_no", "source_pdf": "file"}
    field_map: dict = field(default_factory=dict)
    section_from_text: bool = False        # guess section from the start of the text (unreliable, off by default)
    strict_citation: bool = True           # False: missing law/source fields are warnings, not gate failures

    top_k_each: int = 30          # per retrieval path
    pool: int = 40                # candidates kept after fusion
    final_k: int = 5              # primary passages returned
    max_extra: int = 6            # exceptions / amendments attached
    rrf_k: int = 60

    # OCR noise: share of letters in scripts that should not appear in Bangla/English tax law
    noise_flag: float = 0.005
    noise_penalty: float = 0.15

    # evidence-gate thresholds (sigmoid reranker scores) -- tune on tax_questions_200.json
    min_top_score: float = 0.35
    min_pass_score: float = 0.20
    exception_score: float = 0.20
    max_retries: int = 2


# ----------------------------------------------------------------------------- file loading
SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", "site-packages", ".idea", ".vscode"}


def load_any(path: Path):
    if path.suffix.lower() in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def find_file(roots: list[Path], patterns: list[str]) -> Optional[Path]:
    """Walk each root in order; the largest file matching any pattern wins in the first root that has one."""
    pats = [p.lower() for p in patterns]
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        best: Optional[Path] = None
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in SKIP_DIRS]
            for f in fn:
                if any(fnmatch.fnmatch(f.lower(), p) for p in pats):
                    p = Path(dp) / f
                    if best is None or p.stat().st_size > best.stat().st_size:
                        best = p
        if best is not None:
            return best
    return None


def resolve(path: Optional[Path], roots: list[Path], patterns: list[str]) -> Optional[Path]:
    if path is not None and Path(path).exists():
        return Path(path)
    hit = find_file(roots, patterns)
    if hit:
        log.info("auto-found %s", hit)
    return hit


def as_records(raw) -> list[dict]:
    """Accept a list of dicts, {"chunks": [...]}, or {id: {...}} / {id: "text"}."""
    if isinstance(raw, dict):
        for key in ("chunks", "documents", "data", "records"):
            if isinstance(raw.get(key), list):
                return raw[key]
        out = []
        for k, v in raw.items():
            out.append({"id": k, **v} if isinstance(v, dict) else {"id": k, "text": str(v)})
        return out
    return list(raw)


def find_chroma_dir(cfg: Config) -> Optional[Path]:
    if cfg.chroma_path and Path(cfg.chroma_path).exists():
        return Path(cfg.chroma_path)
    default = Path(cfg.data_root) / "chroma_db"
    if (default / "chroma.sqlite3").exists():
        return default
    hit = find_file([Path(cfg.data_root), Path(cfg.project_root)], ["chroma.sqlite3"])
    return hit.parent if hit else None


def load_from_chroma(cfg: Config):
    """Read chunks (+ their stored BGE-M3 embeddings) straight from the ChromaDB index."""
    db = find_chroma_dir(cfg)
    if db is None:
        return None, None
    try:
        import chromadb
    except ImportError as e:
        raise ImportError("chromadb is not installed: pip install chromadb") from e
    client = chromadb.PersistentClient(path=str(db))
    names = [c.name if hasattr(c, "name") else str(c) for c in client.list_collections()]
    if not names:
        return None, None
    name = cfg.chroma_collection if cfg.chroma_collection in names else names[0]
    data = client.get_collection(name).get(include=["documents", "metadatas", "embeddings"])
    recs = [{**(m or {}), "id": i, "text": d}
            for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])]
    emb = data.get("embeddings")
    emb = np.asarray(emb, dtype="float32") if emb is not None and len(emb) else None
    if emb is not None:
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    log.info("chroma_db at %s, collection %s: %d chunks (stored embeddings: %s)",
             db, name, len(recs), emb is not None)
    return recs, emb


# ----------------------------------------------------------------------------- text utils
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
TOKEN_RE = re.compile(r"[^\s\.,;:!?()\[\]{}\"'“”‘’।|/\\\-–]+")
# scripts that should not appear in Bangla/English tax law -> OCR garbage (Hebrew, Arabic, Devanagari, Tamil,
# Telugu, Hangul, Kana, CJK)
NOISE_RE = re.compile(r"[\u0590-\u05FF\u0600-\u06FF\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F"
                      r"\uAC00-\uD7AF\u3040-\u30FF\u4E00-\u9FFF]")


def norm_text(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").translate(BN_DIGITS)


def tokenize(s: str) -> list[str]:
    # \w splits Bangla at vowel signs, so split on separators instead
    return TOKEN_RE.findall(norm_text(s).lower())


def bangla_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return sum("\u0980" <= c <= "\u09ff" for c in letters) / len(letters) if letters else 0.0


def noise_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return len(NOISE_RE.findall(s)) / len(letters) if letters else 0.0


SEC_LIST_RE = re.compile(
    r"(?:\bsections?\b|\bsec\b\.?|ধারা(?:য়|র)?|\bdhara\b)\s*[:\-]?\s*"
    r"((?:\d{1,3}[A-Za-z]{0,2}(?:\s*(?:,|and|&|ও|এবং)\s*)?)+)", re.I)
SEC_POST_RE = re.compile(r"(\d{1,3}[A-Za-z]{0,2})\s*(?:নং\s*)?ধারা")
SEC_NUM_RE = re.compile(r"\d{1,3}[A-Za-z]{0,2}")
SEC_HEAD_RE = re.compile(r"^\W{0,5}(?:section|sec\.?|ধারা)?\s*[:\-]?\s*(\d{1,3}[A-Za-z]{0,2})\s*[\.\):\-–]", re.I)
SRO_RE = re.compile(r"(?:\bs\.?\s?r\.?\s?o\.?|এস\.?\s?আর\.?\s?ও\.?)\s*(?:no\.?|নং|নম্বর)?\s*[:\-]?\s*(\d{1,4})", re.I)
FY_RE = re.compile(r"(20\d{2})\s*[-–/]\s*(\d{2,4})")
EXC_RE = re.compile(
    r"provided that|notwithstanding|\bexcept\b|\bunless\b|shall not apply|"
    r"তবে শর্ত|ব্যতীত|সত্ত্বেও|প্রযোজ্য হইবে না|প্রযোজ্য হবে না", re.I)
COND_RE = re.compile(r"\b(if|but|unless|except|however|while|whereas|both|jodi|kintu|tobe)\b|যদি|তবে|কিন্তু|এবং|অথবা", re.I)
AMEND_TYPES = {"finance_act", "sro", "ordinance", "amendment"}


def extract_sections(text: str) -> list[str]:
    t = norm_text(text)
    out: list[str] = []
    for m in SEC_LIST_RE.finditer(t):
        out += [x.upper() for x in SEC_NUM_RE.findall(m.group(1))]
    out += [m.group(1).upper() for m in SEC_POST_RE.finditer(t)]
    return list(dict.fromkeys(out))


def parse_date(v) -> Optional[date]:
    try:
        return date.fromisoformat(str(v)[:10]) if v else None
    except ValueError:
        return None


# ----------------------------------------------------------------------------- data
@dataclass
class Chunk:
    cid: str
    text: str
    section: str = ""
    law_name: str = ""
    doc_type: str = ""
    doc_id: str = ""
    parent_id: str = ""
    source_pdf: str = ""
    eff_from: Optional[date] = None
    eff_to: Optional[date] = None
    dates_verified: bool = False
    references: list[str] = field(default_factory=list)
    is_exception: bool = False
    noise: float = 0.0

    @staticmethod
    def from_dict(d: dict, fallback_id: str = "", fmap: Optional[dict] = None,
                  section_from_text: bool = False) -> "Chunk":
        d = {**d, **(d.get("metadata") or {})}
        fmap = fmap or {}

        def pick(name, *aliases, default=""):
            for k in ([fmap[name]] if name in fmap else []) + list(aliases):
                v = d.get(k)
                if v not in (None, "", [], {}):
                    return v
            return default

        eff_from = parse_date(pick("effective_from", "effective_from", "valid_from", "effective_date", default=None))
        eff_to = parse_date(pick("effective_to", "effective_to", "valid_to", "repealed_on", default=None))
        text = str(pick("text", "text", "content", "chunk_text", "page_content", "document"))

        raw_sec = norm_text(str(pick("section", "section", "section_no", "section_number", "section_id", "sec")))
        m = SEC_NUM_RE.search(raw_sec)
        section = m.group(0).upper() if m else ""
        if not section and section_from_text:
            hm = SEC_HEAD_RE.match(norm_text(text).lstrip())
            section = hm.group(1).upper() if hm else ""

        refs = pick("references", "references", default=[])
        if isinstance(refs, str):
            refs = [r for r in re.split(r"[,;\s]+", refs) if r]
        return Chunk(
            cid=str(pick("id", "id", "chunk_id", "uid", default=fallback_id)),
            text=text,
            section=section,
            law_name=str(pick("law_name", "law_name", "law", "act", "act_name")),
            doc_type=str(pick("doc_type", "doc_type", "type", "document_type")).lower(),
            doc_id=str(pick("doc_id", "doc_id", "document_id", "doc_name")),
            parent_id=str(pick("parent_id", "parent_id", "parent")),
            source_pdf=str(pick("source_pdf", "source_pdf", "source", "pdf", "file", "file_name", "filename",
                                "source_file", "pdf_name", "document_name")),
            eff_from=eff_from,
            eff_to=eff_to,
            dates_verified=bool(d.get("dates_verified", bool(eff_from))),
            references=[str(x).upper() for x in refs],
            is_exception=bool(EXC_RE.search(text)),
            noise=noise_ratio(text),
        )


def log_metadata_profile(raw: list[dict]) -> None:
    keys: Counter = Counter()
    for r in raw:
        if not isinstance(r, dict):
            continue
        m = {**r, **(r.get("metadata") or {})}
        for k, v in m.items():
            if k not in ("text", "content", "document", "page_content", "metadata") and v not in (None, "", [], {}):
                keys[k] += 1
    log.info("non-empty metadata keys (key: chunks): %s", dict(keys))


class Corpus:
    def __init__(self, cfg: Config):
        roots = [Path(cfg.data_root), Path(cfg.project_root)]
        self.chroma_emb: Optional[np.ndarray] = None

        cp = resolve(cfg.chunks_path, roots, ["chunks*.json", "*chunks*.pkl", "*chunks*.pickle"])
        if cp is not None:
            raw = as_records(load_any(cp))
            src = str(cp)
        else:
            raw, self.chroma_emb = load_from_chroma(cfg)
            src = "chroma_db"
            if raw is None:
                raise FileNotFoundError(
                    f"No chunks file under {roots} and no chroma_db found. "
                    f"Pass --chunks <path>, --chroma <folder with chroma.sqlite3> or --root <folder>.")
        log_metadata_profile(raw)
        self.chunks: list[Chunk] = [
            Chunk.from_dict(x, fallback_id=f"c{i}", fmap=cfg.field_map, section_from_text=cfg.section_from_text)
            for i, x in enumerate(raw)]
        self.by_id = {c.cid: c for c in self.chunks}
        log.info("loaded %d chunks from %s", len(self.chunks), src)

        if self.chroma_emb is None and cp is not None:
            self._align_chroma_embeddings(cfg)

        self.parents: dict[str, str] = {}
        pp = resolve(cfg.parents_path, roots, ["parents*.json", "*parents*.pkl", "*parents*.pickle"])
        if pp:
            p = load_any(pp)
            if isinstance(p, dict):
                self.parents = {str(k): (v if isinstance(v, str) else v.get("text", "")) for k, v in p.items()}
            else:
                self.parents = {str(x.get("id") or x.get("parent_id")): x.get("text", "") for x in p}
            log.info("loaded %d parents from %s", len(self.parents), pp)
        else:
            log.info("no parents file found: passages will use the chunk text instead of parent text")

        # exact-lookup indexes
        self.by_section: dict[str, list[str]] = defaultdict(list)
        self.by_doc: dict[str, list[str]] = defaultdict(list)
        for c in self.chunks:
            if c.section:
                self.by_section[c.section].append(c.cid)
            if c.doc_id:
                self.by_doc[c.doc_id.lower()].append(c.cid)

        # legal reference graph (section level) + amendment / exception indexes
        self.refs_out: dict[str, set[str]] = defaultdict(set)
        self.refs_in: dict[str, set[str]] = defaultdict(set)
        self.amend_idx: dict[str, list[str]] = defaultdict(list)
        self.exc_idx: dict[str, list[str]] = defaultdict(list)
        for c in self.chunks:
            mentioned = set(extract_sections(c.text)) | set(c.references)
            if c.section:
                for r in mentioned - {c.section}:
                    self.refs_out[c.section].add(r)
                    self.refs_in[r].add(c.section)
                if c.is_exception:
                    self.exc_idx[c.section].append(c.cid)
            if c.doc_type in AMEND_TYPES:
                for r in mentioned:
                    self.amend_idx[r].append(c.cid)

        self._report(cfg)

    def _align_chroma_embeddings(self, cfg: Config) -> None:
        """If chunks came from JSON but ChromaDB holds the same ids, reuse its BGE-M3 vectors (skips re-encoding)."""
        try:
            recs, emb = load_from_chroma(cfg)
        except Exception as e:                                   # chroma optional here
            log.info("chroma embeddings not reused (%s)", e)
            return
        if recs is None or emb is None:
            return
        idx = {r["id"]: i for i, r in enumerate(recs)}
        hit = sum(c.cid in idx for c in self.chunks)
        if hit >= 0.99 * len(self.chunks):
            aligned = np.zeros((len(self.chunks), emb.shape[1]), dtype="float32")
            for j, c in enumerate(self.chunks):
                if c.cid in idx:
                    aligned[j] = emb[idx[c.cid]]
            self.chroma_emb = aligned
            log.info("reusing stored ChromaDB embeddings for %d/%d chunks", hit, len(self.chunks))

    def _report(self, cfg: Config) -> None:
        n = len(self.chunks)
        n_sec = sum(1 for c in self.chunks if c.section)
        n_noisy = sum(1 for c in self.chunks if c.noise > cfg.noise_flag)
        log.info("corpus: %d chunks, %d with section id, %d distinct sections, %d graph edges",
                 n, n_sec, len(self.by_section), sum(len(v) for v in self.refs_out.values()))
        log.info("OCR-noisy chunks (foreign-script share > %.3f): %d of %d", cfg.noise_flag, n_noisy, n)
        log.info("top law_name values: %s", Counter(c.law_name[:50] for c in self.chunks if c.law_name).most_common(4))
        log.info("top section values: %s", Counter(c.section for c in self.chunks if c.section).most_common(6))
        log.info("source_pdf filled: %d | law_name filled: %d | doc_type filled: %d",
                 sum(1 for c in self.chunks if c.source_pdf), sum(1 for c in self.chunks if c.law_name),
                 sum(1 for c in self.chunks if c.doc_type))
        if n and n_sec < 0.3 * n:
            log.warning("fewer than 30%% of chunks have a section id: exact lookup, graph and exception "
                        "attachment will be weak. Use --field-map to point 'section' at the right key.")
        if not any(c.source_pdf for c in self.chunks) or not any(c.law_name for c in self.chunks):
            log.warning("chunks have no source_pdf or law_name: citation checks will fail. Use --field-map, "
                        "or --lenient to make them warnings.")

    def passage_text(self, c: Chunk) -> str:
        return self.parents.get(c.parent_id) or c.text


# ----------------------------------------------------------------------------- 1. router
TOPICS = {
    "rebate": (["rebate", "investment allowance", "রিবেট", "বিনিয়োগ", "sanchay", "সঞ্চয়", "dps", "রেয়াত", "reyat"],
               ["কর রেয়াত", "রেয়াত", "rebate"]),
    "withholding": (["withholding", "tds", "tax deducted", "উৎসে কর", "utshe kor", "utse kor"],
                    ["উৎসে কর কর্তন", "উৎসে কর", "withholding tax"]),
    "exemption": (["exempt", "অব্যাহতি", "obyahoti", "kormukto", "করমুক্ত", "tax free"],
                  ["অব্যাহতি", "করমুক্ত আয়", "exempted income"]),
    "tax_rate": (["tax rate", "slab", "কত কর", "koto kor", "হার"],
                 ["করহার", "কর হার", "rate of tax"]),
    "return_filing": (["return", "রিটার্ন", "due date", "দাখিল", "dakhil", "deadline"],
                      ["রিটার্ন দাখিল", "আয়কর রিটার্ন", "return of income"]),
    "penalty": (["penalty", "fine", "জরিমানা", "jorimana"], ["জরিমানা", "দণ্ড", "penalty"]),
    "salary": (["salary", "বেতন", "beton", "wage"], ["বেতন", "বেতন খাতে আয়", "income from salaries"]),
    "capital_gains": (["capital gain", "মূলধনী", "mulodhoni"], ["মূলধনী আয়", "capital gains"]),
    "minimum_tax": (["minimum tax", "ন্যূনতম কর", "nunotom kor"], ["ন্যূনতম কর", "minimum tax"]),
    "company": (["company", "কোম্পানি"], ["কোম্পানি", "কোম্পানির কর হার"]),
    "house_property": (["house property", "বাড়ি ভাড়া", "bari bhara", "rental income"],
                       ["বাড়ি ভাড়া", "income from house property"]),
}
BANGLISH = {"kor", "ayokor", "koto", "kivabe", "kibhabe", "hobe", "ache", "amar", "dhara", "taka", "jonno",
            "chhar", "chhara", "kormukto", "bebosha", "beton", "bari", "bhara", "dakhil", "jorimana",
            "utse", "utshe", "kothay", "kemon", "lagbe", "dite"}


@dataclass
class QueryPlan:
    raw: str
    text: str
    language: str                       # bn | en | banglish
    sections: list[str]
    sro_ids: list[str]
    doc_ids: list[str]
    tax_year: Optional[tuple[int, int]]  # fiscal/income year as (start_year, end_year)
    ref_date: Optional[date]             # date the law must be in force
    topics: list[str]
    expansion: list[str]
    complexity: str                      # simple | medium | complex
    use_exact: bool
    use_bm25: bool
    use_dense: bool
    use_graph: bool


class RuleRouter:
    def __init__(self, known_doc_ids: set[str]):
        self.doc_ids = {d for d in known_doc_ids if len(d) >= 4}

    def route(self, query: str) -> QueryPlan:
        t = norm_text(query)
        low = t.lower()
        toks = tokenize(t)

        lang = "bn" if bangla_ratio(t) > 0.3 else ("banglish" if BANGLISH & set(toks) else "en")
        sections = extract_sections(t)
        sro = [f"SRO-{m}" for m in SRO_RE.findall(t)]
        docs = [d for d in self.doc_ids if d.lower() in low]

        tax_year = ref = None
        m = FY_RE.search(t)
        if m:
            y = int(m.group(1))
            assessment = bool(re.search(r"assessment|করবর্ষ|কর বর্ষ", low))
            y = y - 1 if assessment else y            # assessment year 2024-25 -> income year 2023-24
            tax_year, ref = (y, y + 1), date(y + 1, 6, 30)

        topics, expansion = [], []
        for name, (triggers, terms) in TOPICS.items():
            if any(tr in low for tr in triggers):
                topics.append(name)
                expansion += terms

        n_refs = len(sections) + len(sro) + len(docs)
        signals = len(COND_RE.findall(t)) + len(topics) + len(sections)
        words = len(toks)
        if n_refs and words <= 12 and not COND_RE.search(t) and not topics:
            cx = "simple"
        elif signals >= 3 or words > 30:
            cx = "complex"
        else:
            cx = "medium"

        return QueryPlan(
            raw=query, text=t, language=lang, sections=sections, sro_ids=sro, doc_ids=docs,
            tax_year=tax_year, ref_date=ref, topics=topics, expansion=list(dict.fromkeys(expansion)),
            complexity=cx, use_exact=n_refs > 0,
            use_bm25=cx != "simple", use_dense=cx != "simple",
            use_graph=cx == "complex",
        )


# ----------------------------------------------------------------------------- 2. retrieval
class AdaptiveRetriever:
    def __init__(self, corpus: Corpus, cfg: Config):
        self.c, self.cfg = corpus, cfg
        self._bm25 = self._dense = self._embedder = None

    # --- lazy indexes
    def _get_bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([tokenize(ch.text) for ch in self.c.chunks])
        return self._bm25

    def _get_embedder(self):
        if self._embedder is None:
            import torch
            from sentence_transformers import SentenceTransformer
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._embedder = SentenceTransformer(self.cfg.embed_model, device=dev)
            if dev == "cuda":
                self._embedder.half()
        return self._embedder

    def _get_dense(self):
        # reuse the embeddings already stored in ChromaDB when available
        if self._dense is None and self.c.chroma_emb is not None \
                and len(self.c.chroma_emb) == len(self.c.chunks):
            self._dense = self.c.chroma_emb
        if self._dense is None:
            self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
            h = hashlib.md5("|".join(ch.cid for ch in self.c.chunks).encode()).hexdigest()[:10]
            f = self.cfg.cache_dir / f"bge_m3_{len(self.c.chunks)}_{h}.npy"
            if f.exists():
                self._dense = np.load(f)
            else:
                self._dense = self._get_embedder().encode(
                    [ch.text for ch in self.c.chunks], batch_size=16, normalize_embeddings=True,
                    show_progress_bar=True).astype("float32")
                np.save(f, self._dense)
        return self._dense

    # --- paths
    def exact(self, plan: QueryPlan, extra_sections=()) -> list[tuple[str, float]]:
        ids: list[str] = []
        for s in list(plan.sections) + list(extra_sections):
            ids += self.c.by_section.get(s, [])
        for d in plan.doc_ids:
            ids += self.c.by_doc.get(d.lower(), [])
        for s in plan.sro_ids:
            ids += self.c.by_doc.get(s.lower(), [])
        return [(i, 1.0) for i in dict.fromkeys(ids)]

    def bm25(self, q: str, k: int) -> list[tuple[str, float]]:
        scores = self._get_bm25().get_scores(tokenize(q))
        idx = np.argsort(-scores)[:k]
        return [(self.c.chunks[i].cid, float(scores[i])) for i in idx if scores[i] > 0]

    def dense(self, q: str, k: int) -> list[tuple[str, float]]:
        qv = self._get_embedder().encode([q], normalize_embeddings=True).astype("float32")[0]
        sims = self._get_dense() @ qv
        idx = np.argsort(-sims)[:k]
        return [(self.c.chunks[i].cid, float(sims[i])) for i in idx]

    def graph(self, seeds: list[str], max_nodes: int = 15) -> list[tuple[str, float]]:
        out: dict[str, float] = {}
        for cid in seeds[:8]:
            sec = self.c.by_id[cid].section
            if not sec:
                continue
            neigh = self.c.refs_out.get(sec, set()) | self.c.refs_in.get(sec, set())
            for n in neigh:
                for nid in self.c.by_section.get(n, [])[:3]:
                    out[nid] = max(out.get(nid, 0), 0.5)
            for nid in self.c.amend_idx.get(sec, []) + self.c.exc_idx.get(sec, []):
                out[nid] = max(out.get(nid, 0), 0.6)
        out = {k: v for k, v in out.items() if k not in seeds}
        return sorted(out.items(), key=lambda x: -x[1])[:max_nodes]

    # --- orchestration
    def retrieve(self, plan: QueryPlan, strat: dict) -> tuple[list[str], dict]:
        k = strat["top_k"]
        terms = plan.expansion + strat.get("extra_terms", [])
        q = plan.text + (" " + " ".join(terms) if terms else "")
        rankings: dict[str, list[tuple[str, float]]] = {}
        if strat["exact"]:
            rankings["exact"] = self.exact(plan, strat.get("force_sections", ()))
        if strat["bm25"]:
            rankings["bm25"] = self.bm25(q, k)
        if strat["dense"]:
            rankings["dense"] = self.dense(q, k)
        weights = {"exact": 2.0, "bm25": 1.0, "dense": 1.0, "graph": 0.7}
        fused: dict[str, float] = defaultdict(float)

        def fuse():
            fused.clear()
            for path, lst in rankings.items():
                for r, (cid, _) in enumerate(lst):
                    fused[cid] += weights[path] / (self.cfg.rrf_k + r + 1)

        fuse()
        if strat["graph"]:
            seeds = [c for c, _ in sorted(fused.items(), key=lambda x: -x[1])[:8]]
            rankings["graph"] = self.graph(seeds)
            fuse()
        ordered = [c for c, _ in sorted(fused.items(), key=lambda x: -x[1])]
        return ordered[: strat["pool"]], {p: len(v) for p, v in rankings.items()}


# ----------------------------------------------------------------------------- 3. tax-year filter
def validity(ch: Chunk, ref: Optional[date]) -> str:
    """'na' (no year asked) | 'in_force' | 'unknown' (dates not verified) | 'out_of_force'"""
    if ref is None:
        return "na"
    if not ch.dates_verified:
        return "unknown"
    if (ch.eff_from and ch.eff_from > ref) or (ch.eff_to and ch.eff_to < ref):
        return "out_of_force"
    return "in_force"


def tax_year_filter(corpus: Corpus, ids: list[str], plan: QueryPlan, relax: bool) -> list[str]:
    if plan.ref_date is None or relax:
        return ids
    return [i for i in ids if validity(corpus.by_id[i], plan.ref_date) != "out_of_force"]


# ----------------------------------------------------------------------------- 4. rerank
@dataclass
class Passage:
    chunk_id: str
    text: str
    section: str
    law_name: str
    doc_type: str
    source_pdf: str
    effective_from: Optional[str]
    effective_to: Optional[str]
    validity: str
    score: float
    role: str = "primary"            # primary | exception | amendment
    ocr_noisy: bool = False


class Reranker:
    def __init__(self, corpus: Corpus, cfg: Config):
        self.c, self.cfg, self._m = corpus, cfg, None

    def _model(self):
        if self._m is None:
            import torch
            from sentence_transformers import CrossEncoder
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._m = CrossEncoder(self.cfg.rerank_model, max_length=512, device=dev)
        return self._m

    def score(self, query: str, ids: list[str]) -> dict[str, float]:
        if not ids:
            return {}
        pairs = [(query, self.c.by_id[i].text[:2000]) for i in ids]
        raw = np.asarray(self._model().predict(pairs, batch_size=16), dtype="float64")
        return dict(zip(ids, (1 / (1 + np.exp(-raw))).tolist()))

    def _passage(self, cid: str, score: float, role: str, ref: Optional[date]) -> Passage:
        ch = self.c.by_id[cid]
        return Passage(cid, self.c.passage_text(ch), ch.section, ch.law_name, ch.doc_type, ch.source_pdf,
                       str(ch.eff_from) if ch.eff_from else None, str(ch.eff_to) if ch.eff_to else None,
                       validity(ch, ref), round(score, 4), role, ch.noise > self.cfg.noise_flag)

    def _penalty(self, cid: str) -> float:
        return self.cfg.noise_penalty if self.c.by_id[cid].noise > self.cfg.noise_flag else 0.0

    def select(self, plan: QueryPlan, ids: list[str], relax: bool) -> list[Passage]:
        scores = self.score(plan.text, ids)
        # newest law wins ties: small recency bonus among candidates with known effective dates
        dated = sorted({self.c.by_id[i].eff_from for i in ids if self.c.by_id[i].eff_from})
        bonus = {d: 0.05 * (n + 1) / len(dated) for n, d in enumerate(dated)}
        ranked = sorted(ids, key=lambda i: -(scores[i] - self._penalty(i) + bonus.get(self.c.by_id[i].eff_from, 0)))

        chosen, seen_parents = [], set()
        for cid in ranked:
            ch = self.c.by_id[cid]
            key = ch.parent_id or cid
            if key in seen_parents:
                continue
            if scores[cid] < self.cfg.min_pass_score and chosen:
                break
            chosen.append(cid)
            seen_parents.add(key)
            if len(chosen) >= self.cfg.final_k:
                break
        out = [self._passage(c, scores[c], "primary", plan.ref_date) for c in chosen]

        # attach necessary exceptions / amendments of the selected sections (only if the section id is known)
        extra: list[str] = []
        for p in out:
            if p.section:
                extra += self.c.exc_idx.get(p.section, []) + self.c.amend_idx.get(p.section, [])
        extra = tax_year_filter(self.c, [e for e in dict.fromkeys(extra) if e not in chosen], plan, relax)
        extra = [e for e in extra if (self.c.by_id[e].parent_id or e) not in seen_parents][: self.cfg.max_extra * 2]
        for cid, s in sorted(self.score(plan.text, extra).items(), key=lambda x: -(x[1] - self._penalty(x[0]))):
            if s >= self.cfg.exception_score and len(out) < self.cfg.final_k + self.cfg.max_extra:
                ch = self.c.by_id[cid]
                role = "amendment" if ch.doc_type in AMEND_TYPES else "exception"
                out.append(self._passage(cid, s, role, plan.ref_date))
        return out


# ----------------------------------------------------------------------------- 5. evidence gate
@dataclass
class GateReport:
    passed: bool
    checks: dict
    reasons: list[str]
    missing_sections: list[str]
    quality: float


class EvidenceGate:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def check(self, plan: QueryPlan, passages: list[Passage]) -> GateReport:
        reasons: list[str] = []
        primary = [p for p in passages if p.role == "primary"]

        top = max((p.score for p in primary), default=0.0)
        relevance = top >= self.cfg.min_top_score
        if not relevance:
            reasons.append(f"low relevance (top score {top:.2f} < {self.cfg.min_top_score})")

        source_ok = bool(passages) and all(p.law_name and p.source_pdf for p in passages)
        citation = bool(passages) and all(p.section and p.law_name and p.source_pdf for p in passages)
        strict = self.cfg.strict_citation
        if passages and not source_ok:
            reasons.append(("" if strict else "(non-blocking) ") + "some passages lack law name or source PDF")
        if passages and not citation:
            reasons.append(("" if strict else "(non-blocking) ") + "missing section / law / source for citation")
        if not strict:
            source_ok = citation = bool(passages)

        if plan.ref_date and passages and all(p.validity == "out_of_force" for p in passages):
            source_ok = False
            reasons.append("all passages out of force for the requested tax year")

        have = {p.section for p in passages}
        missing = [s for s in plan.sections if s not in have]
        complete = not missing
        if missing:
            reasons.append(f"requested sections not found: {missing}")

        checks = dict(relevance=relevance, source_validity=source_ok, completeness=complete, citation=citation)
        return GateReport(all(checks.values()), checks, reasons, missing, sum(checks.values()) + top)


# ----------------------------------------------------------------------------- pipeline
@dataclass
class VerifiedEvidence:
    status: str                              # verified | insufficient
    query: str
    plan: dict
    applicable_tax_year: Optional[str]
    passages: list[Passage]
    gate: GateReport
    attempts: list[dict]
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class LegalRetrievalPipeline:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.corpus = Corpus(self.cfg)
        self.router = RuleRouter({c.doc_id for c in self.corpus.chunks if c.doc_id})
        self.retriever = AdaptiveRetriever(self.corpus, self.cfg)
        self.reranker = Reranker(self.corpus, self.cfg)
        self.gate = EvidenceGate(self.cfg)

    def _strategy(self, plan: QueryPlan, attempt: int, prev: Optional[GateReport]) -> dict:
        s = dict(exact=plan.use_exact, bm25=plan.use_bm25, dense=plan.use_dense, graph=plan.use_graph,
                 top_k=self.cfg.top_k_each, pool=self.cfg.pool, relax_year=False,
                 force_sections=(), extra_terms=[])
        if attempt >= 1:   # widen: every path on, bigger pool, force missing sections
            s.update(bm25=True, dense=True, graph=True, top_k=self.cfg.top_k_each * 2,
                     pool=self.cfg.pool * 2, force_sections=tuple(prev.missing_sections) if prev else ())
        if attempt >= 2:   # last resort: relax year filter (flagged) and expand with topic terms
            s.update(relax_year=True, extra_terms=[t for tp in plan.topics for t in TOPICS[tp][0]])
        return s

    def run(self, query: str) -> VerifiedEvidence:
        plan = self.router.route(query)
        attempts, best, prev = [], None, None
        warnings: list[str] = []

        for attempt in range(self.cfg.max_retries + 1):
            strat = self._strategy(plan, attempt, prev)
            ids, trace = self.retriever.retrieve(plan, strat)
            ids = tax_year_filter(self.corpus, ids, plan, strat["relax_year"])
            passages = self.reranker.select(plan, ids, strat["relax_year"]) if ids else []
            report = self.gate.check(plan, passages)
            attempts.append(dict(attempt=attempt, paths=trace, candidates=len(ids),
                                 relax_year=strat["relax_year"], passed=report.passed, reasons=report.reasons))
            log.info("attempt %d: %s %s", attempt, trace, report.reasons)
            if best is None or report.quality > best[1].quality:
                best = (passages, report, strat["relax_year"])
            if report.passed:
                break
            prev = report

        passages, report, relaxed = best
        if relaxed:
            warnings.append("Tax-year filter was relaxed to find evidence; verify the law was in force.")
        if any(p.validity == "unknown" for p in passages):
            warnings.append("Some passages have unverified effective dates.")
        if any(p.ocr_noisy for p in passages):
            warnings.append("Some passages contain OCR noise (foreign-script characters); check the source PDF.")
        if passages and not all(p.source_pdf for p in passages):
            warnings.append("Some passages have no source PDF in their metadata; they cannot be cited yet.")
        year = f"{plan.tax_year[0]}-{str(plan.tax_year[1])[-2:]}" if plan.tax_year else None
        return VerifiedEvidence(
            status="verified" if report.passed else "insufficient", query=query,
            plan={k: (str(v) if isinstance(v, date) else v) for k, v in asdict(plan).items()},
            applicable_tax_year=year, passages=passages, gate=report, attempts=attempts, warnings=warnings)


# ----------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")       # Bangla output in PowerShell
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="ধারা ৪৪ অনুযায়ী ২০২৩-২৪ করবর্ষে কর রেয়াত কত?")
    ap.add_argument("--chunks")
    ap.add_argument("--parents")
    ap.add_argument("--chroma")
    ap.add_argument("--root")
    ap.add_argument("--field-map", help='JSON, e.g. {"section":"sec_no","source_pdf":"file"}')
    ap.add_argument("--section-from-text", action="store_true")
    ap.add_argument("--lenient", action="store_true", help="missing law/source fields do not fail the gate")
    a = ap.parse_args()
    cfg = Config()
    if a.root:
        cfg.data_root = Path(a.root)
        cfg.cache_dir = Path(a.root) / "cache"
    if a.chunks:
        cfg.chunks_path = Path(a.chunks)
    if a.parents:
        cfg.parents_path = Path(a.parents)
    if a.chroma:
        cfg.chroma_path = Path(a.chroma)
    if a.field_map:
        cfg.field_map = json.loads(a.field_map)
    cfg.section_from_text = a.section_from_text
    cfg.strict_citation = not a.lenient
    res = LegalRetrievalPipeline(cfg).run(a.query)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2, default=str))