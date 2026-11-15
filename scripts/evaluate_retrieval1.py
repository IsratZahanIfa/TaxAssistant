"""
retrieval_eval_v2.py - better retrieval for the Income Tax Act + an honest evaluation.

What changed compared with the HyDE-only script, and the paper behind each change:

 1. Section-level CHUNKING with a context header ("Section 76 - <title>")   [LegalBench-RAG,
    Pipitone & Alami 2024: chunking strategy matters most in legal retrieval]
    The old script cut every section at 512 tokens, so text after that point could never be found.
 2. HYBRID retrieval: BM25 (exact words, numbers, "section 76") + dense BGE-M3
    [BGE M3-Embedding, Chen et al. 2024: dense + lexical work best together]
 3. HyDE as in the paper: the query embedding is averaged with the hypothetical document,
    not replaced by it   [HyDE, Gao et al. ACL 2023]
 4. RECIPROCAL RANK FUSION of all rankings (k = 60)   [Cormack, Clarke & Buettcher, SIGIR 2009]
 5. Results are grouped by SECTION before reranking, so the top 5 are 5 different sections
    (the old top 5 could hold the same section several times)
 6. Cross-encoder reranking (bge-reranker-v2-m3) of the best chunks of the top sections
 7. An explicit "section 76" / "ধারা ৭৬" in the question puts that section first
 8. FIXED EVALUATION: the old check counted a hit when ANY number in the retrieved text
    (a year, an amount, "(1)") equalled a number in the ground truth. Now a hit = the retrieved
    section number is one of the sections named in the ground truth. The old (lenient) score is
    still printed next to it, so both can be compared.

<<<<<<< HEAD
Install:  pip install sentence-transformers rank_bm25 numpy torch chromadb
Run:      python legal_retrieval.py "your query"
          python legal_retrieval.py "your query" --root C:\\Users\\Admin\\Desktop\\TaxAssist\\legal_engine\\data_root
          python legal_retrieval.py "your query" --field-map "{\"section\":\"sec_no\",\"source_pdf\":\"file\"}"
          python legal_retrieval.py "your query" --lenient      (citation fields become warnings, not failures)
=======
Next step for a bigger gain (needs training, see the chat): fine-tune BGE-M3 on questions
generated from each section [GPL, Wang et al. NAACL 2022].

Usage
  pip install rank-bm25 sentence-transformers openpyxl
  python retrieval_eval_v2.py                       # full run (HyDE answers are cached)
  python retrieval_eval_v2.py --no-hyde             # without the LLM step
  python retrieval_eval_v2.py --limit 20            # quick test on 20 questions
>>>>>>> cc85289 (Update TaxAssist project files)
"""

import os
import subprocess
import sys
from pathlib import Path


# ---- Use the project's .venv automatically (VS Code's Run button may start another Python) ----
def _reexec_in_project_venv():
    if os.environ.get("TAXASSIST_VENV_REEXEC"):
        return
    here = Path(__file__).resolve()
    rel = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    for folder in (here.parent, *here.parents):
        venv = folder / ".venv"
        if (venv / rel).exists():
            if Path(sys.prefix).resolve() != venv.resolve():
                env = dict(os.environ, TAXASSIST_VENV_REEXEC="1")
                sys.exit(subprocess.call([str(venv / rel), str(here), *sys.argv[1:]], env=env))
            return


_reexec_in_project_venv()


import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ============================================================ CONFIG
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"
# cleaned corpus made by scripts/build_integrated_corpus.py (real section numbers 1-345)
JSON_PATH = Path("data/processed/integrated_corpus/parents.json")
GOOGLE_SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/17ew_umfBKtAeTUEBFtBknNoyQR5KbWRSgceh47DNeYY"
    "/export?format=csv&gid=0"
)
HYDE_CACHE = Path("data/hyde_cache.json")
OUT_XLSX = "retrieval_v2_results.xlsx"

CHUNK_WORDS = (
    220  # words per chunk (about 350-450 tokens, fits the 512-token reranker well)
)
CHUNK_OVERLAP = 40  # words shared by neighbouring chunks
TOP_CHUNKS_PER_LIST = 150  # chunks taken from each ranking before fusion
RRF_K = 60  # constant from Cormack et al. 2009
TOP_SECTIONS_RERANK = 30  # sections sent to the reranker
CHUNKS_PER_SECTION = 3  # best chunks of a section that the reranker reads
TOP_K_FINAL = 5
RERANK_WEIGHT = 0.8  # final = 0.8 * reranker + 0.2 * fusion (both min-max scaled)
MAX_LEN_EMBED = 512
MAX_LEN_RERANK = 512
BATCH_EMBED = 16
BATCH_RERANK = 8

BN_TO_EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
# a section reference: "section 76", "sec. 76A", "s. 76", "u/s 76", "sections 76 and 77", "ধারা ৭৬"
SECTION_REF = re.compile(
    r"(?:\bsections?\b|\bsec\.?|\bs\.|\bu/s\.?|ধারা)\s*((?:\d{1,3}[a-z]{0,2}"
    r"(?:\s*(?:,|and|or|&|ও|এবং)\s*)?)+)",
    re.I,
)


<<<<<<< HEAD
def _find_project_root() -> Path:
    """First parent folder of this file that contains legal_engine/ (works on any drive)."""
    here = Path(__file__).resolve()
    for folder in (here.parent, *here.parents):
        if (folder / "legal_engine").is_dir():
            return folder
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


# ----------------------------------------------------------------------------- config
@dataclass
class Config:
    project_root: Path = PROJECT_ROOT
    data_root: Path = PROJECT_ROOT / "legal_engine" / "data_root"
    chunks_path: Optional[Path] = None     # None -> auto-search data_root, then project_root, then chroma_db
    parents_path: Optional[Path] = None    # None -> auto-search data_root, then project_root
    chroma_path: Optional[Path] = None     # None -> data_root\chroma_db, then search for chroma.sqlite3
    chroma_collection: str = "bd_tax"
    cache_dir: Path = PROJECT_ROOT / "legal_engine" / "data_root" / "cache"
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
=======
# ============================================================ text helpers
def norm(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return str(text).lower().translate(BN_TO_EN)
>>>>>>> cc85289 (Update TaxAssist project files)


def section_id(value):
    """'76', 'Section 76', 'ধারা ৭৬', '76A' -> '76' / '76a'; '' if there is no number."""
    m = re.search(r"\d{1,3}[a-z]{0,2}", norm(value))
    return m.group(0) if m else ""


<<<<<<< HEAD
def load_any(path: Path):
    if path.suffix.lower() in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)
    return json.loads(path.read_text(encoding="utf-8-sig"))
=======
def sections_named(text):
    """All section numbers written with an explicit 'section'/'ধারা' word."""
    found = set()
    for m in SECTION_REF.finditer(norm(text)):
        found.update(re.findall(r"\d{1,3}[a-z]{0,2}", m.group(1)))
    return found
>>>>>>> cc85289 (Update TaxAssist project files)


def gt_sections(text):
    """Sections named in the ground truth; a range 'Section 285–295' counts every section in it."""
    t = norm(text)
    found = sections_named(t)
    for m in re.finditer(
        r"(?:\bsections?\b|\bsec\.?|ধারা)\s*(\d{1,3})\s*[–—-]\s*(\d{1,3})\b", t
    ):
        a, b = int(m.group(1)), int(m.group(2))
        if a < b <= a + 60:
            found.update(str(i) for i in range(a, b + 1))
    return found


def tokenize(text):
    """BM25 tokens: lower case, Bangla digits -> ASCII, words and numbers (Bangla letters kept)."""
    return re.findall(r"[a-z0-9]+|[ঀ-৿]+", norm(text))


def old_lenient_match(doc_text, ground_truth, threshold=0.3):
    """The ORIGINAL check, kept only for comparison: any number in common counts as a hit."""

    def ids(t):
        return {
            s
            for s in re.findall(r"(?:section|sec|ধারা)?\s*(\d+[a-z]?)", norm(t))
            if len(re.sub(r"\D", "", s)) <= 3
        }

    a, b = ids(doc_text), ids(ground_truth)
    if b and a & b:
        return True
    dw, gw = set(norm(doc_text).split()), set(norm(ground_truth).split())
    if not dw or not gw:
        return False
    return len(dw & gw) / len(dw | gw) > threshold


# ============================================================ corpus
def load_sections(path):
    """Reads parents.json (dict id -> record) or the old list of {section, title, text, page}."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list):
        raise TypeError("JSON must be parents.json or a list of {section, title, text, page}.")
    docs = []
    for item in raw:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        sec = str(item.get("section_no", item.get("section", ""))).strip()
        kind = item.get("record_type", "section")
        if kind in ("section", "act_section"):
            sid = section_id(sec) or sec.lower()
        else:  # schedule paragraph: own id, so paragraph "2" is never taken for section 2
            sid = f"sch:{item.get('id', sec)}"
            sec = f"Schedule p{item.get('page_start', '?')} para {sec}"
        docs.append(
            {
                "section": sec,
                "section_id": sid,
                "title": str(item.get("title", "")).strip(),
                "text": text,
                "page": item.get("page_start", item.get("page", "")),
            }
        )
    return docs


def split_units(text):
    """Split a section at its sub-section / clause starts '(1)', '(a)', '(১)', '(ক)' when possible."""
    parts = re.split(
        r"\n(?=\s*\((?:\d{1,3}[a-z]?|[a-z]{1,3}|[০-৯]{1,3}|[ক-হ])\)\s)", text
    )
    return [p.strip() for p in parts if p.strip()]


def make_chunks(docs):
    """Chunks of about CHUNK_WORDS words; sub-sections are kept together when they fit.
    Every chunk starts with its section number and title (context header)."""
    chunks = []
    for d_idx, d in enumerate(docs):
        header = f"Section {d['section']}: {d['title']}".strip()
        buf, pieces = [], []
        for unit in split_units(d["text"]):
            words = unit.split()
            if len(buf) + len(words) <= CHUNK_WORDS:
                buf += words
                continue
            if buf:
                pieces.append(buf)
            while len(words) > CHUNK_WORDS:  # very long sub-section: windows
                pieces.append(words[:CHUNK_WORDS])
                words = words[CHUNK_WORDS - CHUNK_OVERLAP :]
            buf = words
        if buf:
            pieces.append(buf)
        for p_idx, words in enumerate(pieces):
            body = " ".join(words)
            chunks.append({"doc": d_idx, "part": p_idx, "text": f"{header}\n{body}"})
    return chunks


# ============================================================ HyDE (cached)
def call_ollama(prompt):
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 200},
            },
            timeout=120,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        print(f"Ollama error {r.status_code}")
    except Exception as e:  # noqa: BLE001
        print(f"Ollama request failed: {e}")
    return ""


def hyde_document(query, cache):
    if query in cache:
        return cache[query]
    prompt = (
        "You are an expert on the Bangladesh Income Tax Act, 2023. Write a short passage, in the "
        "style of the Act itself, that answers the question. Use the legal terms the Act would use.\n\n"
        f"Question: {query}\n\nPassage:"
    )
    text = call_ollama(prompt)
    if len(text) <= 10:
        text = ""
    cache[query] = text
    return text


# ============================================================ retrieval
def section_ranking(chunk_order, chunks, limit=TOP_CHUNKS_PER_LIST):
    """Chunk ranking -> section ranking (first time a section appears)."""
    seen, out = set(), []
    for c in chunk_order[:limit]:
        d = chunks[c]["doc"]
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


class Retriever:
    def __init__(self, docs, chunks, dense_model, reranker, bm25):
        self.docs, self.chunks = docs, chunks
        self.dense, self.reranker, self.bm25 = dense_model, reranker, bm25
        print(f"Encoding {len(chunks)} chunks ...")
        self.chunk_emb = dense_model.encode(
            [c["text"] for c in chunks],
            batch_size=BATCH_EMBED,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self.by_section_id = {}
        for i, d in enumerate(docs):
            self.by_section_id.setdefault(d["section_id"], i)

    def search(self, query, hypo=""):
        # ---- rankings
        q_emb = self.dense.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        rankings = []
        dense_q = self.chunk_emb @ q_emb
        rankings.append(section_ranking(np.argsort(-dense_q), self.chunks))
        if hypo:
            h_emb = self.dense.encode(
                [hypo], normalize_embeddings=True, convert_to_numpy=True
            )[0]
            mix = q_emb + h_emb  # HyDE: query + hypothetical document
            mix /= np.linalg.norm(mix) + 1e-9
            dense_h = self.chunk_emb @ mix
            rankings.append(section_ranking(np.argsort(-dense_h), self.chunks))
        bm_q = self.bm25.get_scores(tokenize(query))
        rankings.append(section_ranking(np.argsort(-bm_q), self.chunks))
        if hypo:
            bm_h = self.bm25.get_scores(tokenize(hypo))
            rankings.append(section_ranking(np.argsort(-bm_h), self.chunks))

        fused = rrf(rankings)
        # ---- a section named in the question goes first
        for sid in sections_named(query):
            if sid in self.by_section_id:
                fused[self.by_section_id[sid]] = max(fused.values(), default=0) + 1.0
        candidates = sorted(fused, key=fused.get, reverse=True)[:TOP_SECTIONS_RERANK]
        if not candidates:
            return [], []

        # ---- rerank: the best chunks of each candidate section
        pairs, owner = [], []
        best_dense = dense_h if hypo else dense_q
        for d in candidates:
            idx = [i for i, c in enumerate(self.chunks) if c["doc"] == d]
            idx = sorted(idx, key=lambda i: -best_dense[i])[:CHUNKS_PER_SECTION]
            for i in idx:
                pairs.append([query, self.chunks[i]["text"]])
                owner.append(d)
        scores = self.reranker.predict(
            pairs, batch_size=BATCH_RERANK, show_progress_bar=False
        )
        sec_rr = {}
        for d, s in zip(owner, scores):
            sec_rr[d] = max(sec_rr.get(d, -1e9), float(s))
        rr = minmax([sec_rr[d] for d in candidates])
        fu = minmax([fused[d] for d in candidates])
        final = RERANK_WEIGHT * rr + (1 - RERANK_WEIGHT) * fu
        for j, d in enumerate(candidates):  # keep an explicitly named section first
            if self.docs[d]["section_id"] in sections_named(query):
                final[j] += 10
        order = np.argsort(-final)[:TOP_K_FINAL]
        return [candidates[j] for j in order], [float(final[j]) for j in order]


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
<<<<<<< HEAD
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
=======
    ap.add_argument("--json", default=str(JSON_PATH))
    ap.add_argument("--csv", default=GOOGLE_SHEET_CSV_URL)
    ap.add_argument("--no-hyde", action="store_true")
    ap.add_argument(
        "--limit", type=int, default=0, help="evaluate only the first N questions"
    )
    args = ap.parse_args()

    import torch
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder, SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    docs = load_sections(args.json)
    chunks = make_chunks(docs)
    print(f"{len(docs)} sections -> {len(chunks)} chunks  (device: {device})")

    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    dense = SentenceTransformer("BAAI/bge-m3", device=device)
    dense.max_seq_length = MAX_LEN_EMBED
    reranker = CrossEncoder(
        "BAAI/bge-reranker-v2-m3", device=device, max_length=MAX_LEN_RERANK
    )
    retriever = Retriever(docs, chunks, dense, reranker, bm25)
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    df = pd.read_csv(args.csv)
    if "Question" not in df.columns or "Ground Truth Legal Basis" not in df.columns:
        raise ValueError(
            "CSV needs the columns 'Question' and 'Ground Truth Legal Basis'."
        )
    if args.limit:
        df = df.head(args.limit)

    cache = (
        json.loads(HYDE_CACHE.read_text(encoding="utf-8"))
        if HYDE_CACHE.exists()
        else {}
    )
    rows = []
    hits = {"strict": [0, 0, 0], "old": [0, 0, 0]}
    mrr = {"strict": 0.0, "old": 0.0}
    total = no_ref = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        q = str(row["Question"]).strip()
        gt = str(row.get("Ground Truth Legal Basis", "")).strip()
        if not q or not gt or gt.lower() == "nan":
            continue
        total += 1
        hypo = "" if args.no_hyde else hyde_document(q, cache)
        top, scores = retriever.search(q, hypo)
        gt_secs = gt_sections(gt)
        if not gt_secs:
            no_ref += 1  # e.g. "Finance Act 2024": cannot be scored strictly
        strict = [1 if docs[d]["section_id"] in gt_secs else 0 for d in top]
        old = [
            1 if old_lenient_match(f"{docs[d]['section']} {docs[d]['text']}", gt) else 0
            for d in top
        ]
        for name, m in (("strict", strict), ("old", old)):
            if name == "strict" and not gt_secs:
                continue
            for j, k in enumerate((1, 3, 5)):
                hits[name][j] += int(sum(m[:k]) > 0)
            first = next((i + 1 for i, v in enumerate(m) if v), 0)
            mrr[name] += 1.0 / first if first else 0.0
        rows.append(
            {
                "Query": q,
                "Ground_Truth": gt,
                "GT_sections": ", ".join(sorted(gt_secs)),
                "Retrieved": ", ".join(docs[d]["section"] for d in top),
                "Top1_title": docs[top[0]]["title"] if top else "",
                "Hit@1": strict[0] if strict else 0,
                "Hit@5": int(sum(strict) > 0),
                "Old_Hit@5": int(sum(old) > 0),
            }
        )
        if len(cache) % 10 == 0:
            HYDE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            HYDE_CACHE.write_text(
                json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8"
            )
    HYDE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    HYDE_CACHE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    scored = total - no_ref  # strict metrics only over questions that name a section

    def pct(x, n=total):
        return f"{100 * x / n:.2f}%" if n else "0%"

    summary = pd.DataFrame(
        {
            "Metric": [
                "Total queries",
                "Ground truth without a section number",
                "Hit@1 (strict)",
                "Hit@3 (strict)",
                "Hit@5 (strict)",
                "MRR (strict)",
                "Hit@1 (old lenient)",
                "Hit@3 (old lenient)",
                "Hit@5 (old lenient)",
                "MRR (old lenient)",
            ],
            "Value": [
                total,
                no_ref,
                *[pct(h, scored) for h in hits["strict"]],
                f"{mrr['strict'] / max(scored, 1):.4f}",
                *[pct(h) for h in hits["old"]],
                f"{mrr['old'] / max(total, 1):.4f}",
            ],
        }
    )
    details = pd.DataFrame(rows)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        details.to_excel(w, sheet_name="Details", index=False)
        details[details["Hit@5"] == 0].to_excel(w, sheet_name="Failures", index=False)
    print("\n" + summary.to_string(index=False))
    print(
        f"\nSaved {OUT_XLSX}  (sheet 'Failures' = questions whose section was not in the top 5)"
    )


if __name__ == "__main__":
    main()
>>>>>>> cc85289 (Update TaxAssist project files)
