<<<<<<< HEAD
import os, sys, subprocess
=======
"""
TaxAssist: Hybrid HyDE Retrieval Evaluation
Pipeline:
  1. BGE-M3 dense retrieval using original query and HyDE hypothetical document
  2. BM25 lexical retrieval using the original query
  3. Weighted Reciprocal Rank Fusion (RRF)
  4. Exact legal-section prioritization when a section is explicitly mentioned
  5. BGE-reranker-v2-m3 cross-encoder reranking

Run from E:\\TaxAssist:
    python -u scripts\\evaluate_hybrid_hyde.py

Required packages:
    pip install torch sentence-transformers rank-bm25 pandas openpyxl requests tqdm numpy
Ollama:
    ollama serve
    ollama pull gemma3:12b
"""

import gc
import json
import re
import time
import warnings
>>>>>>> cc85289 (Update TaxAssist project files)
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

import re
import numpy as np
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
<<<<<<< HEAD
=======
from tqdm import tqdm
>>>>>>> cc85289 (Update TaxAssist project files)

warnings.filterwarnings("ignore", category=FutureWarning)

<<<<<<< HEAD
# =====================================================================
# DATA MODELS & SCHEMAS
# =====================================================================

class LegalDocument(BaseModel):
    chunk_id: str
    doc_type: str  # Act, SRO, Paripatra, Finance_Act, Rules
    doc_title: str
    section: str
    subsection: Optional[str] = None
    text: str
    effective_year: str  # e.g., "2023", "2026"
    effective_date: str  # YYYY-MM-DD
    expiry_date: Optional[str] = None  # YYYY-MM-DD
    related_sections: List[str] = Field(default_factory=list)  # For Legal Graph expansion


class QueryIntent(BaseModel):
    raw_query: str
    normalized_query: str
    detected_section: Optional[str] = None
    detected_sro: Optional[str] = None
    detected_year: str = "2026"  # Default to current Assessment/Tax Year
    legal_topic: str = "General Tax"
    complexity: str = "simple"  # simple, moderate, complex
    required_routes: List[str] = Field(default_factory=list)  # ["exact", "bm25", "bge_m3", "graph"]


class RerankedPassage(BaseModel):
    doc: LegalDocument
    rerank_score: float
    retrieval_source: str


class GateVerificationResult(BaseModel):
    status: str  # "SUFFICIENT" or "INSUFFICIENT"
    score: float
    verified_passages: List[RerankedPassage]
    missing_elements: List[str]
    suggested_reformulated_query: Optional[str] = None


# =====================================================================
# STAGE 1: RULE-BASED QUERY ROUTER
# =====================================================================

class RuleBasedQueryRouter:
    """Normalizes input (Bangla/English/Banglish) and determines retrieval routes."""

    BN_DIGITS_TRANS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

    def __init__(self, current_default_year: str = "2026"):
        self.default_year = current_default_year

    def normalize_text(self, text: str) -> str:
        """Lowercases text and converts Bengali digits to English digits."""
        text = text.lower().translate(self.BN_DIGITS_TRANS)
        # Banglish common term replacements
        replacements = {
            "kor": "tax", "dhara": "section", "sro": "sro", 
            "income tax act": "income tax act", "rabate": "rebate"
        }
        for orig, sub in replacements.items():
            text = re.sub(rf'\b{orig}\b', sub, text)
        return text.strip()

    def route_query(self, query: str) -> QueryIntent:
        norm_q = self.normalize_text(query)

        # Detect specific section or SRO references
        sec_match = re.search(r'(?:section|sec|ধারা)\s*(\d+[a-z]?)', norm_q)
        sro_match = re.search(r'(?:sro|এসআরঅ)\s*(\d+)', norm_q)
        year_match = re.search(r'\b(20[1-2][0-9])\b', norm_q)

        section = sec_match.group(1) if sec_match else None
        sro = sro_match.group(1) if sro_match else None
        year = year_match.group(1) if year_match else self.default_year

        # Determine complexity & legal topics
        topic = "General Tax"
        if "rebate" in norm_q or "investment" in norm_q or "rebat" in norm_q:
            topic = "Tax Rebate & Investment"
        elif "salary" in norm_q or "employment" in norm_q:
            topic = "Employment Income"

        complexity = "simple"
        if section or sro:
            complexity = "simple"
        elif len(norm_q.split()) > 8 or "and" in norm_q or "ebong" in norm_q:
            complexity = "complex"

        # Adaptive Routing decision
        routes = []
        if section or sro:
            routes.append("exact")
            routes.append("graph")  # Pull related legal clauses for context
        if complexity in ["simple", "moderate"]:
            routes.append("bm25")
        if complexity in ["moderate", "complex"] or not (section or sro):
            routes.append("bge_m3")

        return QueryIntent(
            raw_query=query,
            normalized_query=norm_q,
            detected_section=section,
            detected_sro=sro,
            detected_year=year,
            legal_topic=topic,
            complexity=complexity,
            required_routes=routes
        )


# =====================================================================
# STAGE 2: ADAPTIVE RETRIEVAL ENGINE & LEGAL GRAPH
# =====================================================================

class AdaptiveRetrievalEngine:
    """Executes only the required retrieval paths based on Query Router instructions."""

    def __init__(self, corpus: List[LegalDocument], dense_model_name: str = "BAAI/bge-m3"):
        self.corpus = corpus
        self.doc_map = {doc.chunk_id: doc for doc in corpus}
        
        # 1. Exact Lookup Maps
        self.section_map: Dict[str, List[LegalDocument]] = {}
        for doc in corpus:
            sec_key = doc.section.lower()
            self.section_map.setdefault(sec_key, []).append(doc)

        # 2. BM25 Sparse Index
        tokenized_corpus = [re.findall(r'\w+', doc.text.lower()) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 3. Dense BGE-M3 Model (Lazy Initialization)
        self.dense_model_name = dense_model_name
        self._dense_model = None
        self.dense_embeddings = None

    @property
    def dense_model(self):
        if self._dense_model is None:
            self._dense_model = SentenceTransformer(self.dense_model_name)
            texts = [f"Section: {d.section}. {d.text}" for d in self.corpus]
            self.dense_embeddings = self._dense_model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True
            )
        return self._dense_model

    def execute_retrieval(self, intent: QueryIntent) -> Dict[str, List[LegalDocument]]:
        retrieved_results: Dict[str, List[LegalDocument]] = {}

        # 1. Exact Lookup
        if "exact" in intent.required_routes and intent.detected_section:
            sec_docs = self.section_map.get(intent.detected_section.lower(), [])
            if sec_docs:
                retrieved_results["exact"] = sec_docs

        # 2. BM25 Keyword Retrieval
        if "bm25" in intent.required_routes:
            q_tokens = re.findall(r'\w+', intent.normalized_query)
            scores = self.bm25.get_scores(q_tokens)
            top_indices = np.argsort(-scores)[:5]
            retrieved_results["bm25"] = [self.corpus[i] for i in top_indices if scores[i] > 0]

        # 3. BGE-M3 Dense Semantic Retrieval
        if "bge_m3" in intent.required_routes:
            q_emb = self.dense_model.encode([intent.normalized_query], normalize_embeddings=True, convert_to_numpy=True)
            sim_scores = np.dot(self.dense_embeddings, q_emb.T).squeeze()
            top_indices = np.argsort(-sim_scores)[:5]
            retrieved_results["bge_m3"] = [self.corpus[i] for i in top_indices]

        # 4. Legal Reference Graph Expansion
        if "graph" in intent.required_routes:
            graph_docs = []
            seed_docs = retrieved_results.get("exact", [])
            for doc in seed_docs:
                for rel_sec in doc.related_sections:
                    graph_docs.extend(self.section_map.get(rel_sec.lower(), []))
            if graph_docs:
                retrieved_results["graph"] = list({d.chunk_id: d for d in graph_docs}.values())

        return retrieved_results


# =====================================================================
# STAGE 3 & 4: TAX-YEAR FILTERING & PRECISION RERANKING
# =====================================================================
=======
# ---------------------------------------------------------------------
# PATHS: relative to the TaxAssist project root
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
JSON_PATH = BASE_DIR / "data" / "Knowledge_Dataset" / "income_tax_act_2023.json"
GOOGLE_SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "17ew_umfBKtAeTUEBFtBknNoyQR5KbWRSgceh47DNeYY/export?format=csv&gid=0"
)
OUTPUT_DIR = BASE_DIR / "outputs" / "retrieval_evaluation_hybrid_hyde"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# EXISTING MODELS (kept from your uploaded code)
# ---------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"

# ---------------------------------------------------------------------
# RETRIEVAL SETTINGS
# ---------------------------------------------------------------------
MAX_LENGTH = 512
DOC_BATCH_SIZE = 16
RERANK_BATCH_SIZE = 8
DENSE_TOP_K = 140
BM25_TOP_K = 140
HYDE_DENSE_TOP_K = 100
RERANK_POOL_K = 100
FINAL_TOP_K = 10
RRF_K = 60

# Weighted RRF: original-query dense and BM25 are primary; HyDE is auxiliary.
WEIGHT_DENSE_QUERY = 1.0
WEIGHT_BM25_QUERY = 1.0
WEIGHT_DENSE_HYDE = 0.65

# Exact section references are a useful legal signal, but are not used as
# ground truth unless the reference is explicitly marked as a section.
EXACT_SECTION_RRF_BONUS = 0.025

# Evaluation fallback for ground truths that contain no explicit section ID.
TITLE_RECALL_THRESHOLD = 0.60
TITLE_PRECISION_THRESHOLD = 0.20

BN_TO_EN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def normalize_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).lower().translate(BN_TO_EN_DIGITS).strip()


def tokenize(text: str) -> List[str]:
    """Unicode-aware tokenizer suitable for English, Bangla and Banglish."""
    text = normalize_text(text)
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


def extract_explicit_sections(text: str) -> set:
    """
    Extract section numbers only when a legal section marker is present.
    Avoids treating every number (years, money, rates) as a section number.
    Supports: Section 82, Sec. 82, ধারা ৮২, ৮২ নং ধারা.
    """
    text = normalize_text(text)
    patterns = [
        r"\bsections?\s*\.?\s*(\d{1,3}[a-z]?(?:\(\d+\))?)",
        r"\bsecs?\.?\s*(\d{1,3}[a-z]?(?:\(\d+\))?)",
        r"ধারা\s*[-:#]?\s*(\d{1,3}[a-z]?(?:\(\d+\))?)",
        r"(\d{1,3}[a-z]?)\s*নং\s*ধারা",
    ]
    found = set()
    for pattern in patterns:
        for value in re.findall(pattern, text, flags=re.IGNORECASE):
            value = value.lower()
            found.add(value)
    return found


def section_base(value: str) -> str:
    return re.match(r"\d+[a-z]?", value.lower()).group(0)


def section_matches(doc: Dict, gt_sections: set) -> bool:
    if not gt_sections:
        return False
    doc_section = normalize_text(doc.get("section", ""))
    doc_sections = extract_explicit_sections(doc_section)
    # JSON section fields commonly contain just the number, without "Section".
    if doc_section and re.fullmatch(r"\d{1,3}[a-z]?(?:\(\d+\))?", doc_section):
        doc_sections.add(doc_section.replace("(", "").replace(")", ""))
    doc_bases = {section_base(x) for x in doc_sections}
    gt_bases = {section_base(x) for x in gt_sections}
    return bool(doc_bases.intersection(gt_bases))


def ground_truth_match(doc: Dict, ground_truth: str) -> bool:
    """
    Primary evaluation: explicit legal section match.
    Fallback: title/evidence token overlap for GT labels without section IDs.
    Does not compare the full legal passage against a short GT string using
    a loose Jaccard threshold.
    """
    gt_sections = extract_explicit_sections(ground_truth)
    if gt_sections:
        return section_matches(doc, gt_sections)

    gt_tokens = set(tokenize(ground_truth))
    if not gt_tokens:
        return False

    title_tokens = set(tokenize(doc.get("title", "")))
    section_tokens = set(tokenize(doc.get("section", "")))
    evidence_tokens = set(tokenize(doc.get("text", "")))
    # Prefer title/section evidence; use passage text only as supporting evidence.
    focused_tokens = title_tokens | section_tokens
    if focused_tokens:
        overlap = len(gt_tokens & focused_tokens)
        recall = overlap / len(gt_tokens)
        precision = overlap / len(focused_tokens) if focused_tokens else 0.0
        if recall >= TITLE_RECALL_THRESHOLD and precision >= TITLE_PRECISION_THRESHOLD:
            return True

    overlap = len(gt_tokens & evidence_tokens)
    recall = overlap / len(gt_tokens)
    return recall >= 0.75


def call_ollama(prompt: str, timeout: int = 90) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 160},
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.RequestException as exc:
        print(f"[WARN] Ollama unavailable; using original query only. Details: {exc}")
        return ""


def generate_hypothetical_doc(query: str) -> str:
    prompt = (
        "You are preparing a retrieval aid for Bangladesh income-tax legislation. "
        "Write a short, neutral hypothetical passage containing likely legal "
        "terminology relevant to the question. Do not invent section numbers, "
        "rates, thresholds, deadlines, or legal conclusions. If uncertain, use "
        "general terminology only. Preserve Bangla terms if the question is Bangla.\n\n"
        f"Question: {query}\n\nHypothetical retrieval passage:"
    )
    return call_ollama(prompt)


def load_documents() -> List[Dict]:
    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"Legal JSON not found: {JSON_PATH}\n"
            "Check the path and run the script from the TaxAssist project."
        )
    with JSON_PATH.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    # Support list JSON and common wrapped JSON formats.
    if isinstance(raw_data, dict):
        for key in ("records", "data", "documents", "sections"):
            if isinstance(raw_data.get(key), list):
                raw_data = raw_data[key]
                break
    if not isinstance(raw_data, list):
        raise TypeError("Legal JSON must be a list or contain a list under records/data/documents/sections.")

    documents = []
    for idx, item in enumerate(raw_data):
        if not isinstance(item, dict):
            continue
        section = str(item.get("section", item.get("section_number", ""))).strip()
        title = str(item.get("title", item.get("section_title", ""))).strip()
        text = str(item.get("text", item.get("content", item.get("body", "")))).strip()
        if not text or text.lower() == "nan":
            continue
        page = item.get("page", item.get("page_number", ""))
        documents.append({
            "doc_id": str(item.get("id", f"legal_{idx}")),
            "section": section,
            "title": title,
            "text": text,
            "page": page,
            "source": item.get("source", "Income Tax Act 2023"),
        })
    if not documents:
        raise ValueError("No usable legal passages found in the JSON.")
    return documents


def load_questions() -> pd.DataFrame:
    try:
        df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    except Exception as exc:
        raise RuntimeError(f"Could not load evaluation Google Sheet CSV: {exc}") from exc

    # Normalize headers to tolerate harmless spaces/case differences.
    header_map = {str(c).strip().lower(): c for c in df.columns}
    question_col = header_map.get("question")
    gt_col = header_map.get("ground truth legal basis")
    if not question_col or not gt_col:
        raise ValueError(
            "Google Sheet must contain columns 'Question' and "
            "'Ground Truth Legal Basis'. Found: " + ", ".join(map(str, df.columns))
        )
    result = df[[question_col, gt_col]].copy()
    result.columns = ["Question", "Ground Truth Legal Basis"]
    result["Question"] = result["Question"].fillna("").astype(str).str.strip()
    result["Ground Truth Legal Basis"] = (
        result["Ground Truth Legal Basis"].fillna("").astype(str).str.strip()
    )
    result = result[result["Question"] != ""].reset_index(drop=True)
    return result


def reciprocal_rank_fusion(
    ranked_lists: List[Tuple[List[int], float]],
    rrf_k: int = RRF_K,
) -> Dict[int, float]:
    fused = {}
    for indices, weight in ranked_lists:
        for rank, doc_idx in enumerate(indices, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + weight / (rrf_k + rank)
    return fused


def safe_encode(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def retrieve_and_rerank(
    query: str,
    documents: List[Dict],
    enriched_texts: List[str],
    doc_embeddings: np.ndarray,
    bm25: BM25Okapi,
    dense_model: SentenceTransformer,
    reranker: CrossEncoder,
    hyde_text: str = "",
) -> Tuple[List[int], List[float], Dict[str, float]]:
    started = time.perf_counter()

    # Dense retrieval from original query.
    query_embedding = safe_encode(dense_model, [query], 1)[0]
    dense_query_scores = np.dot(doc_embeddings, query_embedding)
    dense_query_indices = np.argsort(dense_query_scores)[::-1][:DENSE_TOP_K].tolist()

    # BM25 retrieval from original query; lexical match helps exact legal terms.
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_indices = np.argsort(bm25_scores)[::-1][:BM25_TOP_K].tolist()

    ranked_lists = [
        (dense_query_indices, WEIGHT_DENSE_QUERY),
        (bm25_indices, WEIGHT_BM25_QUERY),
    ]

    # HyDE is auxiliary, not the only query representation.
    hyde_indices = []
    if hyde_text and len(hyde_text.strip()) > 10:
        hyde_embedding = safe_encode(dense_model, [hyde_text], 1)[0]
        hyde_scores = np.dot(doc_embeddings, hyde_embedding)
        hyde_indices = np.argsort(hyde_scores)[::-1][:HYDE_DENSE_TOP_K].tolist()
        ranked_lists.append((hyde_indices, WEIGHT_DENSE_HYDE))

    fused_scores = reciprocal_rank_fusion(ranked_lists)

    # Explicit section references receive a small ranking preference.
    query_sections = extract_explicit_sections(query)
    if query_sections:
        for doc_idx, doc in enumerate(documents):
            if section_matches(doc, query_sections):
                fused_scores[doc_idx] = fused_scores.get(doc_idx, 0.0) + EXACT_SECTION_RRF_BONUS

    candidate_indices = sorted(
        fused_scores.keys(), key=lambda idx: fused_scores[idx], reverse=True
    )[:RERANK_POOL_K]

    # Cross-encoder must see the real user query, not the hypothetical passage.
    pairs = [[query, enriched_texts[idx]] for idx in candidate_indices]
    rerank_scores = reranker.predict(
        pairs,
        batch_size=RERANK_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    rerank_scores = np.asarray(rerank_scores, dtype=np.float32).reshape(-1)

    # Final ordering: cross-encoder score is primary; RRF is a small tie-breaker.
    if len(rerank_scores):
        r_min, r_max = float(rerank_scores.min()), float(rerank_scores.max())
        rerank_norm = (
            (rerank_scores - r_min) / (r_max - r_min + 1e-8)
            if r_max > r_min else np.ones_like(rerank_scores)
        )
        rrf_values = np.array([fused_scores[idx] for idx in candidate_indices], dtype=np.float32)
        f_min, f_max = float(rrf_values.min()), float(rrf_values.max())
        rrf_norm = (
            (rrf_values - f_min) / (f_max - f_min + 1e-8)
            if f_max > f_min else np.ones_like(rrf_values)
        )
        final_scores = 0.85 * rerank_norm + 0.15 * rrf_norm
        order = np.argsort(final_scores)[::-1][:FINAL_TOP_K]
        final_indices = [candidate_indices[i] for i in order]
        final_score_list = [float(final_scores[i]) for i in order]
    else:
        final_indices, final_score_list = [], []

    latency = time.perf_counter() - started
    stats = {
        "dense_candidates": float(len(dense_query_indices)),
        "bm25_candidates": float(len(bm25_indices)),
        "hyde_candidates": float(len(hyde_indices)),
        "rerank_candidates": float(len(candidate_indices)),
        "retrieval_latency_seconds": float(latency),
    }
    return final_indices, final_score_list, stats
>>>>>>> cc85289 (Update TaxAssist project files)

class PrecisionReranker:
    """Applies Tax-Year Temporal Filtering and Cross-Encoder Reranking."""

<<<<<<< HEAD
    def __init__(self, reranker_model_name: str = "BAAI/bge-reranker-base"):
        self.reranker = CrossEncoder(reranker_model_name)
=======
def calculate_metrics(detail_rows: List[Dict], ks=(1, 3, 5, 10)) -> Dict:
    evaluated = [r for r in detail_rows if r["Has_Ground_Truth"]]
    n = len(evaluated)
    metrics = {"total_questions": len(detail_rows), "evaluated_questions": n}
    if not n:
        for k in ks:
            metrics[f"Hit@{k}"] = 0.0
            metrics[f"Precision@{k}"] = 0.0
            metrics[f"Recall@{k}"] = 0.0
            metrics[f"nDCG@{k}"] = 0.0
        metrics["MRR"] = 0.0
        return metrics

    reciprocal_sum = 0.0
    for row in evaluated:
        ranks = row["Relevance"]
        first = next((i + 1 for i, rel in enumerate(ranks) if rel), 0)
        if first:
            reciprocal_sum += 1.0 / first

    metrics["MRR"] = reciprocal_sum / n
    for k in ks:
        hits = precision = recall = ndcg = 0.0
        for row in evaluated:
            rels = row["Relevance"][:k]
            relevant_total = max(1, row["Ground_Truth_Relevant_Count"])
            relevant_found = sum(rels)
            hits += 1.0 if relevant_found else 0.0
            precision += relevant_found / k
            recall += relevant_found / relevant_total
            dcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(rels, start=1))
            ideal_count = min(relevant_total, k)
            idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_count + 1))
            ndcg += dcg / idcg if idcg else 0.0
        metrics[f"Hit@{k}"] = hits / n
        metrics[f"Precision@{k}"] = precision / n
        metrics[f"Recall@{k}"] = recall / n
        metrics[f"nDCG@{k}"] = ndcg / n
    return metrics
>>>>>>> cc85289 (Update TaxAssist project files)

    def filter_by_tax_year(self, docs: List[LegalDocument], target_year: str) -> List[LegalDocument]:
        """Tax-Year-Aware Filtering: Ensures provisions are active for target Assessment Year."""
        valid_docs = []
        target_date = datetime.strptime(f"{target_year}-07-01", "%Y-%m-%d")

<<<<<<< HEAD
        for doc in docs:
            effective = datetime.strptime(doc.effective_date, "%Y-%m-%d")
            if target_date < effective:
                continue
            if doc.expiry_date:
                expiry = datetime.strptime(doc.expiry_date, "%Y-%m-%d")
                if target_date > expiry:
                    continue
            valid_docs.append(doc)
        return valid_docs

    def rerank(self, query: str, candidate_docs: Dict[str, List[LegalDocument]], target_year: str, top_n: int = 3) -> List[RerankedPassage]:
        # Merge candidate documents across active paths
        unique_candidates: Dict[str, Tuple[LegalDocument, str]] = {}
        for source, docs in candidate_docs.items():
            for doc in docs:
                if doc.chunk_id not in unique_candidates:
                    unique_candidates[doc.chunk_id] = (doc, source)

        raw_docs = [item[0] for item in unique_candidates.values()]
        sources = [item[1] for item in unique_candidates.values()]

        # Apply Tax Year Filtering
        filtered_docs_and_sources = [
            (doc, src) for doc, src in zip(raw_docs, sources)
            if doc in self.filter_by_tax_year([doc], target_year)
        ]

        if not filtered_docs_and_sources:
            return []

        # Rerank via Cross-Encoder
        pairs = [[query, f"Act: {doc.doc_title}. Section: {doc.section}. Text: {doc.text}"] for doc, _ in filtered_docs_and_sources]
        scores = self.reranker.predict(pairs)
=======
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print("TAXASSIST HYBRID + HYDE RETRIEVAL EVALUATION")
    print("=" * 78)
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Legal file: {JSON_PATH}")

    documents = load_documents()
    print(f"Loaded legal passages: {len(documents)}")

    enriched_texts = [
        f"Section: {d['section']}. Title: {d['title']}. "
        f"Source: {d['source']}. Text: {d['text']}"
        for d in documents
    ]

    print("\nBuilding BM25 index...")
    tokenized_corpus = [tokenize(text) for text in enriched_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}")
    dense_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    dense_model.max_seq_length = MAX_LENGTH

    print("Encoding legal passages...")
    doc_embeddings = dense_model.encode(
        enriched_texts,
        batch_size=DOC_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"Embedding matrix: {doc_embeddings.shape}")

    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print(f"\nLoading reranker: {RERANKER_MODEL_NAME}")
    reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device, max_length=MAX_LENGTH)

    print("\nLoading evaluation questions...")
    df = load_questions()
    print(f"Questions loaded: {len(df)}")
    print(f"Questions with ground truth: {(df['Ground Truth Legal Basis'].str.strip() != '').sum()}")

    details = []
    hyde_cache = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Retrieving"):
        query = row["Question"]
        ground_truth = row["Ground Truth Legal Basis"]
        has_gt = bool(ground_truth and ground_truth.lower() != "nan")

        # Cache each HyDE output once per unique query.
        if query not in hyde_cache:
            hyde_cache[query] = generate_hypothetical_doc(query)
        hyde_text = hyde_cache[query]

        indices, scores, stats = retrieve_and_rerank(
            query=query,
            documents=documents,
            enriched_texts=enriched_texts,
            doc_embeddings=doc_embeddings,
            bm25=bm25,
            dense_model=dense_model,
            reranker=reranker,
            hyde_text=hyde_text,
        )

        retrieved = [documents[i] for i in indices]
        relevance = [
            1 if has_gt and ground_truth_match(doc, ground_truth) else 0
            for doc in retrieved
        ]
        gt_sections = extract_explicit_sections(ground_truth)
        gt_relevant_count = (
            sum(1 for doc in documents if ground_truth_match(doc, ground_truth))
            if has_gt else 0
        )

        details.append({
            "Question": query,
            "Ground_Truth": ground_truth,
            "Has_Ground_Truth": has_gt,
            "Ground_Truth_Sections": ", ".join(sorted(gt_sections)),
            "Top1_Section": retrieved[0]["section"] if len(retrieved) > 0 else "",
            "Top1_Title": retrieved[0]["title"] if len(retrieved) > 0 else "",
            "Top1_Score": scores[0] if len(scores) > 0 else 0.0,
            "Top2_Section": retrieved[1]["section"] if len(retrieved) > 1 else "",
            "Top2_Title": retrieved[1]["title"] if len(retrieved) > 1 else "",
            "Top3_Section": retrieved[2]["section"] if len(retrieved) > 2 else "",
            "Top3_Title": retrieved[2]["title"] if len(retrieved) > 2 else "",
            "Top4_Section": retrieved[3]["section"] if len(retrieved) > 3 else "",
            "Top4_Title": retrieved[3]["title"] if len(retrieved) > 3 else "",
            "Top5_Section": retrieved[4]["section"] if len(retrieved) > 4 else "",
            "Top5_Title": retrieved[4]["title"] if len(retrieved) > 4 else "",
            "Relevance": relevance,
            "Ground_Truth_Relevant_Count": gt_relevant_count,
            "Hit@1": int(any(relevance[:1])),
            "Hit@3": int(any(relevance[:3])),
            "Hit@5": int(any(relevance[:5])),
            "Hit@10": int(any(relevance[:10])),
            "First_Hit_Rank": next((i + 1 for i, x in enumerate(relevance) if x), 0),
            **stats,
        })

        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    metrics = calculate_metrics(details)
    total_time = sum(r["retrieval_latency_seconds"] for r in details)
    metrics["average_latency_seconds"] = (
        total_time / len(details) if details else 0.0
    )
    metrics["total_retrieval_time_seconds"] = total_time
    metrics["total_legal_passages"] = len(documents)
    metrics["questions_without_ground_truth"] = sum(not r["Has_Ground_Truth"] for r in details)
    metrics["architecture"] = (
        "BGE-M3 dense(query) + BM25(query) + HyDE dense + weighted RRF "
        "+ BGE-reranker-v2-m3"
    )
    metrics["models"] = {
        "embedding": EMBEDDING_MODEL_NAME,
        "reranker": RERANKER_MODEL_NAME,
        "hyde_llm": OLLAMA_MODEL,
    }
    metrics["configuration"] = {
        "dense_top_k": DENSE_TOP_K,
        "bm25_top_k": BM25_TOP_K,
        "hyde_dense_top_k": HYDE_DENSE_TOP_K,
        "rerank_pool_k": RERANK_POOL_K,
        "final_top_k": FINAL_TOP_K,
        "rrf_k": RRF_K,
        "weights": {
            "dense_query": WEIGHT_DENSE_QUERY,
            "bm25_query": WEIGHT_BM25_QUERY,
            "dense_hyde": WEIGHT_DENSE_HYDE,
        },
    }

    details_json = OUTPUT_DIR / "hybrid_hyde_retrieval_results.json"
    summary_json = OUTPUT_DIR / "hybrid_hyde_retrieval_summary.json"
    excel_path = OUTPUT_DIR / "hybrid_hyde_retrieval_results.xlsx"

    with details_json.open("w", encoding="utf-8") as file:
        json.dump(details, file, ensure_ascii=False, indent=2)
    with summary_json.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    details_df = pd.DataFrame(details)
    summary_df = pd.DataFrame(
        [{"Metric": key, "Value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}
         for key, value in metrics.items()]
    )
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        details_df.to_excel(writer, sheet_name="Details", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print("\n" + "=" * 78)
    print("FINAL HYBRID + HYDE RETRIEVAL SUMMARY")
    print("=" * 78)
    for key in ("Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR",
                "Precision@5", "Recall@5", "nDCG@5"):
        print(f"{key:24s}: {metrics.get(key, 0.0):.4f}")
    print(f"Evaluated questions      : {metrics['evaluated_questions']}")
    print(f"Questions without GT     : {metrics['questions_without_ground_truth']}")
    print(f"Legal passages           : {len(documents)}")
    print(f"Average retrieval latency: {metrics['average_latency_seconds']:.4f} sec")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Excel: {excel_path}")
    print("Evaluation completed.")
>>>>>>> cc85289 (Update TaxAssist project files)

        reranked = []
        for idx, score in enumerate(scores):
            doc, src = filtered_docs_and_sources[idx]
            reranked.append(RerankedPassage(doc=doc, rerank_score=float(score), retrieval_source=src))

        # Sort descending by cross-encoder score
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)
        return reranked[:top_n]


# =====================================================================
# STAGE 5 & 6: CORRECTIVE EVIDENCE GATE & RETRY LOOP
# =====================================================================

class CorrectiveEvidenceGate:
    """Verifies evidence completeness, citation clarity, and triggers retries if insufficient."""

    def evaluate_evidence(self, intent: QueryIntent, passages: List[RerankedPassage], min_score_threshold: float = 0.3) -> GateVerificationResult:
        missing_elements = []

        if not passages:
            return GateVerificationResult(
                status="INSUFFICIENT",
                score=0.0,
                verified_passages=[],
                missing_elements=["No passages passed tax-year filtering"],
                suggested_reformulated_query=f"{intent.legal_topic} Bangladesh Income Tax Act {intent.detected_year}"
            )

        top_score = passages[0].rerank_score

        # Verification Check 1: Score Threshold
        if top_score < min_score_threshold:
            missing_elements.append("Low semantic relevance score from reranker")

        # Verification Check 2: Explicit Section Match if requested
        if intent.detected_section:
            matched_sec = any(p.doc.section.lower() == intent.detected_section.lower() for p in passages)
            if not matched_sec:
                missing_elements.append(f"Explicitly requested Section {intent.detected_section} was missing")

        # Decision
        if missing_elements:
            return GateVerificationResult(
                status="INSUFFICIENT",
                score=top_score,
                verified_passages=passages,
                missing_elements=missing_elements,
                suggested_reformulated_query=f"Bangladesh Income Tax Act {intent.legal_topic} rules"
            )

        return GateVerificationResult(
            status="SUFFICIENT",
            score=top_score,
            verified_passages=passages,
            missing_elements=[]
        )


# =====================================================================
# STAGE 7 & 8: PIPELINE ORCHESTRATOR & OUTPUT FORMATTER
# =====================================================================

class LegalRetrievalPipeline:
    def __init__(self, corpus: List[LegalDocument]):
        self.router = RuleBasedQueryRouter()
        self.retriever = AdaptiveRetrievalEngine(corpus)
        self.reranker = PrecisionReranker()
        self.gate = CorrectiveEvidenceGate()

    def run(self, user_query: str, max_retries: int = 1) -> Dict[str, Any]:
        # 1. Route Query
        intent = self.router.route_query(user_query)
        current_query = user_query

        for attempt in range(max_retries + 1):
            # 2. Adaptive Retrieval
            raw_results = self.retriever.execute_retrieval(intent)

            # 3. Precision Reranking & Tax-Year Filtering
            reranked = self.reranker.rerank(current_query, raw_results, target_year=intent.detected_year)

            # 4. Corrective Evidence Gate
            gate_result = self.gate.evaluate_evidence(intent, reranked)

            if gate_result.status == "SUFFICIENT" or attempt == max_retries:
                # Format Verified Legal Evidence Output
                verified_evidence = []
                for p in gate_result.verified_passages:
                    verified_evidence.append({
                        "act_title": p.doc.doc_title,
                        "section": p.doc.section,
                        "subsection": p.doc.subsection,
                        "applicable_tax_year": intent.detected_year,
                        "retrieval_source": p.retrieval_source,
                        "rerank_score": round(p.rerank_score, 4),
                        "text": p.doc.text
                    })

                return {
                    "query": user_query,
                    "normalized_intent": intent.model_dump(),
                    "gate_status": gate_result.status,
                    "retry_attempts": attempt,
                    "verified_evidence": verified_evidence
                }

            # Retry Logic: Expand query & route to BGE-M3 for broader coverage
            current_query = gate_result.suggested_reformulated_query
            intent.required_routes.append("bge_m3")


# =====================================================================
# VERIFICATION / DEMONSTRATION RUN
# =====================================================================

if __name__ == "__main__":
<<<<<<< HEAD
    # Sample Knowledge Base (Income Tax Act 2023 & Amendments)
    sample_corpus = [
        LegalDocument(
            chunk_id="doc_32",
            doc_type="Act",
            doc_title="Income Tax Act 2023",
            section="32",
            text="Statutory deduction for employment income is one-third of gross salary or Tk 500,000, whichever is lower.",
            effective_year="2023",
            effective_date="2023-07-01",
            related_sections=["33", "34"]
        ),
        LegalDocument(
            chunk_id="doc_sixth_sched",
            doc_type="Act",
            doc_title="Income Tax Act 2023 - Sixth Schedule",
            section="Sixth Schedule Part 3",
            text="Tax rebate is calculated at 15% on allowable investments up to 20% of total taxable income or Tk 1 Crore.",
            effective_year="2023",
            effective_date="2023-07-01",
            related_sections=["32"]
        )
    ]

    # Initialize System
    pipeline = LegalRetrievalPipeline(sample_corpus)

    # Test Query (Banglish / Multilingual input)
    query = "Amr salary income a dhara 32 anujayi tax rebate r deduction koto?"
    print(f"\nProcessing User Query: '{query}'...")

    result = pipeline.run(query)

    import json
    print("\n--- VERIFIED LEGAL EVIDENCE OUTPUT ---")
    print(json.dumps(result, indent=2))
=======
    main()
k
>>>>>>> cc85289 (Update TaxAssist project files)
