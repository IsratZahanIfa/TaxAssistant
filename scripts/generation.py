"""
TaxAssist: Act 2023-only Multi-turn RAG Answer Generation
==========================================================

Source:
    E:\\TaxAssist\\data\\Knowledge_Dataset\\income_tax_act_2023.json

Pipeline:
    JSON validation
      -> Dedicated Act 2023 FAISS index
      -> Baseline retrieval (original query)
      -> Proposed retrieval (history-aware query rewriting + HyDE)
      -> Cross-encoder reranking
      -> Cross-reference context expansion
      -> Evidence validation
      -> Ollama / Gemini answer generation
      -> Citation validation
      -> Excel experiment results

Run:
    python -u scripts\\generation.py

Backend:
    Ollama:
        python -u scripts\\generation.py --gen local

    Gemini:
        python -u scripts\\generation.py --gen gemini

Environment variables for Gemini:
    GEMINI_API_KEY or GOOGLE_API_KEY

Required packages:
    pip install torch sentence-transformers FlagEmbedding
    pip install langchain-community langchain-huggingface
    pip install langchain-google-genai faiss-cpu pandas openpyxl requests
"""

import os
import re
import sys
import json
import time
import argparse
import logging
import warnings

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import torch

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from FlagEmbedding import FlagReranker

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(r"E:\TaxAssist")

JSON_PATH = (
    ROOT
    / "data"
    / "Knowledge_Dataset"
    / "income_tax_act_2023.json"
)

OUTPUT_DIR = ROOT / "results" / "generation"
LOG_DIR = ROOT / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Existing question dataset
DATASET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1uWqX_R0ZIToD7LZu0MelN3OsxJbyxroZBWcBqqlnp3s/"
    "export?format=xlsx"
)

OUTPUT_FILE = OUTPUT_DIR / "act_2023_generation_results.xlsx"

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

OLLAMA_MODEL = "gemma3:12b"
GEMINI_MODEL = "gemini-3-27b-it"

TOP_K_RETRIEVAL = 30
TOP_K_RERANK = 5
MAX_CROSS_REFERENCES = 3

USE_HYDE = True
USE_RERANKER = True
USE_CROSS_REFERENCES = True

# Set to True only if you want to rebuild the index every run.
# This implementation builds the index from JSON in memory.
# It never loads the old mixed-law FAISS index.

MAX_CONTEXT_CHARS = 18000
MAX_HISTORY_TURNS = 5

# Keep the experiment reproducible.
EXPERIMENT_NAME = "Act2023_Baseline_vs_Proposed"
PROMPT_VERSION = "v2.0"


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "generation.log",
            encoding="utf-8"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("TaxAssistGeneration")


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("=" * 80)
logger.info("TAXASSIST - INCOME TAX ACT 2023")
logger.info("MULTI-TURN RAG GENERATION")
logger.info("=" * 80)
logger.info("Device: %s", DEVICE)

if torch.cuda.is_available():
    logger.info("GPU: %s", torch.cuda.get_device_name(0))


# ============================================================
# 4. JSON DATA LOADER AND VALIDATOR
# ============================================================

def load_act_2023_json(json_path: Path) -> List[Document]:
    """
    Load only valid Income Tax Act 2023 section records.

    Expected JSON fields:
        section, record_type, title, chapter, part,
        page, source, text, id
    """

    if not json_path.exists():
        raise FileNotFoundError(
            f"Income Tax Act JSON file not found: {json_path}"
        )

    logger.info("Loading legal JSON: %s", json_path)

    with open(json_path, "r", encoding="utf-8") as file:
        raw_data = json.load(file)

    if isinstance(raw_data, list):
        records = raw_data

    elif isinstance(raw_data, dict):
        records = raw_data.get(
            "documents",
            raw_data.get(
                "data",
                raw_data.get("records", [])
            )
        )

    else:
        raise ValueError(
            "Unsupported JSON structure. Expected list or dictionary."
        )

    if not isinstance(records, list) or not records:
        raise ValueError(
            "No records found in Income Tax Act JSON."
        )

    documents = []
    seen_ids = set()

    rejected_source = 0
    rejected_type = 0
    rejected_content = 0
    duplicates = 0

    for index, record in enumerate(records):

        if not isinstance(record, dict):
            rejected_content += 1
            continue

        source = str(record.get("source", "")).strip()
        record_type = str(
            record.get("record_type", "")
        ).strip().lower()

        section = str(record.get("section", "")).strip()
        title = str(record.get("title", "")).strip()
        text = str(record.get("text", "")).strip()

        # Strict source check
        normalized_source = re.sub(
            r"[^a-z0-9]",
            "",
            source.lower()
        )

        if normalized_source != "incometaxact2023":
            rejected_source += 1
            continue

        if record_type != "act_section":
            rejected_type += 1
            continue

        if not section or not text:
            rejected_content += 1
            continue

        document_id = str(
            record.get(
                "id",
                f"ita_2023_sec_{section}_record_{index}"
            )
        ).strip()

        if document_id in seen_ids:
            duplicates += 1
            continue

        seen_ids.add(document_id)

        metadata = {
            "id": document_id,
            "section": section,
            "title": title,
            "chapter": str(record.get("chapter", "")),
            "part": str(record.get("part", "")),
            "page": record.get("page", ""),
            "source": "Income Tax Act, 2023",
            "record_type": "act_section",
            "law_number": section,
        }

        page_content = (
            f"Income Tax Act, 2023\n"
            f"Section: {section}\n"
            f"Title: {title}\n"
            f"Chapter: {metadata['chapter']}\n"
            f"Part: {metadata['part']}\n"
            f"Page: {metadata['page']}\n\n"
            f"{text}"
        )

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata
            )
        )

    logger.info("Total JSON records: %d", len(records))
    logger.info("Accepted Act 2023 sections: %d", len(documents))
    logger.info("Rejected source: %d", rejected_source)
    logger.info("Rejected record type: %d", rejected_type)
    logger.info("Rejected empty/invalid: %d", rejected_content)
    logger.info("Duplicate IDs skipped: %d", duplicates)

    if not documents:
        raise RuntimeError(
            "No valid Income Tax Act 2023 section documents found."
        )

    unique_sections = {
        doc.metadata["section"] for doc in documents
    }

    logger.info(
        "Unique section identifiers: %d",
        len(unique_sections)
    )

    return documents


# ============================================================
# 5. BUILD A DEDICATED ACT 2023 VECTOR INDEX
# ============================================================

def build_act_2023_vector_store(
    documents: List[Document],
    embedding_model: HuggingFaceEmbeddings
) -> FAISS:

    logger.info("Building dedicated Act 2023 FAISS index...")
    logger.info("Documents to embed: %d", len(documents))

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model
    )

    indexed_count = vector_store.index.ntotal

    logger.info("FAISS indexed vectors: %d", indexed_count)

    if indexed_count != len(documents):
        raise RuntimeError(
            "FAISS document count does not match JSON document count."
        )

    # Verify the docstore contains only Act 2023 documents.
    indexed_docs = list(
        vector_store.docstore._dict.values()
    )

    for doc in indexed_docs:

        if doc.metadata.get("source") != "Income Tax Act, 2023":
            raise RuntimeError(
                "Non-Act-2023 document detected in dedicated index."
            )

    logger.info("Act 2023-only index validation successful.")

    return vector_store


# ============================================================
# 6. LLM INITIALIZATION
# ============================================================

def initialize_llm(backend: str):

    backend = backend.lower()

    if backend == "local":

        logger.info(
            "Initializing Ollama model: %s",
            OLLAMA_MODEL
        )

        return Ollama(
            model=OLLAMA_MODEL,
            temperature=0.1
        )

    if backend == "gemini":

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

        except ImportError as exc:
            raise ImportError(
                "Install Gemini integration using:\n"
                "pip install langchain-google-genai"
            ) from exc

        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is not configured."
            )

        logger.info(
            "Initializing Gemini model: %s",
            GEMINI_MODEL
        )

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.1
        )

    raise ValueError(
        "Invalid backend. Choose 'local' or 'gemini'."
    )


def invoke_llm(llm, prompt: str) -> str:
    """Normalize Ollama and Gemini responses."""

    response = llm.invoke(prompt)

    if isinstance(response, str):
        return response.strip()

    if hasattr(response, "content"):
        return str(response.content).strip()

    return str(response).strip()


# ============================================================
# 7. HISTORY-AWARE QUERY REWRITING
# ============================================================

def rewrite_query(
    llm,
    query: str,
    history: List[Dict[str, str]]
) -> str:

    if not history:
        return query

    history_text = []

    for turn in history[-MAX_HISTORY_TURNS:]:

        history_text.append(
            f"User: {turn.get('question', '')}\n"
            f"Assistant: {turn.get('answer', '')}"
        )

    history_block = "\n\n".join(history_text)

    prompt = f"""
You are a query rewriting component for a Bangladesh Income Tax Act 2023 RAG system.

Rewrite the latest user question as a standalone legal retrieval query.

Rules:
1. Preserve the user's original intent.
2. Use previous conversation only when necessary to resolve references.
3. Do not answer the question.
4. Do not invent legal facts, section numbers, tax rates, or dates.
5. Do not add assumptions.
6. Return only one rewritten query.

Conversation history:
{history_block}

Latest question:
{query}

Standalone retrieval query:
"""

    try:
        rewritten = invoke_llm(llm, prompt)

        rewritten = re.sub(
            r"^(rewritten query|standalone retrieval query)\s*:\s*",
            "",
            rewritten,
            flags=re.IGNORECASE
        ).strip().strip('"')

        if not rewritten:
            return query

        return rewritten

    except Exception as exc:

        logger.exception(
            "Query rewriting failed: %s",
            exc
        )

        return query


# ============================================================
# 8. HYDE
# ============================================================

def generate_hyde_document(
    llm,
    query: str
) -> str:

    prompt = f"""
You are assisting a legal information retrieval system.

Write a short hypothetical legal passage that may be relevant
to the following question under the Bangladesh Income Tax Act, 2023.

Important:
- Do not invent section numbers.
- Do not invent tax rates or deadlines.
- Do not claim that the passage is an actual law.
- Focus on legal concepts and terminology.
- This passage is only for retrieval, not for answering the user.

Question:
{query}

Hypothetical retrieval passage:
"""

    try:

        result = invoke_llm(llm, prompt)

        return result if result else query

    except Exception as exc:

        logger.warning(
            "HyDE generation failed; using original query: %s",
            exc
        )

        return query


# ============================================================
# 9. CROSS-ENCODER RERANKER
# ============================================================

class LegalReranker:

    def __init__(self, model_name: str):

        self.model_name = model_name
        self.model = None

    def _load(self):

        if self.model is None:

            logger.info(
                "Loading reranker: %s",
                self.model_name
            )

            try:

                self.model = FlagReranker(
                    self.model_name,
                    use_fp16=torch.cuda.is_available()
                )

            except Exception:

                logger.exception(
                    "GPU reranker initialization failed; "
                    "trying CPU-compatible initialization."
                )

                self.model = FlagReranker(
                    self.model_name,
                    use_fp16=False
                )

    def rerank(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:

        if not documents:
            return []

        self._load()

        pairs = [
            [query, doc.page_content]
            for doc in documents
        ]

        scores = self.model.compute_score(
            pairs,
            normalize=True
        )

        if isinstance(scores, (int, float)):
            scores = [scores]

        ranked = []

        for doc, score in zip(documents, scores):

            doc.metadata["cross_encoder_score"] = float(score)

            ranked.append(doc)

        ranked.sort(
            key=lambda item: item.metadata.get(
                "cross_encoder_score",
                float("-inf")
            ),
            reverse=True
        )

        return ranked


# ============================================================
# 10. ACT 2023 RETRIEVER
# ============================================================

class Act2023Retriever:

    def __init__(
        self,
        vector_store: FAISS,
        reranker: LegalReranker,
        llm
    ):

        self.vector_store = vector_store
        self.reranker = reranker
        self.llm = llm

        self.section_lookup = defaultdict(list)

        for doc in vector_store.docstore._dict.values():

            section = str(
                doc.metadata.get("section", "")
            )

            self.section_lookup[section].append(doc)

    @staticmethod
    def _is_act_2023(doc: Document) -> bool:

        return (
            doc.metadata.get("source")
            == "Income Tax Act, 2023"
            and doc.metadata.get("record_type")
            == "act_section"
        )

    def _dense_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[Document, float]]:

        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=min(
                top_k,
                self.vector_store.index.ntotal
            )
        )

        # Fail-closed source validation.
        return [
            (doc, score)
            for doc, score in results
            if self._is_act_2023(doc)
        ]

    def retrieve(
        self,
        query: str,
        use_hyde: bool = True,
        use_reranker: bool = True,
        top_k: int = TOP_K_RETRIEVAL,
        final_k: int = TOP_K_RERANK
    ) -> Dict[str, Any]:

        started = time.perf_counter()

        hyde_text = (
            generate_hyde_document(self.llm, query)
            if use_hyde
            else query
        )

        search_queries = []

        if hyde_text.strip():
            search_queries.append(hyde_text)

        # Keep original query as a retrieval fallback.
        if query not in search_queries:
            search_queries.append(query)

        candidate_map = {}

        for search_query in search_queries:

            try:

                results = self._dense_search(
                    search_query,
                    top_k
                )

                for doc, distance in results:

                    doc_id = doc.metadata.get("id")

                    if doc_id not in candidate_map:

                        candidate_map[doc_id] = {
                            "doc": doc,
                            "distance": float(distance),
                            "retrieval_query": search_query
                        }

            except Exception as exc:

                logger.warning(
                    "Retrieval attempt failed: %s",
                    exc
                )

        candidates = list(candidate_map.values())

        # Fallback has already searched original query.
        candidate_docs = [
            item["doc"]
            for item in candidates
        ]

        if use_reranker and candidate_docs:

            try:

                ranked_docs = self.reranker.rerank(
                    query,
                    candidate_docs
                )

            except Exception as exc:

                logger.exception(
                    "Reranking failed; using dense retrieval order: %s",
                    exc
                )

                ranked_docs = candidate_docs

        else:

            ranked_docs = candidate_docs

        final_docs = ranked_docs[:final_k]

        for doc in final_docs:

            matching = candidate_map.get(
                doc.metadata.get("id"),
                {}
            )

            doc.metadata["faiss_distance"] = matching.get(
                "distance"
            )

            doc.metadata["retrieval_query"] = matching.get(
                "retrieval_query",
                query
            )

        elapsed = time.perf_counter() - started

        return {
            "documents": final_docs,
            "candidate_count": len(candidate_docs),
            "retrieval_time": elapsed,
            "hyde_used": use_hyde,
        }

    # --------------------------------------------------------
    # CROSS-REFERENCE EXPANSION
    # --------------------------------------------------------

    def expand_cross_references(
        self,
        documents: List[Document],
        max_extra: int = MAX_CROSS_REFERENCES
    ) -> List[Document]:

        if not USE_CROSS_REFERENCES or not documents:
            return documents

        expanded = list(documents)

        seen_ids = {
            doc.metadata.get("id")
            for doc in expanded
        }

        references = []

        # Detect explicit legal section references.
        pattern = re.compile(
            r"\bsections?\s+(\d{1,3})\b",
            flags=re.IGNORECASE
        )

        for doc in documents:

            matches = pattern.findall(
                doc.page_content
            )

            for section in matches:

                if section not in references:
                    references.append(section)

        added = 0

        for section in references:

            if added >= max_extra:
                break

            for referenced_doc in self.section_lookup.get(
                section,
                []
            ):

                doc_id = referenced_doc.metadata.get("id")

                if doc_id in seen_ids:
                    continue

                if not self._is_act_2023(referenced_doc):
                    continue

                expanded.append(referenced_doc)
                seen_ids.add(doc_id)
                added += 1

                break

        return expanded


# ============================================================
# 11. EVIDENCE CONTEXT BUILDER
# ============================================================

def build_evidence_context(
    documents: List[Document]
) -> str:

    context_parts = []
    total_chars = 0

    for index, doc in enumerate(documents, start=1):

        metadata = doc.metadata

        source = metadata.get("source", "")
        section = metadata.get("section", "")
        title = metadata.get("title", "")
        page = metadata.get("page", "")
        chapter = metadata.get("chapter", "")

        if source != "Income Tax Act, 2023":
            continue

        evidence = f"""
[EVIDENCE {index}]
Source: {source}
Section: {section}
Title: {title}
Chapter: {chapter}
Page: {page}

Legal text:
{doc.page_content}
""".strip()

        if total_chars + len(evidence) > MAX_CONTEXT_CHARS:
            break

        context_parts.append(evidence)
        total_chars += len(evidence)

    return "\n\n" + ("-" * 70) + "\n\n".join(
        context_parts
    )


def validate_evidence(
    documents: List[Document]
) -> Tuple[bool, str]:

    valid_docs = [
        doc for doc in documents
        if (
            doc.metadata.get("source")
            == "Income Tax Act, 2023"
            and doc.metadata.get("record_type")
            == "act_section"
            and doc.page_content.strip()
        )
    ]

    if not valid_docs:
        return False, "No valid Act 2023 evidence retrieved."

    return True, "Valid Act 2023 evidence retrieved."


# ============================================================
# 12. ANSWER GENERATION PROMPT
# ============================================================

def build_answer_prompt(
    query: str,
    context: str,
    history: Optional[List[Dict[str, str]]] = None
) -> str:

    history_text = ""

    if history:

        recent_history = history[-MAX_HISTORY_TURNS:]

        history_text = "\n".join(
            [
                (
                    f"User: {item.get('question', '')}\n"
                    f"Assistant: {item.get('answer', '')}"
                )
                for item in recent_history
            ]
        )

    prompt = f"""
You are TaxAssist, a legal information assistant for Bangladesh.

Your task is to answer questions using the supplied evidence
from the Income Tax Act, 2023.

STRICT LEGAL EVIDENCE RULES:

1. Use only the supplied evidence to make claims about the law.
2. Never invent tax rates, thresholds, deadlines, deductions,
   exemptions, eligibility rules, or legal section numbers.
3. Every important legal claim must have a section citation.
4. Use citation format: [Section X, Income Tax Act, 2023].
5. Cite only sections actually present in the supplied evidence.
6. Do not cite a section merely because it appears in conversation history.
7. If the evidence does not answer the question, clearly say so.
8. Do not treat a hypothetical retrieval passage as legal evidence.
9. Do not present investment recommendations as guaranteed returns.
10. Do not claim to be a lawyer or replace professional advice.

ANSWER QUALITY RULES:

- Answer the actual question directly.
- Explain legal conditions and exceptions when available.
- Use simple language.
- If the question is in Bangla, answer in Bangla.
- If the question is in English, answer in English.
- If the evidence is incomplete, explain the limitation.
- Do not fill missing information using general knowledge.

Conversation history (context only):
{history_text}

Retrieved legal evidence:
{context}

Current user question:
{query}

Answer:
"""

    return prompt


def generate_answer(
    llm,
    query: str,
    documents: List[Document],
    history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:

    started = time.perf_counter()

    evidence_ok, evidence_message = validate_evidence(
        documents
    )

    if not evidence_ok:

        return {
            "answer": (
                "The available Income Tax Act 2023 evidence "
                "is insufficient to answer this question."
            ),
            "generation_time": 0.0,
            "evidence_sufficient": False,
            "error": evidence_message,
        }

    context = build_evidence_context(documents)

    prompt = build_answer_prompt(
        query=query,
        context=context,
        history=history
    )

    try:

        answer = invoke_llm(
            llm,
            prompt
        )

        if not answer:

            answer = (
                "The available Income Tax Act 2023 evidence "
                "is insufficient to answer this question."
            )

        elapsed = time.perf_counter() - started

        return {
            "answer": answer,
            "generation_time": elapsed,
            "evidence_sufficient": True,
            "error": None,
        }

    except Exception as exc:

        logger.exception(
            "Answer generation failed: %s",
            exc
        )

        elapsed = time.perf_counter() - started

        return {
            "answer": "",
            "generation_time": elapsed,
            "evidence_sufficient": False,
            "error": str(exc),
        }


# ============================================================
# 13. CITATION VALIDATION
# ============================================================

def validate_citations(
    answer: str,
    retrieved_documents: List[Document]
) -> Dict[str, Any]:

    citation_pattern = re.compile(
        r"\[\s*Section\s+(\d+[A-Za-z]?)"
        r"\s*,\s*Income Tax Act\s*,?\s*2023\s*\]",
        flags=re.IGNORECASE
    )

    cited_sections = citation_pattern.findall(
        answer or ""
    )

    retrieved_sections = {
        str(doc.metadata.get("section", ""))
        for doc in retrieved_documents
    }

    valid_citations = [
        section
        for section in cited_sections
        if section in retrieved_sections
    ]

    invalid_citations = [
        section
        for section in cited_sections
        if section not in retrieved_sections
    ]

    unique_cited = list(
        dict.fromkeys(cited_sections)
    )

    return {
        "cited_sections": ", ".join(unique_cited),
        "valid_cited_sections": ", ".join(
            dict.fromkeys(valid_citations)
        ),
        "invalid_cited_sections": ", ".join(
            dict.fromkeys(invalid_citations)
        ),
        "citation_count": len(unique_cited),
        "citation_valid": (
            len(unique_cited) > 0
            and len(invalid_citations) == 0
        ),
    }


# ============================================================
# 14. DATASET LOADER
# ============================================================

def load_question_dataset() -> pd.DataFrame:

    logger.info("Loading question dataset...")

    try:

        df = pd.read_excel(
            DATASET_URL,
            engine="openpyxl"
        )

    except Exception as exc:

        raise RuntimeError(
            "Could not load the Google Sheets Excel dataset. "
            "Check the URL and internet access."
        ) from exc

    required_columns = [
        "Set_ID",
        "Turn",
        "Query",
        "Ground_Truth",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required dataset columns: {missing}"
        )

    df = df.dropna(
        subset=["Query"]
    ).copy()

    df["Query"] = df["Query"].astype(str).str.strip()

    df["Set_ID"] = (
        df["Set_ID"]
        .fillna("DEFAULT")
        .astype(str)
    )

    df["Turn"] = pd.to_numeric(
        df["Turn"],
        errors="coerce"
    ).fillna(1).astype(int)

    df["Ground_Truth"] = (
        df["Ground_Truth"]
        .fillna("")
        .astype(str)
    )

    df = df.sort_values(
        ["Set_ID", "Turn"]
    ).reset_index(drop=True)

    logger.info(
        "Total questions loaded: %d",
        len(df)
    )

    return df


# ============================================================
# 15. DOCUMENT METADATA HELPERS
# ============================================================

def get_retrieved_sections(
    documents: List[Document]
) -> str:

    sections = []

    for doc in documents:

        section = str(
            doc.metadata.get("section", "")
        )

        if section and section not in sections:
            sections.append(section)

    return ", ".join(sections)


def get_retrieved_titles(
    documents: List[Document]
) -> str:

    titles = []

    for doc in documents:

        section = str(
            doc.metadata.get("section", "")
        )

        title = str(
            doc.metadata.get("title", "")
        )

        item = f"Section {section}: {title}"

        if item not in titles:
            titles.append(item)

    return " | ".join(titles)


def get_retrieval_scores(
    documents: List[Document]
) -> str:

    scores = []

    for doc in documents:

        section = doc.metadata.get("section", "")

        rerank_score = doc.metadata.get(
            "cross_encoder_score"
        )

        distance = doc.metadata.get(
            "faiss_distance"
        )

        scores.append(
            f"Sec.{section}: "
            f"reranker={rerank_score}; "
            f"faiss_distance={distance}"
        )

    return " | ".join(scores)


def get_document_ids(
    documents: List[Document]
) -> str:

    return ", ".join(
        str(doc.metadata.get("id", ""))
        for doc in documents
    )


# ============================================================
# 16. EXPERIMENT RUNNER
# ============================================================

def run_experiment(
    df: pd.DataFrame,
    retriever: Act2023Retriever,
    llm,
) -> pd.DataFrame:

    results = []

    # Proposed conversation history is maintained separately
    # for each Set_ID.
    proposed_histories = defaultdict(list)

    total = len(df)

    for row_number, row in df.iterrows():

        set_id = str(row["Set_ID"])
        turn = int(row["Turn"])
        original_query = str(row["Query"])
        ground_truth = str(row["Ground_Truth"])

        logger.info("-" * 80)
        logger.info(
            "Question %d/%d | Set_ID=%s | Turn=%d",
            row_number + 1,
            total,
            set_id,
            turn
        )
        logger.info("Query: %s", original_query)

        proposed_history = proposed_histories[set_id]

        # ----------------------------------------------------
        # BASELINE: Original query, no conversation history
        # ----------------------------------------------------

        baseline_start = time.perf_counter()

        baseline_retrieval = retriever.retrieve(
            query=original_query,
            use_hyde=USE_HYDE,
            use_reranker=USE_RERANKER,
            top_k=TOP_K_RETRIEVAL,
            final_k=TOP_K_RERANK
        )

        baseline_docs = baseline_retrieval["documents"]

        baseline_docs = retriever.expand_cross_references(
            baseline_docs
        )

        baseline_answer_result = generate_answer(
            llm=llm,
            query=original_query,
            documents=baseline_docs,
            history=None
        )

        baseline_total_time = (
            time.perf_counter() - baseline_start
        )

        baseline_citations = validate_citations(
            baseline_answer_result["answer"],
            baseline_docs
        )

        # ----------------------------------------------------
        # PROPOSED: History-aware query rewriting
        # ----------------------------------------------------

        proposed_start = time.perf_counter()

        rewritten_query = rewrite_query(
            llm=llm,
            query=original_query,
            history=proposed_history
        )

        proposed_retrieval = retriever.retrieve(
            query=rewritten_query,
            use_hyde=USE_HYDE,
            use_reranker=USE_RERANKER,
            top_k=TOP_K_RETRIEVAL,
            final_k=TOP_K_RERANK
        )

        proposed_docs = proposed_retrieval["documents"]

        proposed_docs = retriever.expand_cross_references(
            proposed_docs
        )

        proposed_answer_result = generate_answer(
            llm=llm,
            query=original_query,
            documents=proposed_docs,
            history=proposed_history
        )

        proposed_total_time = (
            time.perf_counter() - proposed_start
        )

        proposed_citations = validate_citations(
            proposed_answer_result["answer"],
            proposed_docs
        )

        # ----------------------------------------------------
        # SAVE PROPOSED HISTORY
        # ----------------------------------------------------

        proposed_histories[set_id].append(
            {
                "question": original_query,
                "answer": proposed_answer_result["answer"],
            }
        )

        # ----------------------------------------------------
        # OUTPUT RECORD
        # ----------------------------------------------------

        result = {

            "Experiment": EXPERIMENT_NAME,
            "Prompt_Version": PROMPT_VERSION,
            "Backend": "Configured LLM",
            "LLM_Model": (
                OLLAMA_MODEL
                if getattr(llm, "__class__", type(llm)).__name__
                == "Ollama"
                else GEMINI_MODEL
            ),

            "Set_ID": set_id,
            "Turn": turn,
            "Query": original_query,
            "Ground_Truth": ground_truth,

            # Baseline
            "Baseline_Rewritten_Query": original_query,
            "Baseline_Retrieved_Sections": get_retrieved_sections(
                baseline_docs
            ),
            "Baseline_Retrieved_Titles": get_retrieved_titles(
                baseline_docs
            ),
            "Baseline_Document_IDs": get_document_ids(
                baseline_docs
            ),
            "Baseline_Retrieval_Scores": get_retrieval_scores(
                baseline_docs
            ),
            "Baseline_Candidate_Count": baseline_retrieval[
                "candidate_count"
            ],
            "Baseline_Retrieval_Time": baseline_retrieval[
                "retrieval_time"
            ],
            "Baseline_Answer": baseline_answer_result[
                "answer"
            ],
            "Baseline_Generation_Time": baseline_answer_result[
                "generation_time"
            ],
            "Baseline_Total_Time": baseline_total_time,
            "Baseline_Cited_Sections": baseline_citations[
                "cited_sections"
            ],
            "Baseline_Valid_Cited_Sections": baseline_citations[
                "valid_cited_sections"
            ],
            "Baseline_Invalid_Cited_Sections": baseline_citations[
                "invalid_cited_sections"
            ],
            "Baseline_Citation_Valid": baseline_citations[
                "citation_valid"
            ],
            "Baseline_Evidence_Sufficient": baseline_answer_result[
                "evidence_sufficient"
            ],
            "Baseline_Error": baseline_answer_result[
                "error"
            ],

            # Proposed
            "Proposed_Rewritten_Query": rewritten_query,
            "Proposed_Retrieved_Sections": get_retrieved_sections(
                proposed_docs
            ),
            "Proposed_Retrieved_Titles": get_retrieved_titles(
                proposed_docs
            ),
            "Proposed_Document_IDs": get_document_ids(
                proposed_docs
            ),
            "Proposed_Retrieval_Scores": get_retrieval_scores(
                proposed_docs
            ),
            "Proposed_Candidate_Count": proposed_retrieval[
                "candidate_count"
            ],
            "Proposed_Retrieval_Time": proposed_retrieval[
                "retrieval_time"
            ],
            "Proposed_Answer": proposed_answer_result[
                "answer"
            ],
            "Proposed_Generation_Time": proposed_answer_result[
                "generation_time"
            ],
            "Proposed_Total_Time": proposed_total_time,
            "Proposed_Cited_Sections": proposed_citations[
                "cited_sections"
            ],
            "Proposed_Valid_Cited_Sections": proposed_citations[
                "valid_cited_sections"
            ],
            "Proposed_Invalid_Cited_Sections": proposed_citations[
                "invalid_cited_sections"
            ],
            "Proposed_Citation_Valid": proposed_citations[
                "citation_valid"
            ],
            "Proposed_Evidence_Sufficient": proposed_answer_result[
                "evidence_sufficient"
            ],
            "Proposed_Error": proposed_answer_result[
                "error"
            ],
        }

        results.append(result)

        logger.info(
            "Baseline sections: %s",
            result["Baseline_Retrieved_Sections"]
        )

        logger.info(
            "Proposed sections: %s",
            result["Proposed_Retrieved_Sections"]
        )

        logger.info(
            "Baseline citation valid: %s",
            result["Baseline_Citation_Valid"]
        )

        logger.info(
            "Proposed citation valid: %s",
            result["Proposed_Citation_Valid"]
        )

        logger.info(
            "Question completed in %.2f seconds",
            baseline_total_time + proposed_total_time
        )

        # Save intermediate results after every question.
        # This prevents losing completed work if a later question fails.
        pd.DataFrame(results).to_excel(
            OUTPUT_FILE,
            index=False,
            engine="openpyxl"
        )

    return pd.DataFrame(results)


# ============================================================
# 17. SUMMARY
# ============================================================

def print_summary(results: pd.DataFrame):

    if results.empty:
        logger.warning("No results generated.")
        return

    logger.info("=" * 80)
    logger.info("GENERATION SUMMARY")
    logger.info("=" * 80)

    logger.info(
        "Total questions: %d",
        len(results)
    )

    for method in ["Baseline", "Proposed"]:

        citation_column = f"{method}_Citation_Valid"
        retrieval_column = f"{method}_Retrieval_Time"
        generation_column = f"{method}_Generation_Time"
        total_column = f"{method}_Total_Time"

        citation_rate = (
            results[citation_column]
            .fillna(False)
            .astype(bool)
            .mean() * 100
        )

        avg_retrieval = results[
            retrieval_column
        ].mean()

        avg_generation = results[
            generation_column
        ].mean()

        avg_total = results[
            total_column
        ].mean()

        logger.info(
            "%s | Citation-valid rate: %.2f%%",
            method,
            citation_rate
        )

        logger.info(
            "%s | Avg retrieval: %.3fs",
            method,
            avg_retrieval
        )

        logger.info(
            "%s | Avg generation: %.3fs",
            method,
            avg_generation
        )

        logger.info(
            "%s | Avg total: %.3fs",
            method,
            avg_total
        )

    logger.info("Results saved to: %s", OUTPUT_FILE)


# ============================================================
# 18. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Act 2023-only TaxAssist multi-turn generation"
        )
    )

    parser.add_argument(
        "--gen",
        choices=["local", "gemini"],
        default="local",
        help="Generation backend"
    )

    args = parser.parse_args()

    logger.info("Generation backend: %s", args.gen)

    # 1. Load JSON
    documents = load_act_2023_json(
        JSON_PATH
    )

    # 2. Initialize embedding model
    logger.info(
        "Loading embedding model: %s",
        EMBEDDING_MODEL
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": DEVICE
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    # 3. Build dedicated index
    vector_store = build_act_2023_vector_store(
        documents,
        embeddings
    )

    # 4. Initialize LLM
    llm = initialize_llm(
        args.gen
    )

    # 5. Initialize reranker
    reranker = LegalReranker(
        RERANKER_MODEL
    )

    # 6. Initialize Act 2023 retriever
    retriever = Act2023Retriever(
        vector_store=vector_store,
        reranker=reranker,
        llm=llm
    )

    # 7. Load evaluation questions
    df = load_question_dataset()

    # 8. Run Baseline vs Proposed
    results = run_experiment(
        df=df,
        retriever=retriever,
        llm=llm
    )

    # 9. Print summary
    print_summary(
        results
    )

    logger.info("=" * 80)
    logger.info("GENERATION COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        logger.warning(
            "Generation interrupted by user."
        )

    except Exception as exc:

        logger.exception(
            "Fatal generation error: %s",
            exc
        )

        raise