import re
import numpy as np
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


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

class PrecisionReranker:
    """Applies Tax-Year Temporal Filtering and Cross-Encoder Reranking."""

    def __init__(self, reranker_model_name: str = "BAAI/bge-reranker-base"):
        self.reranker = CrossEncoder(reranker_model_name)

    def filter_by_tax_year(self, docs: List[LegalDocument], target_year: str) -> List[LegalDocument]:
        """Tax-Year-Aware Filtering: Ensures provisions are active for target Assessment Year."""
        valid_docs = []
        target_date = datetime.strptime(f"{target_year}-07-01", "%Y-%m-%d")

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