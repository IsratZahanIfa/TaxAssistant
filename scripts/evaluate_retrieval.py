import pickle
import re
import torch
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. PREPROCESSING MODULE
# ============================================

# Bengali to English digit mapping
BN_TO_EN_DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')

def normalize_text(text):
    """Normalize text: lowercase and Bengali digits to English."""
    if not text or pd.isna(text):
        return ""
    return str(text).lower().translate(BN_TO_EN_DIGITS)

def extract_section_identifiers(text):
    """Extract section numbers like '19', '19a' from text."""
    text = normalize_text(text)
    # Multiple patterns for section extraction
    patterns = [
        r'(?:section|sec|ধারা|অনুচ্ছেদ|আইন|rule|regulation)?\s*(\d+[a-z]?)',
        r'সেকশন\s*(\d+[a-z]?)',
        r'ধারা\s*(\d+[a-z]?)'
    ]
    raw_sections = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        raw_sections.update(matches)
    
    cleaned = set()
    for sec in raw_sections:
        sec_clean = sec.strip()
        if sec_clean and len(re.sub(r'\D', '', sec_clean)) <= 3:
            cleaned.add(sec_clean)
    return cleaned

def advanced_tokenizer(text):
    """Advanced tokenizer with unigrams, bigrams, and trigrams."""
    text = normalize_text(text)
    tokens = re.findall(r'[\w]+', text)
    bigrams = ['_'.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    trigrams = ['_'.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
    return tokens + bigrams + trigrams

# ============================================
# 2. DOCUMENT ENRICHMENT
# ============================================

def enrich_documents(documents):
    """Enrich documents with parent metadata."""
    enriched = []
    for d in documents:
        p_sec = d.get('parent_sec', '')
        p_title = d.get('parent_title', '')
        c_text = d.get('child_text', '')
        # Multiple context formats for better retrieval
        full_context = f"Section: {p_sec}. Title: {p_title}. Text: {c_text}"
        enriched.append(full_context)
    return enriched

# ============================================
# 3. DYNAMIC ALPHA FUNCTION
# ============================================

def get_dynamic_alpha(query, q_sections):
    """Calculate dynamic alpha based on query characteristics."""
    if q_sections:
        return 0.4  # Section queries → more BM25
    elif len(query.split()) > 10:
        return 0.2  # Long queries → more dense
    else:
        return 0.3  # Default

# ============================================
# 4. MAIN RETRIEVAL PIPELINE
# ============================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Tax RAG Pipeline on device: {device.upper()}")
    print("=" * 70)

    # Paths
    PKL_PATH = r"E:\TaxAssist\taxdatabase\tax_data_parent_child.pkl"
    GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/17ew_umfBKtAeTUEBFtBknNoyQR5KbWRSgceh47DNeYY/export?format=csv&gid=0"

    # Load documents
    print("Loading documents...")
    with open(PKL_PATH, "rb") as f:
        documents = pickle.load(f)
    print(f"  Loaded {len(documents)} documents")

    # Enrich documents
    print("Enriching documents...")
    enriched_texts = enrich_documents(documents)

    # ===== STAGE 1: BM25 INDEX =====
    print("Building BM25 index...")
    corpus = [advanced_tokenizer(t) for t in enriched_texts]
    bm25 = BM25Okapi(corpus)

    # ===== STAGE 2: DENSE MODEL (BGE-M3) =====
    print("Loading BGE-M3 dense model...")
    dense_model = SentenceTransformer("BAAI/bge-m3", device=device)
    print("Encoding documents with BGE-M3...")
    doc_embeddings = dense_model.encode(
        enriched_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # ===== STAGE 3: CROSS-ENCODER RERANKER =====
    print("Loading Cross-Encoder (BAAI/bge-reranker-v2-m3)...")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)

    # Load evaluation data
    print("Loading evaluation data...")
    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    print(f"  Loaded {len(df)} queries")

    # Initialize metrics
    hit_at_1, hit_at_3, hit_at_5 = 0, 0, 0
    mrr_score = 0.0
    total_queries = 0

    # Configurable parameters
    TOP_K = 200
    SECTION_BOOST = 0.7

    print("\nStarting evaluation...")
    print("=" * 70)

    for idx, row in df.iterrows():
        query = str(row.get("Question") or row.get("Query") or "").strip()
        ground_truth = str(row.get("Ground Truth Legal Basis") or "").strip()

        if not query or not ground_truth or ground_truth.lower() == 'nan':
            continue

        total_queries += 1
        norm_query = normalize_text(query)
        q_sections = extract_section_identifiers(query)

        # Find documents containing the same sections
        section_indices = []
        if q_sections:
            for i, d in enumerate(documents):
                doc_secs = extract_section_identifiers(d.get('parent_sec', '')).union(
                            extract_section_identifiers(d.get('child_text', '')))
                if q_sections.intersection(doc_secs):
                    section_indices.append(i)

        # ===== HYBRID RETRIEVAL =====
        # BM25 scores
        tokenized_q = advanced_tokenizer(norm_query)
        bm25_scores = bm25.get_scores(tokenized_q)
        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)

        # Dense scores
        q_embed = dense_model.encode([norm_query], normalize_embeddings=True)
        dense_scores = np.dot(doc_embeddings, q_embed.T).squeeze()
        if np.max(dense_scores) > 0:
            dense_scores = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores) + 1e-8)

        # Section boost
        boost = np.zeros(len(documents))
        for i in section_indices:
            doc_secs = extract_section_identifiers(documents[i].get('parent_sec', '')).union(
                        extract_section_identifiers(documents[i].get('child_text', '')))
            if q_sections.intersection(doc_secs):
                boost[i] = SECTION_BOOST * (1 + len(q_sections.intersection(doc_secs)) * 0.2)

        # Dynamic alpha
        alpha = get_dynamic_alpha(query, q_sections)

        # Hybrid scores
        hybrid_scores = alpha * bm25_scores + (1 - alpha) * dense_scores + boost
        top_k_idx = np.argsort(hybrid_scores)[::-1][:TOP_K].tolist()

        # ===== CROSS-ENCODER RERANKING =====
        rerank_pairs = [[query, enriched_texts[i]] for i in top_k_idx]
        rerank_scores = reranker.predict(rerank_pairs)

        # Get top 5
        sorted_pairs = sorted(zip(rerank_scores, top_k_idx), reverse=True)
        final_top_5_idx = [doc_idx for _, doc_idx in sorted_pairs[:5]]

        # ===== SECTION FALLBACK =====
        if section_indices:
            for sec_idx in section_indices:
                if sec_idx not in final_top_5_idx:
                    final_top_5_idx[-1] = sec_idx
                    break

        # ===== EVALUATION =====
        retrieved_docs = [documents[i] for i in final_top_5_idx]
        gt_sections = extract_section_identifiers(ground_truth)

        matches = []
        for doc in retrieved_docs:
            doc_secs = extract_section_identifiers(doc.get('parent_sec', '')).union(
                        extract_section_identifiers(doc.get('child_text', '')))
            is_match = 1 if gt_sections.intersection(doc_secs) else 0
            matches.append(is_match)

        if sum(matches[:1]) > 0:
            hit_at_1 += 1
        if sum(matches[:3]) > 0:
            hit_at_3 += 1
        if sum(matches[:5]) > 0:
            hit_at_5 += 1

        first_hit = next((r + 1 for r, m in enumerate(matches) if m), 0)
        if first_hit:
            mrr_score += 1.0 / first_hit

        # Progress indicator
        if total_queries % 10 == 0:
            print(f"  Processed {total_queries} queries...")

    # ===== FINAL RESULTS =====
    print("\n" + "=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)
    if total_queries > 0:
        print(f"Total Queries      : {total_queries}")
        print("-" * 70)
        print(f"Hit@1  Accuracy    : {100 * hit_at_1 / total_queries:.2f}%")
        print(f"Hit@3  Accuracy    : {100 * hit_at_3 / total_queries:.2f}%")
        print(f"Hit@5  Accuracy    : {100 * hit_at_5 / total_queries:.2f}%")
        print("-" * 70)
        print(f"MRR                : {mrr_score / total_queries:.4f}")
        print("=" * 70)

        # Performance analysis
        print("\n📊 PERFORMANCE ANALYSIS:")
        print(f"✓ Hit@5: {100 * hit_at_5 / total_queries:.1f}%")
        if (100 * hit_at_5 / total_queries) >= 80:
            print("🎯 TARGET ACHIEVED! Hit@5 > 80%")
        else:
            remaining = 80 - (100 * hit_at_5 / total_queries)
            print(f"📈 Need {remaining:.1f}% more improvement to reach 80%")
            print("\n💡 RECOMMENDATIONS:")
            print("  1. Add LGBM Reranker (adds 5-10% accuracy)")
            print("  2. Fine-tune BGE-M3 on your dataset (adds 5-10%)")
            print("  3. Use Matryoshka embeddings (512D) for faster retrieval")
            print("  4. Add query expansion with synonyms")

if __name__ == '__main__':
    main()