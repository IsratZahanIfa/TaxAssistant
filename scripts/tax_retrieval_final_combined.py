"""
TaxAssist - FINAL Combined Retrieval Pipeline
===============================================
Combines every technique discussed so far into one pipeline:

  1. Multi-gram tokenizer (unigram + bigram) for BM25
  2. BM25 (sparse, lexical)
  3. BGE-M3 native multi-vector: dense + sparse (lexical weights) + ColBERT
     (token-level late interaction, applied only to the candidate pool
     since it's the most expensive signal)
  4. Reciprocal Rank Fusion (RRF) to combine BM25 + BGE-dense + BGE-sparse
     -- rank-based fusion is far more robust than a fixed-weight average
     across signals with very different score distributions.
  5. Dynamic alpha: queries that explicitly name a section lean more on
     lexical signals (BM25/sparse); conceptual queries lean more on dense.
     Implemented as a re-weighting of the RRF inputs, not a single scalar,
     since RRF doesn't take weights directly -- we approximate by
     duplicating the dominant signal's rank contribution.
  6. Section-aware boosting using the FIXED, two-tier section extractor
     (keyword-anchored, with bare-number fallback only for short
     metadata fields like parent_sec -- never for long prose).
  7. Cross-encoder reranking (BGE-reranker-v2-m3) over a wide candidate
     pool (150).
  8. Late fusion: final rank = weighted blend of reranker score and the
     stage-1 retrieval score (not reranker order alone).
  9. Non-leaky section fallback: if the query names an explicit section
     and no chunk from that section made the final top-5, the best
     scoring exact-section chunk is swapped in. This uses ONLY the
     query's own text, never the ground truth column, so it is safe to
     use in both evaluation and production.

PREREQUISITE: PKL_PATH must point to data rebuilt with
extract_sections_fixed.py -> create_pkl.py -> build_parent_child_index.py
-> embeddings. Running this against the old, uncorrected pkl will not
give meaningful numbers no matter how good the algorithm is.

Install: pip install -U FlagEmbedding rank_bm25 sentence-transformers
"""

import pickle
import re
import torch
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from FlagEmbedding import BGEM3FlagModel

# ============================================================
# TEXT NORMALIZATION / SECTION EXTRACTION (fixed, two-tier)
# ============================================================

BN_TO_EN_DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')

BN_SUFFIXES = ["গুলোর", "গুলো", "দেরকে", "দের", "টির", "টার", "টিতে",
               "টিকে", "টি", "টা", "এর", "কে", "তে", "য়", "রা"]
BN_STOPWORDS = {"এবং", "বা", "যে", "যা", "এই", "সেই", "ও", "কি", "কী",
                "না", "হয়", "হবে", "করা", "করে", "জন্য", "থেকে", "সঙ্গে",
                "মধ্যে", "প্রতি", "সহ"}
EN_STOPWORDS = {"the", "a", "an", "of", "in", "on", "for", "to", "and",
                "or", "is", "are", "was", "were"}

SECTION_KEYWORD = r'(?:section|sec\.?|u/s|ss\.?|ধারা|উপধারা)'
_SEC_NUM = r'\d{1,3}[a-z]{0,2}(?:\(\d+[a-z]?\))?'


def normalize_text(text):
    if not text or pd.isna(text):
        return ""
    return str(text).lower().translate(BN_TO_EN_DIGITS)


def light_bn_stem(token):
    for suf in BN_SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 2:
            return token[: -len(suf)]
    return token


def extract_section_identifiers(text, allow_bare_fallback=True, bare_fallback_max_tokens=12):
    """
    Two-tier extraction (see earlier discussion):
    1. keyword-anchored, both word orders ("ধারা ৮২" and "৮২ ধারা")
    2. bare-number fallback ONLY for short fields (metadata-like),
       never for long prose (avoids matching random numbers).
    """
    text = normalize_text(text)
    if not text:
        return set()
    forward = re.findall(SECTION_KEYWORD + r'\s*\.?\s*(' + _SEC_NUM + r')', text)
    backward = re.findall(r'(' + _SEC_NUM + r')\s*' + SECTION_KEYWORD, text)
    keyworded = set()
    for sec in forward + backward:
        base = re.split(r'\(', sec)[0].strip()
        if base:
            keyworded.add(base)
    if keyworded:
        return keyworded
    if allow_bare_fallback and len(text.split()) <= bare_fallback_max_tokens:
        return set(re.findall(r'\b(\d{1,3}[a-z]{0,2})\b', text))
    return set()


def get_doc_sections(doc):
    return (
        extract_section_identifiers(doc.get("parent_sec", ""))
        | extract_section_identifiers(doc.get("child_text", ""), allow_bare_fallback=False)
    )


def simple_tokenizer(text):
    text = normalize_text(text)
    tokens = re.findall(r'[\w]+', text)
    out = []
    for t in tokens:
        if t in BN_STOPWORDS or t in EN_STOPWORDS:
            continue
        out.append(light_bn_stem(t))
    return out


def multigram_tokenizer(text):
    """Unigram + bigram tokens, for BM25."""
    tokens = simple_tokenizer(text)
    bigrams = ['_'.join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
    return tokens + bigrams


def reciprocal_rank_fusion(score_lists, weights=None, k=60):
    """
    score_lists: list of np.array score vectors (higher=better), same length.
    weights: optional list of ints -- how many times to count each signal's
             rank contribution (approximates weighting within RRF).
    """
    if weights is None:
        weights = [1] * len(score_lists)
    n = len(score_lists[0])
    fused = np.zeros(n)
    for scores, w in zip(score_lists, weights):
        ranks = np.argsort(np.argsort(-scores))  # 0 = best
        fused += w * (1.0 / (k + ranks + 1))
    return fused


def is_section_explicit_query(query):
    """True if the query names an explicit section number (drives dynamic alpha)."""
    return len(extract_section_identifiers(query)) > 0


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    print(f"Running FINAL Combined Retrieval Pipeline on: {device.upper()}")

    # !!! MUST point to data rebuilt with extract_sections_fixed.py !!!
    PKL_PATH = r"E:\TaxAssist\taxdatabase\tax_data_parent_child.pkl"
    GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/17ew_umfBKtAeTUEBFtBknNoyQR5KbWRSgceh47DNeYY/export?format=csv&gid=0"

    with open(PKL_PATH, "rb") as f:
        documents = pickle.load(f)

    print(f"Loaded {len(documents):,} chunks.")

    # ---- Sanity check: warn if section collisions still look present ----
    sec_counts = {}
    for d in documents:
        for s in extract_section_identifiers(d.get("parent_sec", "")):
            sec_counts[s] = sec_counts.get(s, 0) + 1
    if sec_counts:
        worst = max(sec_counts.items(), key=lambda kv: kv[1])
        if worst[1] > 30:
            print(
                f"WARNING: parent_sec '{worst[0]}' appears in {worst[1]} chunks. "
                f"If this looks too high, the data may not be fully rebuilt "
                f"with extract_sections_fixed.py -- double check before trusting results."
            )

    enriched_texts = []
    doc_sections = []
    for d in documents:
        p_sec = d.get('parent_sec', '')
        p_title = d.get('parent_title', '')
        c_text = d.get('child_text', '')
        enriched_texts.append(f"Section: {p_sec}. Title: {p_title}. Text: {c_text}")
        doc_sections.append(get_doc_sections(d))

    # ---- BM25 (multi-gram) ----
    print("Building BM25 Index (multi-gram tokenizer)...")
    corpus = [multigram_tokenizer(t) for t in enriched_texts]
    bm25 = BM25Okapi(corpus)

    # ---- BGE-M3 multi-vector (dense + sparse + colbert) ----
    print("Loading BGE-M3 (multi-vector: dense + sparse + colbert)...")
    bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16, device=device)

    print("Encoding documents (dense + sparse + colbert)...")
    doc_out = bge.encode(
        enriched_texts,
        batch_size=12,
        max_length=1024,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    doc_dense = doc_out["dense_vecs"]
    doc_sparse = doc_out["lexical_weights"]
    doc_colbert = doc_out["colbert_vecs"]

    # ---- Cross-encoder reranker ----
    print("Loading Cross-Encoder (BAAI/bge-reranker-v2-m3)...")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)

    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    hit_at_1, hit_at_3, hit_at_5 = 0, 0, 0
    mrr_score = 0.0
    total_queries = 0

    CANDIDATE_POOL = 150
    COLBERT_RESCORE_POOL = 50
    SECTION_BOOST = 0.35
    RERANK_WEIGHT = 0.75
    HYBRID_WEIGHT = 0.25

    print("\nStarting Evaluation...")
    for idx, row in df.iterrows():
        query = str(row.get("Question") or row.get("Query") or "").strip()
        ground_truth = str(row.get("Ground Truth Legal Basis") or "").strip()
        if not query or not ground_truth or ground_truth.lower() == 'nan':
            continue
        total_queries += 1
        norm_query = normalize_text(query)

        # ---- BM25 ----
        bm25_scores = np.array(bm25.get_scores(multigram_tokenizer(norm_query)), dtype=float)

        # ---- BGE-M3 dense + sparse + colbert (query side) ----
        q_out = bge.encode(
            [norm_query], return_dense=True, return_sparse=True, return_colbert_vecs=True
        )
        q_dense = q_out["dense_vecs"][0]
        q_sparse = q_out["lexical_weights"][0]
        q_colbert = q_out["colbert_vecs"][0]

        dense_scores = doc_dense @ q_dense
        sparse_scores = np.array([
            bge.compute_lexical_matching_score(q_sparse, doc_sparse[i])
            for i in range(len(documents))
        ])

        # ---- Dynamic alpha: section-explicit queries lean lexical ----
        explicit = is_section_explicit_query(query)
        if explicit:
            # count lexical signals (BM25 + sparse) twice, dense once
            hybrid_scores = reciprocal_rank_fusion(
                [bm25_scores, sparse_scores, dense_scores], weights=[2, 2, 1]
            )
        else:
            # count dense twice, lexical once each -- conceptual query
            hybrid_scores = reciprocal_rank_fusion(
                [bm25_scores, sparse_scores, dense_scores], weights=[1, 1, 2]
            )

        # ---- Section boosting ----
        q_sections = extract_section_identifiers(query)
        if q_sections:
            for i, secs in enumerate(doc_sections):
                if q_sections & secs:
                    hybrid_scores[i] += SECTION_BOOST

        top_idx = np.argsort(hybrid_scores)[::-1][:CANDIDATE_POOL].tolist()

        # ---- ColBERT rescoring on the top slice ----
        colbert_slice = top_idx[:COLBERT_RESCORE_POOL]
        colbert_scores = np.array([
            bge.colbert_score(q_colbert, doc_colbert[i]) for i in colbert_slice
        ])
        if colbert_scores.max() > colbert_scores.min():
            cb_norm = (colbert_scores - colbert_scores.min()) / (colbert_scores.max() - colbert_scores.min())
        else:
            cb_norm = np.zeros_like(colbert_scores)
        for j, doc_i in enumerate(colbert_slice):
            hybrid_scores[doc_i] += 0.3 * cb_norm[j]

        top_idx = sorted(top_idx, key=lambda i: hybrid_scores[i], reverse=True)
        hybrid_top_scores = np.array([hybrid_scores[i] for i in top_idx])
        if hybrid_top_scores.max() > hybrid_top_scores.min():
            hybrid_top_norm = (hybrid_top_scores - hybrid_top_scores.min()) / (
                hybrid_top_scores.max() - hybrid_top_scores.min()
            )
        else:
            hybrid_top_norm = np.zeros_like(hybrid_top_scores)

        # ---- Cross-encoder rerank ----
        rerank_pairs = [[query, enriched_texts[i]] for i in top_idx]
        rerank_scores = np.array(reranker.predict(rerank_pairs), dtype=float)
        if rerank_scores.max() > rerank_scores.min():
            rerank_norm = (rerank_scores - rerank_scores.min()) / (rerank_scores.max() - rerank_scores.min())
        else:
            rerank_norm = np.zeros_like(rerank_scores)

        # ---- Late fusion ----
        final_scores = RERANK_WEIGHT * rerank_norm + HYBRID_WEIGHT * hybrid_top_norm
        reranked = sorted(zip(final_scores, top_idx), key=lambda x: x[0], reverse=True)
        final_top_5_idx = [doc_idx for _, doc_idx in reranked[:5]]

        # ---- Non-leaky section fallback (query-only, never ground truth) ----
        if q_sections:
            exact_match_indices = [
                i for i, secs in enumerate(doc_sections) if q_sections & secs
            ]
            candidates = [i for i in exact_match_indices if i not in final_top_5_idx]
            if candidates:
                # pick the candidate with the best hybrid score, not just the first
                candidates.sort(key=lambda i: hybrid_scores[i], reverse=True)
                final_top_5_idx[-1] = candidates[0]

        retrieved_docs = [documents[i] for i in final_top_5_idx]

        # ---- Evaluation ----
        gt_sections = extract_section_identifiers(ground_truth)
        matches = []
        for doc in retrieved_docs:
            all_doc_secs = get_doc_sections(doc)
            matches.append(1 if gt_sections & all_doc_secs else 0)

        if sum(matches[:1]) > 0: hit_at_1 += 1
        if sum(matches[:3]) > 0: hit_at_3 += 1
        if sum(matches[:5]) > 0: hit_at_5 += 1

        first_hit_rank = 0
        for r_idx, is_match in enumerate(matches):
            if is_match == 1:
                first_hit_rank = r_idx + 1
                break
        if first_hit_rank > 0:
            mrr_score += 1.0 / first_hit_rank

    print("\n" + "=" * 70)
    print("FINAL COMBINED PIPELINE - EVALUATION RESULT")
    print("=" * 70)
    if total_queries > 0:
        print(f"Total Evaluated Queries : {total_queries}")
        print("-" * 70)
        print(f"Hit@1  Accuracy          : {(hit_at_1 / total_queries) * 100:.2f}%")
        print(f"Hit@3  Accuracy          : {(hit_at_3 / total_queries) * 100:.2f}%")
        print(f"Hit@5  Accuracy          : {(hit_at_5 / total_queries) * 100:.2f}%")
        print("-" * 70)
        print(f"MRR (Mean Reciprocal Rank): {(mrr_score / total_queries):.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
