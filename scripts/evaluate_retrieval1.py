import json
import re
import gc
import torch
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG – tweak these for your GPU
# ============================================================
JSON_PATH = Path("data/Knowledge_Dataset/income_tax_act_2023.json")
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/17ew_umfBKtAeTUEBFtBknNoyQR5KbWRSgceh47DNeYY/export?format=csv&gid=0"

TOP_K_RETRIEVE = 100         # BM25 & Dense থেকে আলাদাভাবে কয়টি করে নেয়া হবে
TOP_K_RERANK = 60            # RRF প্রয়োগের পর Reranker-এ সর্বোচ্চ যতটি যাবে (Speed & Memory safe)
TOP_K_FINAL = 5              # Final Top-5 Output
MAX_LENGTH = 512             # BGE-M3 sequence length limit
BATCH_SIZE_ENCODE = 32       # Batch size for document encoding
BATCH_SIZE_RERANK = 8        # Batch size for cross-encoder reranking
JACCARD_THRESHOLD = 0.3      # Ground truth matching threshold
RRF_K = 60                   # Reciprocal Rank Fusion constant

# ============================================================
# TEXT & METADATA HELPERS
# ============================================================
BN_TO_EN_DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
EN_TO_BN_DIGITS = str.maketrans('0123456789', '০১২৩৪৫৬৭৮৯')

def normalize_text(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    return str(text).lower().translate(BN_TO_EN_DIGITS)

def extract_section_identifiers(text: str) -> set:
    text = normalize_text(text)
    raw_sections = re.findall(r'(?:section|sec|ধারা)?\s*(\d+[a-z]?)', text)
    cleaned = set()
    for s in raw_sections:
        s = s.strip()
        if s and len(re.sub(r'\D', '', s)) <= 3:
            cleaned.add(s)
    return cleaned

def preprocess_query(query: str) -> str:
    """Query expansion and domain specific normalization."""
    query = normalize_text(query)
    # Acronym extensions
    query = re.sub(r'\btin\b', 'tin টিআইএন কর শনাক্তকরণ নম্বর', query)
    query = re.sub(r'\btds\b', 'tds উৎস কর উৎসে কর কর্তন', query)
    
    # Extract sections and append both BN and EN section terms
    secs = extract_section_identifiers(query)
    if secs:
        en_secs = " ".join([f"section {s}" for s in secs])
        bn_secs = " ".join([f"ধারা {s.translate(EN_TO_BN_DIGITS)}" for s in secs])
        query += f" {en_secs} {bn_secs}"
        
    return query

def tokenize_for_bm25(text: str) -> List[str]:
    """Simple word tokenization for BM25 search."""
    clean_text = normalize_text(text)
    return re.findall(r'\w+', clean_text)

def is_match_with_ground_truth(doc_text: str, ground_truth: str) -> bool:
    doc_secs = extract_section_identifiers(doc_text)
    gt_secs = extract_section_identifiers(ground_truth)
    if gt_secs and doc_secs.intersection(gt_secs):
        return True
    doc_clean = normalize_text(doc_text)
    gt_clean = normalize_text(ground_truth)
    if not gt_clean:
        return False
    doc_words = set(doc_clean.split())
    gt_words = set(gt_clean.split())
    if not doc_words or not gt_words:
        return False
    intersection = len(doc_words & gt_words)
    union = len(doc_words | gt_words)
    jaccard = intersection / union if union > 0 else 0.0
    return jaccard > JACCARD_THRESHOLD

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Optimized Hybrid Retrieval Pipeline running on {device.upper()}")

    # ---- 1. Load JSON Data ----
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON not found: {JSON_PATH}")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if not isinstance(raw_data, list):
        raise TypeError("JSON must be a list of documents.")

    documents = []
    for item in raw_data:
        sec = str(item.get("section", ""))
        title = str(item.get("title", ""))
        text = str(item.get("text", ""))
        if not text.strip():
            continue
        documents.append({
            "parent_sec": sec,
            "parent_title": title,
            "child_text": text,
            "text": text,
            "page": item.get("page", ""),
        })
    print(f"Loaded {len(documents)} documents")

    # ---- 2. Build Rich Enriched Texts (Dual Language Metadata) ----
    enriched_texts = []
    for d in documents:
        sec_en = normalize_text(d['parent_sec'])
        sec_bn = sec_en.translate(EN_TO_BN_DIGITS)
        title = d['parent_title']
        body = d['child_text']
        
        # Dual-language header boosting exact section matching
        rich_header = f"ধারা: {sec_bn} | Section: {sec_en} | শিরোনাম: {title}"
        enriched_texts.append(f"{rich_header}\n{body}")

    # ---- 3. Build BM25 Index ----
    print("Building BM25 Index...")
    corpus_tokens = [tokenize_for_bm25(doc) for doc in enriched_texts]
    bm25 = BM25Okapi(corpus_tokens)

    # ---- 4. Dense Model (BGE-M3) Encoding ----
    print("Loading BGE-M3 Dense Model...")
    dense_model = SentenceTransformer("BAAI/bge-m3", device=device)
    dense_model.max_seq_length = MAX_LENGTH

    print("Encoding documents into dense vectors...")
    doc_embeddings = dense_model.encode(
        enriched_texts,
        batch_size=BATCH_SIZE_ENCODE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    print(f"Encoded {doc_embeddings.shape[0]} docs, Dim={doc_embeddings.shape[1]}")

    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    # ---- 5. Load Reranker Model ----
    print("Loading BGE Cross-Encoder Reranker...")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)

    # ---- 6. Load Dataset ----
    print("Loading Evaluation Queries...")
    try:
        df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    if "Question" not in df.columns or "Ground Truth Legal Basis" not in df.columns:
        raise ValueError("CSV must contain 'Question' and 'Ground Truth Legal Basis' columns.")

    # ---- 7. Hybrid Retrieval & Pure Cross-Encoder Rerank ----
    def hybrid_retrieve_and_rerank(query: str):
        processed_q = preprocess_query(query)
        
        # A. Lexical Search (BM25)
        q_tokens = tokenize_for_bm25(processed_q)
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:TOP_K_RETRIEVE]
        
        # B. Dense Search (BGE-M3)
        q_emb = dense_model.encode([processed_q], normalize_embeddings=True, convert_to_numpy=True)
        dense_scores = np.dot(doc_embeddings, q_emb.T).squeeze()
        dense_top_idx = np.argsort(dense_scores)[::-1][:TOP_K_RETRIEVE]

        # C. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, idx in enumerate(bm25_top_idx):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (RRF_K + rank + 1))
            
        for rank, idx in enumerate(dense_top_idx):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (RRF_K + rank + 1))

        # Sort combined top candidates for Reranking
        sorted_rrf_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_indices = [idx for idx, _ in sorted_rrf_candidates[:TOP_K_RERANK]]

        # D. Cross-Encoder Rerank on Original Query
        rerank_pairs = [[query, enriched_texts[idx]] for idx in candidate_indices]
        rerank_scores = reranker.predict(rerank_pairs, batch_size=BATCH_SIZE_RERANK)

        # Sort purely by Cross-Encoder Confidence
        final_sorted_idx = np.argsort(rerank_scores)[::-1]
        final_indices = [candidate_indices[i] for i in final_sorted_idx[:TOP_K_FINAL]]
        final_scores_list = [float(rerank_scores[i]) for i in final_sorted_idx[:TOP_K_FINAL]]

        return final_indices, final_scores_list

    # ---- 8. Evaluation Loop ----
    details_records = []
    hit_at_1, hit_at_3, hit_at_5 = 0, 0, 0
    mrr_score = 0.0
    total_queries = 0

    print("\nProcessing queries...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        query = str(row["Question"]).strip()
        ground_truth = str(row.get("Ground Truth Legal Basis", "")).strip()
        if not query or not ground_truth or ground_truth.lower() == 'nan':
            continue

        total_queries += 1

        # Retrieve and Rerank
        final_indices, final_scores = hybrid_retrieve_and_rerank(query)
        retrieved_docs = [documents[i] for i in final_indices]

        # Check Ground Truth Matches
        matches = []
        for doc in retrieved_docs:
            doc_text = doc.get('parent_sec', '') + " " + doc.get('child_text', '')
            matches.append(1 if is_match_with_ground_truth(doc_text, ground_truth) else 0)

        if sum(matches[:1]) > 0: hit_at_1 += 1
        if sum(matches[:3]) > 0: hit_at_3 += 1
        if sum(matches[:5]) > 0: hit_at_5 += 1

        first_hit_rank = 0
        for r_idx, m in enumerate(matches):
            if m == 1:
                first_hit_rank = r_idx + 1
                break
        if first_hit_rank > 0:
            mrr_score += 1.0 / first_hit_rank

        # Store Results
        details_records.append({
            "Query": query,
            "Ground_Truth": ground_truth,
            "Top1_Doc": retrieved_docs[0].get('parent_sec', '') if len(retrieved_docs)>0 else "",
            "Top1_Score": final_scores[0] if len(final_scores)>0 else 0,
            "Top2_Doc": retrieved_docs[1].get('parent_sec', '') if len(retrieved_docs)>1 else "",
            "Top2_Score": final_scores[1] if len(final_scores)>1 else 0,
            "Top3_Doc": retrieved_docs[2].get('parent_sec', '') if len(retrieved_docs)>2 else "",
            "Top3_Score": final_scores[2] if len(final_scores)>2 else 0,
            "Top4_Doc": retrieved_docs[3].get('parent_sec', '') if len(retrieved_docs)>3 else "",
            "Top4_Score": final_scores[3] if len(final_scores)>3 else 0,
            "Top5_Doc": retrieved_docs[4].get('parent_sec', '') if len(retrieved_docs)>4 else "",
            "Top5_Score": final_scores[4] if len(final_scores)>4 else 0,
            "Hit@1": matches[0] if len(matches)>0 else 0,
            "Hit@3": 1 if sum(matches[:3])>0 else 0,
            "Hit@5": 1 if sum(matches[:5])>0 else 0,
            "First_Hit_Rank": first_hit_rank
        })

        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- 9. Export Results to Excel ----
    excel_path = "hybrid_evaluation_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(details_records).to_excel(writer, sheet_name="Details", index=False)
        summary = {
            "Metric": ["Total Queries", "Hit@1", "Hit@3", "Hit@5", "MRR"],
            "Value": [
                total_queries,
                f"{(hit_at_1/total_queries)*100:.2f}%" if total_queries else "0%",
                f"{(hit_at_3/total_queries)*100:.2f}%" if total_queries else "0%",
                f"{(hit_at_5/total_queries)*100:.2f}%" if total_queries else "0%",
                f"{(mrr_score/total_queries):.4f}" if total_queries else "0.0000"
            ]
        }
        pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)

    print(f"\n✅ Results saved to {excel_path}")

    # ---- 10. Summary Display ----
    print("\n" + "="*70)
    print("HYBRID (BM25 + DENSE) RETRIEVAL EVALUATION SUMMARY")
    print("="*70)
    if total_queries > 0:
        print(f"Total Evaluated Queries : {total_queries}")
        print("-"*70)
        print(f"Hit@1  Accuracy : {(hit_at_1/total_queries)*100:.2f}%")
        print(f"Hit@3  Accuracy : {(hit_at_3/total_queries)*100:.2f}%")
        print(f"Hit@5  Accuracy : {(hit_at_5/total_queries)*100:.2f}%")
        print("-"*70)
        print(f"MRR : {(mrr_score/total_queries):.4f}")
    print("="*70)

if __name__ == "__main__":
    main()