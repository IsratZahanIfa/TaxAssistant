import pickle
import re
import torch
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"
PKL_PATH = r"E:\TaxAssist\taxdatabase\tax_data_parent_child.pkl"
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/17ew_umfBKtAeTUEBFtBknNoyQR5KbWRSgceh47DNeYY/export?format=csv&gid=0"
TOP_K_RETRIEVE = 200
TOP_K_RERANK = 5
JACCARD_THRESHOLD = 0.4

# Bengali digit conversion
BN_TO_EN_DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')

def normalize_text(text):
    if not text or pd.isna(text):
        return ""
    return str(text).lower().translate(BN_TO_EN_DIGITS)

def extract_section_identifiers(text):
    text = normalize_text(text)
    raw_sections = re.findall(r'(?:section|sec|ধারা)?\s*(\d+[a-z]?)', text)
    cleaned_sections = set()
    for sec in raw_sections:
        sec_clean = sec.strip()
        if sec_clean and len(re.sub(r'\D', '', sec_clean)) <= 3:
            cleaned_sections.add(sec_clean)
    return cleaned_sections

def is_match_with_ground_truth(doc_text: str, ground_truth: str) -> bool:
    """Match by section numbers first, then Jaccard similarity."""
    # Section overlap
    doc_secs = extract_section_identifiers(doc_text)
    gt_secs = extract_section_identifiers(ground_truth)
    if gt_secs and doc_secs.intersection(gt_secs):
        return True
    # Jaccard similarity fallback
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
# Ollama Helper (for HyDE)
# ============================================================
def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 200}
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            print(f"Ollama error: {response.status_code}")
            return ""
    except Exception as e:
        print(f"Ollama request failed: {e}")
        return ""

# ============================================================
# HyDE Retriever
# ============================================================
class HyDERetriever:
    def __init__(self, documents: List[Dict], doc_embeddings: np.ndarray, dense_model, device: str):
        self.documents = documents
        self.doc_embeddings = doc_embeddings
        self.dense_model = dense_model
        self.device = device

    def _generate_hypothetical_doc(self, query: str) -> str:
        prompt = (
            f"You are an expert on the Bangladesh Income Tax Act. "
            f"Write a short paragraph that would contain the answer to the following query. "
            f"Only include factual information from the act.\n\n"
            f"Query: {query}\n\n"
            f"Hypothetical document:"
        )
        hypo = call_ollama(prompt)
        if hypo and len(hypo) > 10:
            return hypo
        # Fallback (if Ollama fails)
        return f"This document provides information about {query} in the context of the Bangladesh Income Tax Act."

    def retrieve(self, query: str, top_k: int) -> Tuple[List[int], List[float]]:
        hypo = self._generate_hypothetical_doc(query)
        q_emb = self.dense_model.encode([hypo], normalize_embeddings=True, convert_to_numpy=True)
        scores = np.dot(self.doc_embeddings, q_emb.T).squeeze()
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        return top_indices.tolist(), top_scores.tolist()

# ============================================================
# Main – evaluate all queries
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running HyDE Evaluation (no rewrite) on Device: {device.upper()}")

    # Load documents
    with open(PKL_PATH, "rb") as f:
        documents = pickle.load(f)

    # Enrich texts
    enriched_texts = []
    for d in documents:
        p_sec = d.get('parent_sec', '')
        p_title = d.get('parent_title', '')
        c_text = d.get('child_text', '')
        full_context = f"Section: {p_sec}. Title: {p_title}. Text: {c_text}"
        enriched_texts.append(full_context)

    # Dense model
    print("Loading Dense Model (BAAI/bge-m3)...")
    dense_model = SentenceTransformer("BAAI/bge-m3", device=device)
    print("Encoding Documents...")
    doc_embeddings = dense_model.encode(
        enriched_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # Reranker
    print("Loading Cross-Encoder (BAAI/bge-reranker-v2-m3)...")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)

    # HyDE retriever
    hyde = HyDERetriever(documents, doc_embeddings, dense_model, device)

    # Load evaluation dataset
    print("Loading query dataset...")
    try:
        df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if "Question" not in df.columns or "Ground Truth Legal Basis" not in df.columns:
        raise ValueError("CSV must contain 'Question' and 'Ground Truth Legal Basis' columns.")

    # Prepare results storage
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

        # 1. Retrieve + rerank
        cand_indices, _ = hyde.retrieve(query, top_k=TOP_K_RETRIEVE)
        rerank_pairs = [[query, enriched_texts[i]] for i in cand_indices]
        rerank_scores = reranker.predict(rerank_pairs, batch_size=32)
        sorted_tuples = sorted(zip(rerank_scores, cand_indices), key=lambda x: x[0], reverse=True)
        top5_indices = [doc_idx for _, doc_idx in sorted_tuples[:TOP_K_RERANK]]
        top5_scores = [score for score, _ in sorted_tuples[:TOP_K_RERANK]]

        # 2. Evaluate matches
        retrieved_docs = [documents[i] for i in top5_indices]
        matches = []
        for doc in retrieved_docs:
            doc_text = doc.get('parent_sec', '') + " " + doc.get('child_text', '')
            is_match = 1 if is_match_with_ground_truth(doc_text, ground_truth) else 0
            matches.append(is_match)

        # Update metrics
        if sum(matches[:1]) > 0:
            hit_at_1 += 1
        if sum(matches[:3]) > 0:
            hit_at_3 += 1
        if sum(matches[:5]) > 0:
            hit_at_5 += 1

        first_hit_rank = 0
        for r_idx, is_match in enumerate(matches):
            if is_match == 1:
                first_hit_rank = r_idx + 1
                break
        if first_hit_rank > 0:
            mrr_score += 1.0 / first_hit_rank

        # Store details
        details_records.append({
            "Query": query,
            "Ground_Truth": ground_truth,
            "Top1_Doc": retrieved_docs[0].get('parent_sec', ''),
            "Top1_Score": top5_scores[0] if len(top5_scores) > 0 else 0,
            "Top2_Doc": retrieved_docs[1].get('parent_sec', '') if len(retrieved_docs) > 1 else "",
            "Top2_Score": top5_scores[1] if len(top5_scores) > 1 else 0,
            "Top3_Doc": retrieved_docs[2].get('parent_sec', '') if len(retrieved_docs) > 2 else "",
            "Top3_Score": top5_scores[2] if len(top5_scores) > 2 else 0,
            "Top4_Doc": retrieved_docs[3].get('parent_sec', '') if len(retrieved_docs) > 3 else "",
            "Top4_Score": top5_scores[3] if len(top5_scores) > 3 else 0,
            "Top5_Doc": retrieved_docs[4].get('parent_sec', '') if len(retrieved_docs) > 4 else "",
            "Top5_Score": top5_scores[4] if len(top5_scores) > 4 else 0,
            "Hit@1": matches[0] if len(matches) > 0 else 0,
            "Hit@3": 1 if sum(matches[:3]) > 0 else 0,
            "Hit@5": 1 if sum(matches[:5]) > 0 else 0,
            "First_Hit_Rank": first_hit_rank
        })

    # ---- Write Excel ----
    excel_path = "hyde_evaluation_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        details_df = pd.DataFrame(details_records)
        details_df.to_excel(writer, sheet_name="Details", index=False)

        summary_data = {
            "Metric": ["Total Queries", "Hit@1 Accuracy", "Hit@3 Accuracy", "Hit@5 Accuracy", "MRR"],
            "Value": [
                total_queries,
                f"{(hit_at_1 / total_queries) * 100:.2f}%" if total_queries > 0 else "0%",
                f"{(hit_at_3 / total_queries) * 100:.2f}%" if total_queries > 0 else "0%",
                f"{(hit_at_5 / total_queries) * 100:.2f}%" if total_queries > 0 else "0%",
                f"{(mrr_score / total_queries):.4f}" if total_queries > 0 else "0.0000"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"\nResults saved to {excel_path}")

    # Console summary
    print("\n" + "=" * 70)
    print("HYDE EVALUATION SUMMARY (No Rewrite)")
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