import json
import re
import gc
import torch
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Tuple
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"


JSON_PATH = Path("data/Knowledge_Dataset/income_tax_act_2023.json")
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/17ew_umfBKtAeTUEBFtBknNoyQR5KbWRSgceh47DNeYY/export?format=csv&gid=0"


TOP_K_RETRIEVE = 140          
TOP_K_RERANK = 100           
TOP_K_FINAL = 5              
MAX_LENGTH = 512              
BATCH_SIZE_ENCODE = 16        
BATCH_SIZE_RERANK = 4        
JACCARD_THRESHOLD = 0.3       


BN_TO_EN_DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')


def normalize_text(text):
    if not text or pd.isna(text):
        return ""
    return str(text).lower().translate(BN_TO_EN_DIGITS)


def extract_section_identifiers(text):
    text = normalize_text(text)
    raw_sections = re.findall(r'(?:section|sec|ধারা)?\s*(\d+[a-z]?)', text)
    cleaned = set()
    for s in raw_sections:
        s = s.strip()
        if s and len(re.sub(r'\D', '', s)) <= 3:
            cleaned.add(s)
    return cleaned


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


def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 200}   # shorter HyDE output
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


def generate_hypothetical_doc(query: str) -> str:
    prompt = (
        f"You are a legal expert on the Bangladesh Income Tax Act. "
        f"Write a short paragraph that answers the following query, citing specific sections if possible.\n\n"
        f"Query: {query}\n\n"
        f"Answer:"
    )
    hypo = call_ollama(prompt)
    if hypo and len(hypo) > 10:
        return hypo
    return f"This document addresses: {query} under the Bangladesh Income Tax Act."

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 HyDE Memory‑Safe on {device.upper()}")


    # ---- 1. Load JSON ----
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON not found: {JSON_PATH}")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if not isinstance(raw_data, list):
        raise TypeError("JSON must be a list.")

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


    # ---- 2. Build enriched texts (for retrieval) ----
    enriched_texts = []
    for d in documents:
        enriched_texts.append(
            f"Section: {d['parent_sec']}. Title: {d['parent_title']}. Text: {d['child_text']}"
        )

    print("Loading BGE-M3 (max_length=512)...")
    dense_model = SentenceTransformer("BAAI/bge-m3", device=device)
    dense_model.max_seq_length = MAX_LENGTH


    print("Encoding documents...")
    doc_embeddings = dense_model.encode(
        enriched_texts,
        batch_size=BATCH_SIZE_ENCODE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,         
    )
    print(f"Encoded {doc_embeddings.shape[0]} docs, dim={doc_embeddings.shape[1]}")


    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()


    print("Loading BGE-reranker-v2-m3 (will be used on top candidates)...")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)

    print("Loading query dataset...")
    try:
        df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    if "Question" not in df.columns or "Ground Truth Legal Basis" not in df.columns:
        raise ValueError("CSV must contain 'Question' and 'Ground Truth Legal Basis' columns.")

    def retrieve_and_rerank(query: str):
        # HyDE
        hypo = generate_hypothetical_doc(query)
        q_emb = dense_model.encode(
            [hypo],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        scores = np.dot(doc_embeddings, q_emb.T).squeeze()
        top_indices = np.argsort(scores)[::-1][:TOP_K_RETRIEVE]
        top_scores = scores[top_indices]
        rerank_indices = top_indices[:TOP_K_RERANK]
        rerank_pairs = [[query, enriched_texts[i]] for i in rerank_indices]
        rerank_scores = reranker.predict(rerank_pairs, batch_size=BATCH_SIZE_RERANK)


        hyde_sub = top_scores[:TOP_K_RERANK]
        if len(hyde_sub) == 1:
            hyde_norm = np.array([1.0])
        else:
            hyde_norm = (hyde_sub - hyde_sub.min()) / (hyde_sub.max() - hyde_sub.min() + 1e-8)
        if len(rerank_scores) == 1:
            rerank_norm = np.array([1.0])
        else:
            rerank_norm = (rerank_scores - rerank_scores.min()) / (rerank_scores.max() - rerank_scores.min() + 1e-8)


        final_scores = 0.7 * rerank_norm + 0.3 * hyde_norm
        sorted_idx = np.argsort(final_scores)[::-1]
        final_indices = [rerank_indices[i] for i in sorted_idx[:TOP_K_FINAL]]
        final_scores_list = [float(final_scores[i]) for i in sorted_idx[:TOP_K_FINAL]]
        return final_indices, final_scores_list

    details_records = []
    hit_at_1, hit_at_3, hit_at_5 = 0, 0, 0
    mrr_score = 0.0
    total_queries = 0


    print("\nProcessing queries (memory‑safe)...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        query = str(row["Question"]).strip()
        ground_truth = str(row.get("Ground Truth Legal Basis", "")).strip()
        if not query or not ground_truth or ground_truth.lower() == 'nan':
            continue


        total_queries += 1
        gt_sections = extract_section_identifiers(ground_truth)


        # Retrieve
        final_indices, final_scores = retrieve_and_rerank(query)
        retrieved_docs = [documents[i] for i in final_indices]


        # Evaluate matches
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


        # Store
        details_records.append({
            "Query": query,
            "Ground_Truth": ground_truth,
            "Top1_Doc": retrieved_docs[0].get('parent_sec', ''),
            "Top1_Score": final_scores[0] if final_scores else 0,
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


        # Clear cache after each query
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()


    # ---- 8. Excel export ----
    excel_path = "hyde_memory_safe_results.xlsx"
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


    print("HYDE MEMORY-SAFE EVALUATION SUMMARY")

    if total_queries > 0:
        print(f"Total Evaluated Queries : {total_queries}")
        print("-"*70)
        print(f"Hit@1  Accuracy : {(hit_at_1/total_queries)*100:.2f}%")
        print(f"Hit@3  Accuracy : {(hit_at_3/total_queries)*100:.2f}%")
        print(f"Hit@5  Accuracy : {(hit_at_5/total_queries)*100:.2f}%")
        print("-"*70)
        print(f"MRR : {(mrr_score/total_queries):.4f}")
   


if __name__ == "__main__":
    main()
