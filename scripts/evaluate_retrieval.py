"""
Retrieval ablation + diagnostics for the Bangladesh Income Tax RAG.

Runs ALL variants on the SAME queries with the SAME strict section matching:
  1. BM25 only
  2. Dense only
  3. Hybrid (BM25 + Dense)
  4. HyDE dense only
  5. Dense + Rerank
  6. Hybrid + Rerank
  7. HyDE + Rerank
  8. HyDE + Rerank + fusion (0.7 rerank + 0.3 HyDE)   <- original pipeline

Diagnostics:
  - Corpus coverage: are the ground-truth sections even present in the corpus?
  - Recall@10/20/50/100 for every variant (is the bottleneck first stage or reranker?)
  - Failure categories + "near-miss" flag (top-1 section number close to the GT section)
  - Results also reported on the subset of queries whose GT section exists in the corpus

HyDE paragraphs are cached in hyde_cache.json, so re-runs are fast and an interrupted
run can resume.
"""
import json
import re
import gc
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# =========================== CONFIG ===========================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"

JSON_PATH = Path("data/Knowledge_Dataset/income_tax_act_2023.json")
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/17ew_umfBKtAeTUEBFtBknNoyQR5KbWRSgceh47DNeYY/export?format=csv&gid=0"

HYDE_CACHE_PATH = Path("hyde_cache.json")
OUTPUT_XLSX = "retrieval_ablation_results.xlsx"

RUN_HYDE = True               # False = skip all HyDE variants (no Ollama needed)
RERANK_N = 50                 # candidates sent to the reranker for every rerank variant
MAX_LENGTH = 512
BATCH_SIZE_ENCODE = 16
BATCH_SIZE_RERANK = 8

K_VALUES = [1, 3, 5]          # Hit / Precision / Recall / F1 / nDCG / MAP at these K
RECALL_KS = [10, 20, 50, 100] # extra Recall@N to diagnose the first stage
MAX_EVAL = max(K_VALUES + RECALL_KS)

W_BM25, W_DENSE = 0.5, 0.5    # hybrid fusion
W_RERANK, W_HYDE = 0.7, 0.3   # HyDE + rerank fusion

FAILURE_VARIANT = "HyDE + Rerank + fusion"   # variant used for the failure-category table

BN_TO_EN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

V_BM25 = "BM25 only"
V_DENSE = "Dense only"
V_HYB = "Hybrid (BM25+Dense)"
V_HYDE = "HyDE dense only"
V_DR = "Dense + Rerank"
V_HR = "Hybrid + Rerank"
V_HYR = "HyDE + Rerank"
V_HYRF = "HyDE + Rerank + fusion"
HYDE_VARIANTS = {V_HYDE, V_HYR, V_HYRF}


# =========================== TEXT / SECTION MATCHING ===========================
def normalize_text(text):
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return ""
    return str(text).lower().translate(BN_TO_EN_DIGITS)


def simple_tokenizer(text):
    return re.findall(r"[\w]+", normalize_text(text))


# Strict: only numbers right after "section / sec / ধারা" (lists like "Section 44, 45 and 46" supported)
_SEC_PATTERN = re.compile(
    r"(?:sections?|sec\.?|ধারা)\s*"
    r"(\d+[a-z]?(?:\s*(?:,|and|&|ও|এবং)\s*\d+[a-z]?)*)"
)
_ID_PATTERN = re.compile(r"\d+[a-z]?")


def extract_gt_sections(text) -> set:
    text = normalize_text(text)
    ids = set()
    for group in _SEC_PATTERN.findall(text):
        ids.update(_ID_PATTERN.findall(group))
    return ids


def extract_doc_sections(parent_sec) -> set:
    """Section ids from the document's section field only (body text has cross-references)."""
    ids = set(_ID_PATTERN.findall(normalize_text(parent_sec)))
    return {i for i in ids if len(re.sub(r"\D", "", i)) <= 3}


def section_numbers(sec_set) -> set:
    nums = set()
    for s in sec_set:
        d = re.sub(r"\D", "", s)
        if d:
            nums.add(int(d))
    return nums


# =========================== METRICS ===========================
def eval_ranking(ranking, gt_sections, doc_secs):
    """
    matches : 1 if chunk is relevant (raw)                      -> Precision, Hit, MRR
    gains   : 1 only the FIRST time a GT section is covered     -> Recall, nDCG, AP
    """
    top = ranking[:MAX_EVAL]
    matches, gains, covered = [], [], set()
    for i in top:
        m = doc_secs[i] & gt_sections
        matches.append(1 if m else 0)
        new = m - covered
        gains.append(1 if new else 0)
        covered |= m

    n_gt = len(gt_sections)
    res = {}
    for k in K_VALUES:
        m, g = matches[:k], gains[:k]
        precision = sum(m) / k
        recall = sum(g) / n_gt
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        dcg = sum(x / math.log2(i + 2) for i, x in enumerate(g))
        ideal_n = min(k, n_gt)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        hits, ap_sum = 0, 0.0
        for i, x in enumerate(g):
            if x:
                hits += 1
                ap_sum += hits / (i + 1)
        ap = ap_sum / ideal_n if ideal_n > 0 else 0.0

        res[f"Hit@{k}"] = 1 if sum(m) > 0 else 0
        res[f"Precision@{k}"] = precision
        res[f"Recall@{k}"] = recall
        res[f"F1@{k}"] = f1
        res[f"nDCG@{k}"] = ndcg
        res[f"MAP@{k}"] = ap

    for n in RECALL_KS:
        res[f"Recall@{n}"] = sum(gains[:n]) / n_gt

    first = next((i + 1 for i, m in enumerate(matches) if m), 0)
    res["First_Rank"] = first
    res["MRR"] = (1.0 / first) if (0 < first <= max(K_VALUES)) else 0.0   # MRR over top-5, same as before
    return res


def failure_category(first_rank, in_corpus):
    if not in_corpus:
        return "GT section not in corpus"
    if first_rank == 1:
        return "Correct @1"
    if 2 <= first_rank <= 5:
        return "Found in top-5, not first"
    if 6 <= first_rank <= RERANK_N:
        return f"Found rank 6-{RERANK_N}"
    if first_rank > RERANK_N:
        return f"Found rank {RERANK_N + 1}-{MAX_EVAL}"
    return f"Not in top-{MAX_EVAL}"


# =========================== OLLAMA / HyDE ===========================
def call_ollama(prompt: str) -> str:
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 200},
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        print(f"Ollama error: {r.status_code}")
    except Exception as e:
        print(f"Ollama request failed: {e}")
    return ""


def hyde_prompt(query: str) -> str:
    return (
        "You are a legal expert on the Bangladesh Income Tax Act. "
        "Write a short paragraph that answers the following query, citing specific sections if possible.\n\n"
        f"Query: {query}\n\nAnswer:"
    )


def load_hyde_cache() -> dict:
    if HYDE_CACHE_PATH.exists():
        with open(HYDE_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_hyde_cache(cache: dict):
    with open(HYDE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


# =========================== SMALL HELPERS ===========================
def minmax(x):
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return np.ones_like(x)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def rank_desc(scores):
    return np.argsort(-np.asarray(scores)).tolist()


def merge_rank(reranked_idx, first_stage_rank):
    """Reranked candidates first, then the remaining docs in first-stage order."""
    seen = set(reranked_idx)
    return list(reranked_idx) + [i for i in first_stage_rank if i not in seen]


# =========================== MAIN ===========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Retrieval ablation on {device.upper()}")

    # ---- Load corpus ----
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON not found: {JSON_PATH}")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    documents = []
    for item in raw:
        text = str(item.get("text", ""))
        if not text.strip():
            continue
        documents.append({
            "parent_sec": str(item.get("section", "")),
            "parent_title": str(item.get("title", "")),
            "child_text": text,
        })
    enriched = [f"Section: {d['parent_sec']}. Title: {d['parent_title']}. Text: {d['child_text']}" for d in documents]
    doc_secs = [extract_doc_sections(d["parent_sec"]) for d in documents]
    corpus_secs = set().union(*doc_secs) if doc_secs else set()
    print(f"Loaded {len(documents)} documents; {len(corpus_secs)} distinct section ids in corpus")

    empty_sec_docs = [i for i, s in enumerate(doc_secs) if not s]
    print(f"Docs with NO parseable section id in 'section' field: {len(empty_sec_docs)}")
    for i in empty_sec_docs[:5]:
        print(f"   e.g. section field = {documents[i]['parent_sec']!r}")

    # ---- Load queries ----
    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    if "Question" not in df.columns or "Ground Truth Legal Basis" not in df.columns:
        raise ValueError("CSV must contain 'Question' and 'Ground Truth Legal Basis' columns.")

    queries = []
    skipped_no_section = 0
    for _, row in df.iterrows():
        q = str(row["Question"]).strip()
        gt = str(row.get("Ground Truth Legal Basis", "")).strip()
        if not q or not gt or gt.lower() == "nan":
            continue
        gt_secs = extract_gt_sections(gt)
        if not gt_secs:
            skipped_no_section += 1
            continue
        queries.append({"query": q, "gt_text": gt, "gt_secs": gt_secs})
    print(f"Queries to evaluate: {len(queries)} (skipped {skipped_no_section} without section id)")

    # ---- Coverage diagnostic ----
    all_gt = set().union(*[q["gt_secs"] for q in queries])
    missing = sorted(all_gt - corpus_secs)
    print("\n=== CORPUS COVERAGE ===")
    print(f"GT sections found in corpus : {len(all_gt & corpus_secs)}/{len(all_gt)}")
    print(f"GT sections MISSING         : {missing}")
    none_in = sum(1 for q in queries if not (q["gt_secs"] & corpus_secs))
    print(f"Queries whose GT is entirely absent from corpus: {none_in}/{len(queries)} "
          f"(max possible Hit@K = {(len(queries) - none_in) / len(queries) * 100:.1f}%)")

    # ---- Phase 1: HyDE generation (cached) ----
    hyde_cache = load_hyde_cache()
    if RUN_HYDE:
        todo = [q["query"] for q in queries if q["query"] not in hyde_cache]
        if todo:
            print(f"\nGenerating HyDE paragraphs for {len(todo)} queries (cached: {len(hyde_cache)})...")
            for q in tqdm(todo, desc="HyDE"):
                hypo = call_ollama(hyde_prompt(q))
                if hypo and len(hypo) > 10:
                    hyde_cache[q] = hypo
                    save_hyde_cache(hyde_cache)      # save every time -> resumable
        else:
            print("\nAll HyDE paragraphs already cached.")

    # ---- Models ----
    print("\nLoading BGE-M3...")
    dense_model = SentenceTransformer("BAAI/bge-m3", device=device)
    dense_model.max_seq_length = MAX_LENGTH
    doc_emb = dense_model.encode(
        enriched, batch_size=BATCH_SIZE_ENCODE, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    print("Building BM25...")
    bm25 = BM25Okapi([simple_tokenizer(t) for t in enriched])

    print("Loading BGE-reranker-v2-m3...")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)

    variants = [V_BM25, V_DENSE, V_HYB, V_DR, V_HR]
    if RUN_HYDE:
        variants += [V_HYDE, V_HYR, V_HYRF]

    results = {v: [] for v in variants}
    details = []
    pool_used = []

    # ---- Phase 2: evaluate ----
    print("\nEvaluating all variants...")
    for qi, qd in enumerate(tqdm(queries, desc="Queries")):
        query, gt_secs = qd["query"], qd["gt_secs"]
        in_corpus = len(gt_secs & corpus_secs) > 0
        fully_in_corpus = gt_secs <= corpus_secs
        rr_cache = {}

        def rerank_scores(idxs):
            missing_idx = [i for i in idxs if i not in rr_cache]
            if missing_idx:
                pairs = [[query, enriched[i]] for i in missing_idx]
                sc = reranker.predict(pairs, batch_size=BATCH_SIZE_RERANK, show_progress_bar=False)
                for i, s in zip(missing_idx, sc):
                    rr_cache[i] = float(s)
            return np.array([rr_cache[i] for i in idxs])

        # first-stage scores
        bm = bm25.get_scores(simple_tokenizer(query))
        bm_n = bm / bm.max() if bm.max() > 0 else bm
        q_emb = dense_model.encode([normalize_text(query)], normalize_embeddings=True, convert_to_numpy=True)
        dense = np.dot(doc_emb, q_emb.T).squeeze()
        dense_n = minmax(dense)
        hybrid = W_BM25 * bm_n + W_DENSE * dense_n

        rank_bm, rank_dense, rank_hyb = rank_desc(bm), rank_desc(dense), rank_desc(hybrid)

        rankings = {
            V_BM25: rank_bm,
            V_DENSE: rank_dense,
            V_HYB: rank_hyb,
        }

        # Dense + Rerank
        cand = rank_dense[:RERANK_N]
        order = [cand[i] for i in rank_desc(rerank_scores(cand))]
        rankings[V_DR] = merge_rank(order, rank_dense)

        # Hybrid + Rerank
        cand = rank_hyb[:RERANK_N]
        order = [cand[i] for i in rank_desc(rerank_scores(cand))]
        rankings[V_HR] = merge_rank(order, rank_hyb)

        # HyDE variants
        if RUN_HYDE:
            hypo = hyde_cache.get(query) or f"This document addresses: {query} under the Bangladesh Income Tax Act."
            h_emb = dense_model.encode([hypo], normalize_embeddings=True, convert_to_numpy=True)
            dense_h = np.dot(doc_emb, h_emb.T).squeeze()
            rank_h = rank_desc(dense_h)
            rankings[V_HYDE] = rank_h

            cand = rank_h[:RERANK_N]
            rs = rerank_scores(cand)
            order = [cand[i] for i in rank_desc(rs)]
            rankings[V_HYR] = merge_rank(order, rank_h)

            final = W_RERANK * minmax(rs) + W_HYDE * minmax(dense_h[cand])
            order = [cand[i] for i in rank_desc(final)]
            rankings[V_HYRF] = merge_rank(order, rank_h)

        row = {
            "Query": query,
            "GT_Sections": ", ".join(sorted(gt_secs)),
            "GT_in_corpus": int(in_corpus),
            "GT_fully_in_corpus": int(fully_in_corpus),
        }
        for v in variants:
            m = eval_ranking(rankings[v], gt_secs, doc_secs)
            m["_in_corpus"] = in_corpus
            results[v].append(m)
            row[f"{v} | Rank"] = m["First_Rank"]
            row[f"{v} | Top1"] = documents[rankings[v][0]]["parent_sec"]

        # failure analysis for the chosen variant
        fv = FAILURE_VARIANT if FAILURE_VARIANT in rankings else V_HR
        fm = results[fv][-1]
        top1_secs = doc_secs[rankings[fv][0]]
        gt_nums, top1_nums = section_numbers(gt_secs), section_numbers(top1_secs)
        near = int(fm["First_Rank"] != 1 and any(0 < abs(a - b) <= 3 for a in gt_nums for b in top1_nums))
        row["Failure_Variant"] = fv
        row["Failure_Category"] = failure_category(fm["First_Rank"], in_corpus)
        row["Top1_Near_GT_Section"] = near
        details.append(row)

        if device == "cuda" and qi % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # ---- Summaries ----
    metric_cols = (
        [f"{m}@{k}" for k in K_VALUES for m in ["Hit", "Precision", "Recall", "F1", "nDCG", "MAP"]]
        + [f"Recall@{n}" for n in RECALL_KS] + ["MRR"]
    )

    def summarize(subset_only_in_corpus=False):
        rows = []
        for v in variants:
            recs = [r for r in results[v] if (r["_in_corpus"] or not subset_only_in_corpus)]
            if not recs:
                continue
            d = {"Variant": v, "Queries": len(recs)}
            for c in metric_cols:
                d[c] = float(np.mean([r[c] for r in recs]))
            rows.append(d)
        return pd.DataFrame(rows)

    summary_all = summarize(False)
    summary_in = summarize(True)

    det_df = pd.DataFrame(details)
    cat_counts = det_df["Failure_Category"].value_counts().rename_axis("Category").reset_index(name="Queries")
    cat_counts["Percent"] = (cat_counts["Queries"] / len(det_df) * 100).round(2)
    near_total = int(det_df["Top1_Near_GT_Section"].sum())

    coverage_df = pd.DataFrame({
        "GT section": sorted(all_gt),
        "In corpus": ["yes" if s in corpus_secs else "NO" for s in sorted(all_gt)],
    })

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        summary_all.to_excel(writer, sheet_name="Summary_all", index=False)
        summary_in.to_excel(writer, sheet_name="Summary_GT_in_corpus", index=False)
        cat_counts.to_excel(writer, sheet_name="Failure_Categories", index=False)
        coverage_df.to_excel(writer, sheet_name="Coverage", index=False)
        det_df.to_excel(writer, sheet_name="Details", index=False)

    # ---- Console report ----
    show = ["Variant", "Queries", "Hit@1", "Hit@3", "Hit@5", "Recall@5", "nDCG@5", "MRR",
            "Recall@10", "Recall@20", "Recall@50", "Recall@100"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n" + "=" * 100)
    print("ABLATION - ALL QUERIES")
    print("=" * 100)
    print(summary_all[show].round(4).to_string(index=False))
    print("\n" + "=" * 100)
    print("ABLATION - ONLY QUERIES WHOSE GT SECTION EXISTS IN CORPUS")
    print("=" * 100)
    print(summary_in[show].round(4).to_string(index=False))
    print("\n" + "=" * 100)
    print(f"FAILURE CATEGORIES ({FAILURE_VARIANT})")
    print("=" * 100)
    print(cat_counts.to_string(index=False))
    print(f"\nQueries where top-1 was a numerically nearby section (within 3) but wrong: {near_total}")
    print(f"\nSaved: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()