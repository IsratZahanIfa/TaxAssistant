"""
End-to-end retrieval evaluation for the Bangladesh Income Tax RAG.

Pipeline (each stage can be switched off in CONFIG for ablation):
  1. First stage : Reciprocal Rank Fusion of
                     - BM25 (Bangla-safe tokenizer)
                     - BGE-M3 dense search with the raw query
                     - BGE-M3 dense search with a HyDE paragraph (Gemma3 via Ollama, cached)
  2. Rerank      : bge-reranker-v2-m3 on the top RERANK_N chunks (query, chunk) pairs
  3. Boost       : if the question itself names a section ("Section 33", "ধারা ৩৩"), that
                   section's chunks are moved to the front (counted and reported separately)
  4. Metrics     : on SECTIONS (chunks de-duplicated by section), strict section matching

Fixes compared with the previous version:
  * strict ground-truth matching (no "any shared number" and no Jaccard fallback)
  * a chunk's section comes from its 'section' field only, first number ("76(3)" -> 76)
  * Bangla words are no longer split into letters by the BM25 tokenizer
  * the query is embedded as written (documents are not lower-cased either)
  * HyDE paragraphs are cached and every fallback is counted
  * the main() body is complete: encode, retrieve, rerank, evaluate, save Excel

Run from the project root:  python scripts/Evaluation_retrieval1.py
"""
import gc
import json
import sys
from pathlib import Path
import warnings


# ---- Use the project's .venv automatically --------------------------------------------
# VS Code's Run button may start this file with a Python that has no packages installed.
# If a .venv exists in this folder or a parent folder and we are not running inside it,
# restart this script with the .venv's Python (same arguments, same terminal).
def _reexec_in_project_venv() -> None:
    import os
    import subprocess
    if os.environ.get("TAXASSIST_VENV_REEXEC"):
        return  # already restarted once; never loop
    here = Path(__file__).resolve()
    rel = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    for folder in (here.parent, *here.parents):
        venv = folder / ".venv"
        if (venv / rel).exists():
            if Path(sys.prefix).resolve() != venv.resolve():
                print(f"[info] Restarting with the project environment: {venv / rel}")
                env = dict(os.environ, TAXASSIST_VENV_REEXEC="1")
                sys.exit(subprocess.call([str(venv / rel), str(here), *sys.argv[1:]], env=env))
            return


_reexec_in_project_venv()

try:
    import numpy as np
    import pandas as pd
    import requests
    import torch
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from tqdm import tqdm
except ModuleNotFoundError as e:
    sys.exit(f"Missing package '{e.name}'. From the project folder run:\n"
             "  uv pip install --python .venv numpy pandas openpyxl requests torch tqdm "
             "rank_bm25 sentence-transformers")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
try:  # the metrics file may be saved as Eval_recall.py or eval_recall.py
    from Eval_recall import (
        dedupe, extract_doc_section, extract_gt_sections, query_metrics, recall_at, tokenize,
    )
except ModuleNotFoundError:
    from eval_recall import (
        dedupe, extract_doc_section, extract_gt_sections, query_metrics, recall_at, tokenize,
    )


def _find_project_root() -> Path:
    """First parent folder that contains data/ (so the script works from any folder)."""
    for folder in (_HERE, *_HERE.parents):
        if (folder / "data").is_dir():
            return folder
    return Path.cwd()


PROJECT_ROOT = _find_project_root()

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"
JSON_PATH = PROJECT_ROOT / "data" / "Knowledge_Dataset" / "income_tax_act_2023.json"
GOOGLE_SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/17ew_umfBKtAeTUEBFtBknNoyQR5KbWRSgceh47DNeYY"
    "/export?format=csv&gid=0"
)
QUESTION_COL = "Question"
GT_COL = "Ground Truth Legal Basis"

HYDE_CACHE_PATH = PROJECT_ROOT / "hyde_cache_v2.json"   # new prompt -> new cache file
OUTPUT_XLSX = PROJECT_ROOT / "retrieval_eval_results.xlsx"

USE_BM25 = True
USE_HYDE = True
USE_RERANK = True
EXPLICIT_SECTION_BOOST = True

RRF_K = 60
RERANK_N = 100            # chunks sent to the reranker
MAX_LENGTH = 512
BATCH_SIZE_ENCODE = 16
BATCH_SIZE_RERANK = 4

K_VALUES = (1, 3, 5)
DIAG_RECALL_N = (10, 20, 50, 100)   # first-stage recall (in distinct sections)


# ============================================================
# HyDE (Ollama, cached)
# ============================================================
def hyde_prompt(query: str) -> str:
    return (
        "You are drafting text in the style of the Bangladesh Income Tax Act 2023.\n"
        "Write one short paragraph (80-120 words) that reads like the legal provision answering "
        "the question below. Use the same language as the question. Do not invent section numbers.\n\n"
        f"Question: {query}\n\nProvision:"
    )


def ollama_ready() -> bool:
    """True if the Ollama server answers and OLLAMA_MODEL has been pulled."""
    base = OLLAMA_URL.split("/api/")[0]
    try:
        r = requests.get(f"{base}/api/tags", timeout=3)
    except requests.RequestException:
        print(f"[warning] Ollama is not running at {base}. Start the Ollama app "
              "(or 'ollama serve'); install it from https://ollama.com/download if needed.")
        return False
    names = [m.get("name", "") for m in r.json().get("models", [])] if r.ok else []
    if not any(n == OLLAMA_MODEL or n.startswith(OLLAMA_MODEL + ":") for n in names):
        print(f"[warning] Ollama is running but '{OLLAMA_MODEL}' is not installed: "
              f"run 'ollama pull {OLLAMA_MODEL}'.")
        return False
    return True


def call_ollama(prompt: str) -> str:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 256}},
            timeout=180,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        print(f"Ollama error: HTTP {r.status_code}")
    except requests.RequestException as e:
        print(f"Ollama request failed: {e}")
    return ""


def load_cache() -> dict:
    if HYDE_CACHE_PATH.exists():
        with open(HYDE_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    tmp = HYDE_CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    tmp.replace(HYDE_CACHE_PATH)   # atomic: an interrupted run never corrupts the cache


# ============================================================
# RANKING HELPERS
# ============================================================
def rank_desc(scores) -> np.ndarray:
    return np.argsort(-np.asarray(scores, dtype=float), kind="stable")


def rrf(rankings, n_docs: int, k: int = RRF_K) -> np.ndarray:
    fused = np.zeros(n_docs)
    for r in rankings:
        fused[r] += 1.0 / (k + np.arange(1, len(r) + 1))
    return rank_desc(fused)


def failure_category(first_rank: int, in_corpus: bool) -> str:
    if not in_corpus:
        return "GT section not in corpus"
    if first_rank == 1:
        return "Correct @1"
    if 2 <= first_rank <= 5:
        return "In top-5, not first"
    if 6 <= first_rank <= 20:
        return "Rank 6-20"
    if first_rank > 20:
        return "Rank > 20"
    return "Not retrieved"


# ============================================================
# MAIN
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Retrieval evaluation on {device.upper()}")

    # ---- Corpus ----
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON not found: {JSON_PATH}")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise TypeError("Corpus JSON must be a list of objects.")

    docs = []
    for item in raw:
        text = str(item.get("text", "")).strip()
        if text:
            docs.append({"section": str(item.get("section", "")),
                         "title": str(item.get("title", "")),
                         "text": text})
    if not docs:
        raise ValueError("Corpus is empty.")

    enriched = [f"Section: {d['section']}. Title: {d['title']}. Text: {d['text']}" for d in docs]
    chunk_sec = [extract_doc_section(d["section"]) for d in docs]
    # chunks without a parseable section get their own id so they still count as one result
    rank_id = [s if s else f"_chunk{i}" for i, s in enumerate(chunk_sec)]
    corpus_secs = {s for s in chunk_sec if s}
    n_docs = len(docs)

    no_sec = [i for i, s in enumerate(chunk_sec) if not s]
    print(f"Loaded {n_docs} chunks, {len(corpus_secs)} distinct sections")
    print(f"Chunks with no parseable section id: {len(no_sec)}")
    for i in no_sec[:5]:
        print(f"   e.g. section field = {docs[i]['section']!r}")

    # ---- Queries ----
    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    missing_cols = {QUESTION_COL, GT_COL} - set(df.columns)
    if missing_cols:
        raise ValueError(f"Query sheet is missing columns: {missing_cols}")

    queries, skipped = [], []
    for _, row in df.iterrows():
        q = str(row[QUESTION_COL]).strip()
        gt = row[GT_COL]
        if not q or q.lower() == "nan":
            continue
        gt_secs = extract_gt_sections(gt)
        if not gt_secs:
            skipped.append({"Question": q, "Ground truth": gt})
            continue
        queries.append({"query": q, "gt_text": str(gt), "gt_secs": gt_secs})
    print(f"Queries: {len(queries)} evaluated, {len(skipped)} skipped (no section id in ground truth)")

    all_gt = set().union(*(q["gt_secs"] for q in queries)) if queries else set()
    absent = sorted(all_gt - corpus_secs, key=lambda s: (len(s), s))
    none_in = sum(1 for q in queries if not (q["gt_secs"] & corpus_secs))
    print(f"GT sections in corpus: {len(all_gt & corpus_secs)}/{len(all_gt)}; missing: {absent}")
    if queries:
        print(f"Upper bound on Hit@K: {(len(queries) - none_in) / len(queries):.1%}")

    # ---- HyDE generation (cached, resumable) ----
    # HyDE is used only if every query has a paragraph, so a run never mixes HyDE and non-HyDE queries.
    cache = load_cache()
    use_hyde = USE_HYDE
    if use_hyde:
        todo = [q["query"] for q in queries if q["query"] not in cache]
        if todo and not ollama_ready():
            print(f"[warning] {len(todo)} queries have no cached HyDE paragraph and Ollama is unavailable, "
                  "so HyDE is switched OFF for this run (results are reported as 'HyDE: False').")
            use_hyde = False
        elif todo:
            print(f"Generating HyDE for {len(todo)} queries ({len(cache)} cached)...")
            failures = 0
            for q in tqdm(todo, desc="HyDE"):
                hypo = call_ollama(hyde_prompt(q))
                if len(hypo) > 20:
                    cache[q] = hypo
                    save_cache(cache)
                    failures = 0
                else:
                    failures += 1
                    if failures >= 3:
                        print("[warning] 3 HyDE calls in a row failed; stopping generation. "
                              "Fix Ollama and re-run: cached paragraphs are kept.")
                        break
            if any(q["query"] not in cache for q in queries):
                print("[warning] Some queries still have no HyDE paragraph, so HyDE is OFF for this run.")
                use_hyde = False

    # ---- Models / indexes ----
    print("Loading BGE-M3...")
    dense_model = SentenceTransformer("BAAI/bge-m3", device=device)
    dense_model.max_seq_length = MAX_LENGTH
    doc_emb = dense_model.encode(enriched, batch_size=BATCH_SIZE_ENCODE, show_progress_bar=True,
                                 normalize_embeddings=True, convert_to_numpy=True)
    bm25 = BM25Okapi([tokenize(t) for t in enriched]) if USE_BM25 else None
    reranker = None
    if USE_RERANK:
        print("Loading bge-reranker-v2-m3...")
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, max_length=MAX_LENGTH)
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- Evaluate ----
    per_query, details = [], []
    hyde_fallbacks = boosted = 0

    for qi, qd in enumerate(tqdm(queries, desc="Queries")):
        query, gt_secs = qd["query"], qd["gt_secs"]

        q_emb = dense_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        rankings = [rank_desc(doc_emb @ q_emb)]
        if bm25 is not None:
            bm = bm25.get_scores(tokenize(query))
            if bm.max() > 0:
                rankings.append(rank_desc(bm))
        hyde_used = False
        if use_hyde:
            hypo = cache.get(query)
            if hypo:
                h_emb = dense_model.encode([hypo], normalize_embeddings=True, convert_to_numpy=True)[0]
                rankings.append(rank_desc(doc_emb @ h_emb))
                hyde_used = True
            else:
                hyde_fallbacks += 1

        first_stage = rrf(rankings, n_docs) if len(rankings) > 1 else rankings[0]
        final = list(first_stage)

        if reranker is not None:
            cand = list(first_stage[:RERANK_N])
            scores = reranker.predict([[query, enriched[i]] for i in cand],
                                      batch_size=BATCH_SIZE_RERANK, show_progress_bar=False)
            reranked = [cand[i] for i in rank_desc(scores)]
            final = reranked + list(first_stage[RERANK_N:])

        named = extract_gt_sections(query) & corpus_secs if EXPLICIT_SECTION_BOOST else set()
        if named:
            boosted += 1
            final = [i for i in final if chunk_sec[i] in named] + \
                    [i for i in final if chunk_sec[i] not in named]

        sec_ranking = dedupe(rank_id[i] for i in final)
        fs_ranking = dedupe(rank_id[i] for i in first_stage)

        m = query_metrics(sec_ranking, gt_secs, K_VALUES)
        for n in DIAG_RECALL_N:
            m[f"FirstStage_Recall@{n}"] = recall_at(fs_ranking, gt_secs, n)
        per_query.append(m)

        in_corpus = bool(gt_secs & corpus_secs)
        details.append({
            "Query": query,
            "GT_Text": qd["gt_text"],
            "GT_Sections": ", ".join(sorted(gt_secs)),
            "GT_in_corpus": int(in_corpus),
            "Top5_Sections": ", ".join(sec_ranking[:5]),
            "First_Rank": m["First_Rank"],
            "Failure_Category": failure_category(m["First_Rank"], in_corpus),
            "HyDE_used": int(hyde_used),
            "Section_named_in_query": int(bool(named)),
            **{f"Hit@{k}": m[f"Hit@{k}"] for k in K_VALUES},
        })

        if device == "cuda" and qi % 10 == 0:
            torch.cuda.empty_cache()

    if not per_query:
        print("No queries to evaluate.")
        return

    # ---- Summaries ----
    det_df = pd.DataFrame(details)
    metric_keys = [k for k in per_query[0] if k != "First_Rank"]

    def summarize(mask, label):
        rows = [p for p, keep in zip(per_query, mask) if keep]
        if not rows:
            return None
        out = {"Subset": label, "Queries": len(rows)}
        out.update({k: float(np.mean([r[k] for r in rows])) for k in metric_keys})
        return out

    in_corpus_mask = det_df["GT_in_corpus"].astype(bool).tolist()
    not_named_mask = (~det_df["Section_named_in_query"].astype(bool)).tolist()
    summary = pd.DataFrame([s for s in (
        summarize([True] * len(per_query), "All queries"),
        summarize(in_corpus_mask, "GT section in corpus"),
        summarize(not_named_mask, "Section NOT named in question"),
    ) if s])

    config = pd.DataFrame([{
        "BM25": USE_BM25, "HyDE": use_hyde, "Rerank": USE_RERANK,
        "Explicit_section_boost": EXPLICIT_SECTION_BOOST, "RERANK_N": RERANK_N, "RRF_K": RRF_K,
        "Chunks": n_docs, "Queries": len(queries), "Skipped_no_section": len(skipped),
        "HyDE_fallbacks": hyde_fallbacks, "Queries_naming_a_section": boosted,
    }])
    failures = det_df["Failure_Category"].value_counts().rename_axis("Category").reset_index(name="Queries")
    failures["Percent"] = (failures["Queries"] / len(det_df) * 100).round(2)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        config.to_excel(w, sheet_name="Config", index=False)
        failures.to_excel(w, sheet_name="Failure_Categories", index=False)
        det_df[det_df["First_Rank"] != 1].to_excel(w, sheet_name="Failures", index=False)
        det_df.to_excel(w, sheet_name="Details", index=False)
        pd.DataFrame(skipped).to_excel(w, sheet_name="Skipped", index=False)

    # ---- Console report ----
    show = ["Subset", "Queries"] + [f"Hit@{k}" for k in K_VALUES] + \
           ["MRR@5", "nDCG@5", "FirstStage_Recall@20", "FirstStage_Recall@100"]
    pd.set_option("display.width", 220)
    print("\n" + "=" * 90)
    print("RETRIEVAL EVALUATION (section-level, strict)")
    print("=" * 90)
    print(summary[show].round(4).to_string(index=False))
    print(f"\nHyDE fallbacks: {hyde_fallbacks} | Questions naming a section: {boosted}")
    print("\n" + failures.to_string(index=False))
    print(f"\nSaved: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
