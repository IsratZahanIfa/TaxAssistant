"""
run_experiments.py
Runs the same retrieval on the three preprocessing variants and prints one comparison table.

  A baseline (raw records)  |  B integrated (clean + section ids + identity header)  |  C hierarchical (subsection children)

Same questions, same ground truth, same embedding model (BGE-M3), same retriever (BM25 + dense -> RRF,
optional bge-reranker-v2-m3). Only the preprocessing differs, so any change in the metrics comes from it.
Metrics are at SECTION level: chunks are mapped to their section and de-duplicated before scoring.

Needs:  build_integrated_corpus.py already run;  data\QA_Dataset\questions.json (questions + gold sections).
Run (PowerShell, from E:\TaxAssist):
    python scripts\run_experiments.py                 # BM25 + BGE-M3 dense + RRF, all 3 variants
    python scripts\run_experiments.py --rerank        # add bge-reranker-v2-m3 on the fused top-50
    python scripts\run_experiments.py --no-dense      # BM25 only (quick check, no model download)
Not measured here: Citation Accuracy (needs answer generation).
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"E:\TaxAssist")
CORPUS = ROOT / "data/processed/integrated_corpus"


def find_eval():
    """QA input: questions.json (or qa_eval_140.json) - flexible file resolution and schema fallback."""
    seen = []
    gold_candidate_keys = ("gold_keys", "gold_sections", "sections", "gold", "gold_ids")
    for name in ("questions.json", "qa_eval_140.json"):
        for d in (ROOT / "data/QA_Dataset", Path(__file__).resolve().parent, Path.cwd(), Path.home() / "Downloads"):
            f = d / name
            if not f.exists():
                continue
            try:
                data = json.load(open(f, encoding="utf-8"))
                items = data if isinstance(data, list) else data.get("questions", data.get("data", []))
                if not items:
                    continue
                first = items[0]
            except Exception:
                continue
            
            # Match if any known gold key is present or if any element looks like a query
            if any(k in first for k in gold_candidate_keys) or "question" in first or "query" in first:
                return f
            seen.append(str(f))
            
    raise SystemExit("No valid QA evaluation file found. Ensure questions.json is present in "
                     f"{ROOT / 'data/QA_Dataset'}." + (f" Files checked: {seen}" if seen else ""))


EVAL = find_eval()
LAW = "ita_2023"
OUT = ROOT / "outputs/retrieval_evaluation"
POOL, K, RRF_K = 50, 5, 60

STOP = set("a an the and or of to in on for is are was were be been by with at as it its this that these those "
           "from what which who whom when where how why does do did can could will would shall should may might "
           "i my me we our you your they their he she his her not no any all if under".split())


def toks(s):
    if not s:
        return []
    return [t for t in re.findall(r"\w+", s.lower()) if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.n, self.k1, self.b = len(docs), k1, b
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = self.len.mean() or 1.0
        self.post = defaultdict(list)
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                self.post[t].append((i, f))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def scores(self, q):
        s = np.zeros(self.n)
        for t in set(q):
            for i, f in self.post.get(t, ()):
                s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def rrf(lists):
    sc = defaultdict(float)
    for lst in lists:
        for r, i in enumerate(lst, 1):
            sc[i] += 1.0 / (RRF_K + r)
    return sorted(sc, key=sc.get, reverse=True)


def metrics(keys, gold):
    first = next((r for r, k in enumerate(keys, 1) if k in gold), None)
    dcg = sum(1 / math.log2(r + 1) for r, k in enumerate(keys, 1) if k in gold)
    idcg = sum(1 / math.log2(r + 1) for r in range(1, min(len(gold), K) + 1))
    return dict(h1=bool(first and first <= 1), h5=bool(first and first <= K),
                mrr=1 / first if first else 0.0, ndcg=dcg / idcg if idcg else 0.0, first=first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--include-broad", action="store_true", help="also score questions whose gold is a huge range")
    args = ap.parse_args()

    raw_json = json.load(open(EVAL, encoding="utf-8"))
    qa = raw_json if isinstance(raw_json, list) else raw_json.get("questions", raw_json.get("data", []))
    if not qa:
        raise SystemExit(f"{EVAL} contains no valid question entries.")

    ref = {}
    by_num = {}
    for idx, q in enumerate(qa):
        q_num = q.get("number") or q.get("question_id") or q.get("id") or idx
        ref[q_num] = q
        by_num[q_num] = q

    qs = []
    for idx, q in enumerate(qa):
        q_num = q.get("number") or q.get("question_id") or q.get("id") or idx
        g = ref.get(q_num, q)
        
        # Flexibly extract gold key arrays across different schema conventions
        raw_gold = (g.get("gold_keys") or g.get("gold_sections") or 
                    g.get("sections") or g.get("gold") or g.get("gold_ids") or [])
        if isinstance(raw_gold, (str, int)):
            raw_gold = [raw_gold]
        
        # Standardize gold key labels (prepend 'ita_2023:' if absent)
        gold = set()
        for k in raw_gold:
            k_str = str(k).strip()
            if not k_str:
                continue
            if ":" in k_str:
                gold.add(k_str)
            else:
                gold.add(f"{LAW}:{k_str}")

        is_broad = g.get("broad_range", False)
        if not gold or (is_broad and not args.include_broad):
            continue

        q_text = q.get("question") or q.get("query") or q.get("text") or ""
        q_lang = q.get("language", "English")
        
        pair_num = g.get("pair_number")
        pair = by_num.get(pair_num) if pair_num is not None else None
        
        pair_text = (pair.get("question") or pair.get("query") or pair.get("text")) if pair else None
        bm_text = pair_text if (pair and q_lang != "English") else q_text

        qs.append(dict(number=q_num, id=str(q.get("question_id", q_num)), lang=q_lang, text=q_text,
                       bm25=bm_text, gold=gold))

    print(f"{len(qs)} questions with an ITA 2023 gold section "
          f"(language breakdown: {dict(Counter(q['lang'] for q in qs))})")

    model = reranker = qvec = None
    if not args.no_dense:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-m3")
        model.max_seq_length = 512
        qvec = model.encode([q["text"] for q in qs], normalize_embeddings=True, batch_size=16)
    if args.rerank:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

    summary, detail = {}, {}
    for v in args.variants:
        chunks_file = CORPUS / f"chunks_{v}.json"
        if not chunks_file.exists():
            print(f"Skipping variant {v}: {chunks_file} not found.")
            continue

        chunks = json.load(open(chunks_file, encoding="utf-8"))
        bm = BM25([toks(c.get("embed_text", "")) for c in chunks])
        E = None
        if model is not None:
            cache = CORPUS / f"emb_{v}.npy"
            if cache.exists() and np.load(cache).shape[0] == len(chunks):
                E = np.load(cache)
            else:
                print(f"[{v}] embedding {len(chunks)} chunks ...")
                E = model.encode([c.get("embed_text", "") for c in chunks], normalize_embeddings=True, batch_size=16,
                                 show_progress_bar=True)
                np.save(cache, E)
        rows = []
        for qi, q in enumerate(qs):
            s = bm.scores(toks(q["bm25"]))
            lists = [[int(i) for i in np.argsort(-s)[:POOL] if s[i] > 0]]
            if E is not None:
                lists.append([int(i) for i in np.argsort(-(E @ qvec[qi]))[:POOL]])
            cand = rrf(lists)[:POOL]
            if reranker is not None and cand:
                sc = reranker.predict([(q["text"], chunks[i].get("embed_text", "")) for i in cand])
                cand = [cand[j] for j in np.argsort(-np.asarray(sc))]
            keys = list(dict.fromkeys(chunks[i].get("key", "") for i in cand))[:K]
            m = metrics(keys, q["gold"])
            rows.append(dict(id=q["id"], lang=q["lang"], gold=sorted(q["gold"]), top=keys, **m))
        detail[v] = rows
        agg = {}
        for name, sel in [("ALL", rows)] + [(l, [r for r in rows if r["lang"] == l]) for l in ("English", "Bangla", "Mixed")]:
            if sel:
                agg[name] = dict(n=len(sel), **{k: round(sum(r[k] for r in sel) / len(sel), 4)
                                                for k in ("h1", "h5", "mrr", "ndcg")})
        summary[v] = agg

    mode = ("BM25" if args.no_dense else "BM25+BGE-M3") + ("+reranker" if args.rerank else "")
    print(f"\nretriever: {mode}  (RRF k={RRF_K}, pool={POOL}, section-level, K={K})")
    print(f"{'variant':8s}{'group':9s}{'n':>4s}{'Hit@1':>8s}{'Hit@5':>8s}{'MRR@5':>8s}{'nDCG@5':>8s}")
    for v, agg in summary.items():
        for g, a in agg.items():
            print(f"{v:8s}{g:9s}{a['n']:4d}{a['h1']:8.3f}{a['h5']:8.3f}{a['mrr']:8.3f}{a['ndcg']:8.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"experiments_ABC_{mode.replace('+', '_')}.json"
    json.dump(dict(retriever=mode, summary=summary, per_question=detail), open(f, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("saved ->", f)


if __name__ == "__main__":
    main()