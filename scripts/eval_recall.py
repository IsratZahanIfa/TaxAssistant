"""
Section-level retrieval metrics for the Bangladesh Income Tax RAG.

Everything is measured on SECTIONS, not chunks:
  * chunk ids are mapped to their section id and de-duplicated (first occurrence wins),
    so several chunks of the same section cannot fill the top-k or inflate recall;
  * ground-truth matching is strict: a section id is only taken from text that names it
    ("Section 44", "sections 44, 45 and 46", "ধারা ৩৩", "৩৩ ধারা").

Metrics per query: Hit@k, Precision@k, Recall@k, MRR@k, nDCG@k, MRR (full ranking), First_Rank.
Queries without ground-truth sections are skipped and counted, never scored as 0.

Other scripts import the helpers from here, so there is a single definition of the metrics.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

BN_TO_EN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
DEFAULT_KS = (1, 3, 5)


# =========================== TEXT HELPERS ===========================
def normalize_text(text: Any) -> str:
    """Lower-case and convert Bangla digits to ASCII digits. None/NaN -> ''."""
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    return str(text).lower().translate(BN_TO_EN_DIGITS)


# \w alone splits Bangla words at every vowel sign / hasanta, so include the Bengali block.
_TOKEN = re.compile(r"[\wঀ-৿]+")


def tokenize(text: Any) -> List[str]:
    """Tokenizer for BM25 that keeps Bangla words whole."""
    return _TOKEN.findall(normalize_text(text))


# =========================== SECTION IDS ===========================
_ID = r"\d+[a-z]?"
_LIST = rf"{_ID}(?:\s*(?:,|and|&|ও|এবং)\s*{_ID})*"
# "Section 44", "sections 44, 45 and 46", "sec. 33", "ধারা ৩৩", "ধারার ৩৩"
_SEC_BEFORE = re.compile(rf"(?:sections?|sec\.?|s\.|ধারা(?:র|য়)?)\s*({_LIST})")
# Bangla word order: "৩৩ ধারা", "৩৩ নং ধারা"
_SEC_AFTER = re.compile(rf"({_ID})\s*(?:নং|নম্বর)?\s*ধারা")
_ID_RE = re.compile(_ID)
# A section field such as "Section 76(3)" -> "76" (never the sub-section number)
_DOC_SEC = re.compile(rf"\s*(?:sections?|sec\.?|ধারা)?\s*({_ID})")


def extract_gt_sections(text: Any) -> Set[str]:
    """Section ids named in a ground-truth / question string (strict)."""
    t = normalize_text(text)
    ids: Set[str] = set()
    for group in _SEC_BEFORE.findall(t):
        ids.update(_ID_RE.findall(group))
    ids.update(_SEC_AFTER.findall(t))
    return ids


def extract_doc_section(section_field: Any) -> Optional[str]:
    """The section id of a chunk, taken from its 'section' field only (first number)."""
    m = _DOC_SEC.match(normalize_text(section_field))
    return m.group(1) if m else None


# =========================== RANKING HELPERS ===========================
def normalize_id(x: Any) -> str:
    return normalize_text(x).strip()


def dedupe(ids: Iterable[Any]) -> List[str]:
    """Normalize ids and keep only the first occurrence of each, in order."""
    seen, out = set(), []
    for i in ids:
        n = normalize_id(i)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def to_section_ranking(retrieved_chunk_ids: Sequence[Any],
                       chunk_to_section: Mapping[Any, Any]) -> List[str]:
    """Map chunk ids to section ids (unknown chunks keep their own id), then de-duplicate."""
    return dedupe(chunk_to_section.get(c) or c for c in retrieved_chunk_ids)


# =========================== METRICS ===========================
def query_metrics(ranking: Sequence[Any], ground_truth: Iterable[Any],
                  ks: Sequence[int] = DEFAULT_KS) -> Dict[str, float]:
    """All metrics for one query. `ranking` is a ranked list of SECTION ids."""
    ranked = dedupe(ranking)
    gt = set(dedupe(ground_truth))
    if not gt:
        raise ValueError("query_metrics needs at least one ground-truth section")

    rel = [1 if r in gt else 0 for r in ranked]
    first = next((i + 1 for i, x in enumerate(rel) if x), 0)
    res: Dict[str, float] = {"First_Rank": first, "MRR": (1.0 / first) if first else 0.0}

    for k in ks:
        top = rel[:k]
        hits = sum(top)
        dcg = sum(x / math.log2(i + 2) for i, x in enumerate(top))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gt))))
        res[f"Hit@{k}"] = 1.0 if hits else 0.0
        res[f"Precision@{k}"] = hits / k
        res[f"Recall@{k}"] = hits / len(gt)
        res[f"MRR@{k}"] = (1.0 / first) if 0 < first <= k else 0.0
        res[f"nDCG@{k}"] = dcg / idcg if idcg else 0.0
    return res


def recall_at(ranking: Sequence[Any], ground_truth: Iterable[Any], n: int) -> float:
    gt = set(dedupe(ground_truth))
    if not gt:
        return 0.0
    return len(set(dedupe(ranking)[:n]) & gt) / len(gt)


def calculate_recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Recall@K for one query (kept for backward compatibility)."""
    return recall_at(retrieved_ids, ground_truth_ids, k)


def evaluate_dataset(dataset: List[Dict[str, Any]], ks: Sequence[int] = DEFAULT_KS,
                     chunk_to_section: Optional[Mapping[Any, Any]] = None) -> Dict[str, Any]:
    """
    Mean metrics over a dataset of {"retrieved_ids": [...], "ground_truth_ids": [...]}.
    If `chunk_to_section` is given, retrieved_ids are chunk ids and are mapped to sections.
    """
    per_query, skipped = [], 0
    for sample in dataset:
        gt = dedupe(sample.get("ground_truth_ids") or [])
        if not gt:
            skipped += 1
            continue
        ranking = sample.get("retrieved_ids") or []
        if chunk_to_section is not None:
            ranking = to_section_ranking(ranking, chunk_to_section)
        per_query.append(query_metrics(ranking, gt, ks))

    summary: Dict[str, Any] = {"queries": len(per_query), "skipped_no_ground_truth": skipped}
    if per_query:
        for key in per_query[0]:
            if key != "First_Rank":
                summary[key] = sum(q[key] for q in per_query) / len(per_query)
    return summary


def evaluate_dataset_recall(dataset: List[Dict[str, Any]], k: int = 5) -> float:
    """Mean Recall@K (kept for backward compatibility). Queries without ground truth are skipped."""
    return evaluate_dataset(dataset, ks=(k,)).get(f"Recall@{k}", 0.0)


# =========================== DEMO ===========================
if __name__ == "__main__":
    # Illustrative ids only. retrieved_ids are chunk ids; two chunks belong to section 56.
    chunk_to_section = {
        "c10": "10", "c56a": "56", "c56b": "56", "c100": "100", "c57": "57", "c2": "2", "c124": "124",
    }
    sample_dataset = [
        {"query": "What is the tax rate for non-residents?",
         "ground_truth_ids": ["56", "57"],
         "retrieved_ids": ["c10", "c56a", "c56b", "c100", "c57", "c2"]},
        {"query": "What is the penalty for late return filing?",
         "ground_truth_ids": ["124"],
         "retrieved_ids": ["c124", "c56a"]},
        {"query": "Question without a section in its ground truth",
         "ground_truth_ids": [],
         "retrieved_ids": ["c2"]},
    ]

    summary = evaluate_dataset(sample_dataset, ks=(1, 3, 5), chunk_to_section=chunk_to_section)
    print("================ RETRIEVAL EVALUATION ================")
    print(f"Queries evaluated : {summary['queries']} (skipped {summary['skipped_no_ground_truth']})")
    for k in (1, 3, 5):
        print(f"K={k}: Hit {summary[f'Hit@{k}']:.2%} | Recall {summary[f'Recall@{k}']:.2%} | "
              f"nDCG {summary[f'nDCG@{k}']:.4f} | MRR@{k} {summary[f'MRR@{k}']:.4f}")
    print(f"MRR (full ranking): {summary['MRR']:.4f}")
    print("======================================================")
