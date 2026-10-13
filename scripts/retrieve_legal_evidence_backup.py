import json
import re
from pathlib import Path
from collections import Counter

from rank_bm25 import BM25Okapi


# ============================================================
# CONFIGURATION
# ============================================================

QUESTIONS_FILE = Path(
    "data/QA_Dataset/questions.json"
)

LEGAL_FILE = Path(
    "data/Knowledge_Dataset/income_tax_act_2023.json"
)

OUTPUT_FILE = Path(
    "data/QA_Dataset/qa_evidence.json"
)

TOP_K = 5


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "the", "is", "are", "a", "an", "of", "to", "for",
    "in", "on", "and", "or", "my", "me", "i", "do",
    "does", "can", "how", "what", "which", "who",
    "why", "when", "where", "be", "from", "with",
    "under", "if", "it", "this", "that",

    # Common Bangla/Romanized words
    "ki", "kivabe", "kibhabe", "amar", "kon", "kono",
    "ache", "hobe", "hoy", "korle", "korte", "korbo",
    "theke", "jonno", "eta", "ei", "ar", "ki",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize English, Bangla and Romanized Bangla text.
    """

    if not text:
        return ""

    text = str(text)

    # Remove zero-width characters
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", " ", text)

    # Lowercase English
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Keep Bangla Unicode, English and numbers
    text = re.sub(
        r"[^\w\u0980-\u09FF\s\-]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Tokenize text for BM25.
    """

    text = normalize_text(text)

    tokens = text.split()

    # Remove very short tokens and stopwords
    tokens = [
        token
        for token in tokens
        if len(token) > 1
        and token not in STOPWORDS
    ]

    return tokens


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions():

    print("=" * 70)
    print("LOADING QUESTIONS")
    print("=" * 70)

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Questions file not found:\n{QUESTIONS_FILE}"
        )

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        questions = json.load(f)

    if not isinstance(questions, list):
        raise ValueError(
            "questions.json must contain a JSON list."
        )

    print(
        f"Questions loaded: {len(questions)}"
    )

    return questions


# ============================================================
# LOAD LEGAL DOCUMENT
# ============================================================

def load_legal_data():

    print()
    print("=" * 70)
    print("LOADING INCOME TAX ACT")
    print("=" * 70)

    if not LEGAL_FILE.exists():
        raise FileNotFoundError(
            f"Legal file not found:\n{LEGAL_FILE}"
        )

    with open(
        LEGAL_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        legal_data = json.load(f)

    if not isinstance(legal_data, list):
        raise ValueError(
            "income_tax_act_2023.json must contain a JSON list."
        )

    print(
        f"Legal records loaded: {len(legal_data)}"
    )

    return legal_data


# ============================================================
# BUILD SEARCH DOCUMENTS
# ============================================================

def build_search_documents(legal_data):

    documents = []

    for item in legal_data:

        section = str(
            item.get("section", "")
        )

        title = str(
            item.get("title", "")
        )

        text = str(
            item.get("text", "")
        )

        # Searchable representation
        searchable = (
            f"section {section} "
            f"{title} "
            f"{text}"
        )

        documents.append(searchable)

    return documents


# ============================================================
# BUILD BM25
# ============================================================

def build_bm25(search_documents):

    print()
    print("=" * 70)
    print("BUILDING BM25 INDEX")
    print("=" * 70)

    tokenized_documents = [
        tokenize(document)
        for document in search_documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    print(
        f"BM25 documents indexed: "
        f"{len(tokenized_documents)}"
    )

    return bm25


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def normalize_scores(scores):

    if not scores:
        return []

    maximum = max(scores)

    minimum = min(scores)

    if maximum == minimum:

        return [
            0.0
            for _ in scores
        ]

    normalized = [
        (score - minimum) /
        (maximum - minimum)
        for score in scores
    ]

    return normalized


# ============================================================
# RETRIEVE EVIDENCE
# ============================================================

def retrieve_evidence(
    question,
    legal_data,
    bm25,
    top_k=TOP_K
):

    query = question.get(
        "question",
        ""
    )

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(
        query_tokens
    )

    # Get candidate indices
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    # Keep top candidates
    ranked_indices = ranked_indices[:top_k]

    evidence = []

    for rank, index in enumerate(
        ranked_indices,
        start=1
    ):

        item = legal_data[index]

        raw_score = float(
            scores[index]
        )

        evidence.append({

            "rank": rank,

            "retrieval_score": round(
                raw_score,
                4
            ),

            "section": item.get(
                "section"
            ),

            "title": item.get(
                "title"
            ),

            "chapter": item.get(
                "chapter"
            ),

            "part": item.get(
                "part"
            ),

            "page": item.get(
                "page"
            ),

            "source": item.get(
                "source"
            ),

            "text": item.get(
                "text"
            ),

            "source_id": item.get(
                "id"
            )
        })

    return evidence


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TaxAssist - Legal Evidence Retrieval")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    questions = load_questions()

    legal_data = load_legal_data()

    # --------------------------------------------------------
    # Validate questions
    # --------------------------------------------------------

    question_numbers = {
        q.get("number")
        for q in questions
    }

    expected = set(
        range(1, 201)
    )

    if question_numbers != expected:

        missing = sorted(
            expected - question_numbers
        )

        extra = sorted(
            question_numbers - expected
        )

        raise ValueError(
            f"Question numbering problem.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    print(
        "✓ All 200 questions verified"
    )

    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    search_documents = (
        build_search_documents(
            legal_data
        )
    )

    bm25 = build_bm25(
        search_documents
    )

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RETRIEVING LEGAL EVIDENCE")
    print("=" * 70)

    results = []

    for counter, question in enumerate(
        questions,
        start=1
    ):

        evidence = retrieve_evidence(
            question,
            legal_data,
            bm25,
            TOP_K
        )

        result = {

            "question_id": question.get(
                "question_id"
            ),

            "number": question.get(
                "number"
            ),

            "question": question.get(
                "question"
            ),

            "language": question.get(
                "language"
            ),

            "evidence": evidence,

            "evidence_status":
                "retrieved"
                if evidence
                else "not_found"
        }

        results.append(result)

        if counter % 20 == 0:
            print(
                f"Processed "
                f"{counter}/"
                f"{len(questions)}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(
        f"Questions processed : "
        f"{len(results)}"
    )

    with_evidence = sum(
        1
        for r in results
        if r["evidence"]
    )

    without_evidence = (
        len(results)
        - with_evidence
    )

    print(
        f"With evidence       : "
        f"{with_evidence}"
    )

    print(
        f"Without evidence    : "
        f"{without_evidence}"
    )

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FIRST 5 RETRIEVAL RESULTS")
    print("=" * 70)

    for result in results[:5]:

        print()
        print(
            f"Q{result['number']}: "
            f"{result['question']}"
        )

        for evidence in result[
            "evidence"
        ][:3]:

            print(
                f"  Rank {evidence['rank']} | "
                f"Section {evidence['section']} | "
                f"Page {evidence['page']} | "
                f"Score {evidence['retrieval_score']}"
            )

            print(
                f"  Title: "
                f"{evidence['title']}"
            )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVIDENCE DATASET CREATED")
    print("=" * 70)

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        "✓ 200 questions processed"
    )

    print(
        "✓ Legal sections preserved"
    )

    print(
        "✓ Section numbers preserved"
    )

    print(
        "✓ Page numbers preserved"
    )

    print(
        "✓ Original legal text preserved"
    )

    print(
        "✓ No LLM-generated answers"
    )

    print()
    print("DONE")


if __name__ == "__main__":
    main()