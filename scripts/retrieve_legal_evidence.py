import json
import re
from pathlib import Path

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
    "data/QA_Dataset/qa_evidence_validated.json"
)

TOP_K_BM25 = 10
FINAL_EVIDENCE = 5


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "what", "which", "how", "why", "when", "where",
    "is", "are", "am", "i", "my", "me", "do", "does",
    "can", "could", "should", "would",
    "the", "a", "an", "of", "to", "for", "from",
    "in", "on", "at", "and", "or", "with",
    "under", "about", "be", "if", "this", "that",

    "কি", "কী", "কিভাবে", "কীভাবে", "কোন", "কোনটি",
    "আমার", "আমি", "হলে", "থেকে", "জন্য", "এর",
    "এবং", "বা", "কি", "করতে", "হয়", "হবে"
}


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    """
    Tokenize English + Bangla + numbers.
    """

    text = text.lower()

    tokens = re.findall(
        r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*|"
        r"[\u0980-\u09FF]+|"
        r"\d+(?:[.,]\d+)?",
        text
    )

    tokens = [
        token
        for token in tokens
        if token not in STOPWORDS
    ]

    return tokens


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions():

    print("=" * 70)
    print("LOADING QUESTIONS")
    print("=" * 70)

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Questions file not found: {QUESTIONS_FILE}"
        )

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        questions = json.load(f)

    print(f"Questions loaded: {len(questions)}")

    if len(questions) != 200:
        raise ValueError(
            f"Expected 200 questions, found {len(questions)}"
        )

    return questions


# ============================================================
# LOAD LEGAL DATA
# ============================================================

def load_legal_data():

    print()
    print("=" * 70)
    print("LOADING INCOME TAX ACT")
    print("=" * 70)

    if not LEGAL_FILE.exists():
        raise FileNotFoundError(
            f"Legal file not found: {LEGAL_FILE}"
        )

    with open(
        LEGAL_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        legal_data = json.load(f)

    print(f"Legal records loaded: {len(legal_data)}")

    if not legal_data:
        raise ValueError("Legal dataset is empty.")

    return legal_data


# ============================================================
# BUILD DOCUMENT TEXT
# ============================================================

def build_documents(legal_data):

    documents = []

    for record in legal_data:

        title = normalize(
            record.get("title", "")
        )

        text = normalize(
            record.get("text", "")
        )

        combined = f"{title} {text}"

        documents.append(combined)

    return documents


# ============================================================
# KEYWORD OVERLAP
# ============================================================

def keyword_overlap(question_tokens, document_tokens):

    if not question_tokens:
        return 0.0

    question_set = set(question_tokens)
    document_set = set(document_tokens)

    overlap = question_set.intersection(
        document_set
    )

    return len(overlap) / len(question_set)


# ============================================================
# RETRIEVE
# ============================================================

def retrieve_evidence(
    question,
    bm25,
    legal_data,
    tokenized_documents
):

    question_tokens = tokenize(question)

    if not question_tokens:
        return []

    # --------------------------------------------------------
    # BM25 TOP 10
    # --------------------------------------------------------

    scores = bm25.get_scores(question_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:TOP_K_BM25]

    candidates = []

    # --------------------------------------------------------
    # RE-RANK
    # --------------------------------------------------------

    for index in ranked_indices:

        record = legal_data[index]

        document_tokens = tokenized_documents[index]

        overlap = keyword_overlap(
            question_tokens,
            document_tokens
        )

        bm25_score = float(scores[index])

        # Combined score
        final_score = (
            bm25_score * 0.75
            +
            overlap * 10.0
        )

        candidates.append(
            {
                "index": index,
                "bm25_score": round(
                    bm25_score,
                    4
                ),
                "keyword_overlap": round(
                    overlap,
                    4
                ),
                "relevance_score": round(
                    final_score,
                    4
                ),
                "section": record.get(
                    "section"
                ),
                "title": record.get(
                    "title"
                ),
                "chapter": record.get(
                    "chapter"
                ),
                "part": record.get(
                    "part"
                ),
                "page": record.get(
                    "page"
                ),
                "source": record.get(
                    "source"
                ),
                "text": record.get(
                    "text"
                )
            }
        )

    candidates.sort(
        key=lambda x: x["relevance_score"],
        reverse=True
    )

    return candidates[:FINAL_EVIDENCE]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TaxAssist - Improved Legal Evidence Retrieval")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    questions = load_questions()

    legal_data = load_legal_data()

    # --------------------------------------------------------
    # VERIFY QUESTION IDs
    # --------------------------------------------------------

    question_numbers = sorted(
        q["number"]
        for q in questions
    )

    expected = list(range(1, 201))

    if question_numbers != expected:

        raise ValueError(
            "Question numbers are not exactly 1-200."
        )

    print("✓ All 200 questions verified")

    # --------------------------------------------------------
    # BUILD BM25
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING BM25 INDEX")
    print("=" * 70)

    documents = build_documents(
        legal_data
    )

    tokenized_documents = [
        tokenize(doc)
        for doc in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    print(
        f"BM25 documents indexed: "
        f"{len(tokenized_documents)}"
    )

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RETRIEVING AND RE-RANKING")
    print("=" * 70)

    results = []

    for count, question_record in enumerate(
        questions,
        start=1
    ):

        question = question_record[
            "question"
        ]

        evidence = retrieve_evidence(
            question,
            bm25,
            legal_data,
            tokenized_documents
        )

        results.append(
            {
                "question_id":
                    question_record["question_id"],

                "number":
                    question_record["number"],

                "question":
                    question,

                "language":
                    question_record.get(
                        "language"
                    ),

                "evidence":
                    evidence
            }
        )

        if count % 20 == 0:

            print(
                f"Processed {count}/200"
            )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    with_evidence = sum(
        1
        for item in results
        if item["evidence"]
    )

    without_evidence = (
        len(results) - with_evidence
    )

    print(
        f"Questions processed : {len(results)}"
    )

    print(
        f"With evidence       : {with_evidence}"
    )

    print(
        f"Without evidence    : {without_evidence}"
    )

    # --------------------------------------------------------
    # SAVE
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
    # DISPLAY SAMPLE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FIRST 5 RETRIEVAL RESULTS")
    print("=" * 70)

    for item in results[:5]:

        print()
        print(
            f"Q{item['number']}: "
            f"{item['question']}"
        )

        for rank, ev in enumerate(
            item["evidence"],
            start=1
        ):

            print(
                f"  Rank {rank} | "
                f"Section {ev['section']} | "
                f"Page {ev['page']} | "
                f"Score {ev['relevance_score']}"
            )

            print(
                f"  Title: "
                f"{normalize(ev['title'])}"
            )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATED EVIDENCE DATASET CREATED")
    print("=" * 70)

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print("✓ 200 questions processed")
    print("✓ Top-10 BM25 candidates generated")
    print("✓ Candidates re-ranked")
    print("✓ Top-5 evidence preserved")
    print("✓ Legal section numbers preserved")
    print("✓ Page numbers preserved")
    print("✓ Original legal text preserved")
    print("✓ No LLM-generated answers")

    print()
    print("DONE")


if __name__ == "__main__":
    main()