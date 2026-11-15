import fitz
import re
import json
from pathlib import Path


INPUT_PDF = Path(
    "data/dataset/TaxAssist_Questions.pdf"
)

OUTPUT_JSON = Path(
    "data/QA_Dataset/questions.json"
)

RAW_TEXT_FILE = Path(
    "data/QA_Dataset/questions_raw.txt"
)

def extract_pdf_text(pdf_path):

    print("=" * 70)
    print("READING PDF")
    print("=" * 70)

    print(f"Opening PDF: {pdf_path}")

    document = fitz.open(pdf_path)

    print(
        f"Total PDF pages: {len(document)}"
    )

    all_pages = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        print(
            f"Reading page {page_number}/{len(document)}..."
        )

        text = page.get_text("text")

        if text:

            all_pages.append(text)

    document.close()

    full_text = "\n".join(all_pages)

    print()
    print(
        f"Extracted characters: {len(full_text):,}"
    )

    return full_text


def clean_pdf_text(text):

    invisible_chars = [
        "\u200b",
        "\u200c",
        "\u200d",
        "\ufeff",
        "\u2060",
    ]

    for char in invisible_chars:

        text = text.replace(
            char,
            ""
        )

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )


    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text


def extract_questions(text):

    print()
    print("=" * 70)
    print("EXTRACTING QUESTIONS")
    print("=" * 70)

    questions = []


    marker_pattern = re.compile(
        r"(?m)^\s*(\d{1,3})\s*\.\s*"
    )

    matches = list(
        marker_pattern.finditer(text)
    )

    print(
        f"Question markers detected: {len(matches)}"
    )


    for index, match in enumerate(matches):

        number = int(
            match.group(1)
        )

        if not 1 <= number <= 200:
            continue

        start = match.end()

        if index + 1 < len(matches):

            next_match = matches[
                index + 1
            ]

            end = next_match.start()

        else:

            end = len(text)

        question_text = text[
            start:end
        ]

        question_text = clean_question(
            question_text
        )

        if not question_text:
            continue

        language = detect_language(
            question_text
        )

        questions.append({

            "question_id":
                f"q_{number:03d}",

            "number":
                number,

            "question":
                question_text,

            "language":
                language
        })

    return questions

def clean_question(question):

    question = question.replace(
        "\u200b",
        ""
    )

    question = question.replace(
        "\u200c",
        ""
    )

    question = question.replace(
        "\u200d",
        ""
    )

    question = question.replace(
        "\ufeff",
        ""
    )

    question = question.replace(
        "\u2060",
        ""
    )

    question = re.sub(
        r"\s+",
        " ",
        question
    )

    question = re.sub(
        r"\s+([?.!,;:])",
        r"\1",
        question
    )


    question = re.sub(
        r"([?.!,;:])\s+",
        r"\1 ",
        question
    )


    question = question.strip()


    question = question.strip(
        " \t\n\r"
    )

    return question


def detect_language(question):

    has_bangla = bool(
        re.search(
            r"[\u0980-\u09FF]",
            question
        )
    )

    has_english = bool(
        re.search(
            r"[A-Za-z]",
            question
        )
    )

    if has_bangla and has_english:

        return "Mixed"

    elif has_bangla:

        return "Bangla"

    else:

        return "English"


def remove_duplicates(questions):

    unique = {}

    duplicates = []

    for question in questions:

        number = question["number"]

        if number in unique:

            duplicates.append(
                number
            )

        else:

            unique[number] = question

    if duplicates:

        print()
        print(
            "Duplicate question numbers detected:"
        )

        print(
            sorted(
                set(duplicates)
            )
        )

    return list(
        unique.values()
    )

def validate_questions(questions):

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    numbers = [
        q["number"]
        for q in questions
    ]

    print(
        f"Total questions found : {len(questions)}"
    )

    print(
        f"Unique question IDs   : {len(set(numbers))}"
    )

    expected = set(
        range(1, 201)
    )

    actual = set(
        numbers
    )

    missing = sorted(
        expected - actual
    )

    extra = sorted(
        actual - expected
    )

    if missing:

        print()
        print(
            f"Missing questions ({len(missing)}):"
        )

        print(
            missing
        )

    else:

        print(
            "✓ No missing questions"
        )


    if extra:

        print()
        print(
            "Extra question numbers:"
        )

        print(
            extra
        )

    duplicates = [
        number
        for number in sorted(
            set(numbers)
        )
        if numbers.count(number) > 1
    ]

    if duplicates:

        print()
        print(
            "Duplicate questions:"
        )

        print(
            duplicates
        )

    else:

        print(
            "✓ No duplicate questions"
        )

    empty_questions = [
        q["number"]
        for q in questions
        if not q["question"].strip()
    ]

    if empty_questions:

        print()
        print(
            "Empty questions:"
        )

        print(
            empty_questions
        )

    else:

        print(
            "✓ No empty questions"
        )


    language_counts = {

        "English": 0,

        "Bangla": 0,

        "Mixed": 0
    }

    for question in questions:

        language = question[
            "language"
        ]

        if language in language_counts:

            language_counts[
                language
            ] += 1

    print()
    print(
        "Language distribution:"
    )

    print(
        f"English : {language_counts['English']}"
    )

    print(
        f"Bangla  : {language_counts['Bangla']}"
    )

    print(
        f"Mixed   : {language_counts['Mixed']}"
    )

    return (
        len(questions) == 200
        and not missing
        and not extra
        and not duplicates
        and not empty_questions
    )


def save_json(
    questions,
    output_path
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            questions,
            file,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(
        f"JSON saved to: {output_path}"
    )

def print_questions(
    questions
):

    print()
    print("=" * 70)
    print("FIRST 10 QUESTIONS")
    print("=" * 70)

    for q in questions[:10]:

        print(
            f"{q['question_id']} | "
            f"{q['language']} | "
            f"{q['question']}"
        )

    print()
    print("=" * 70)
    print("LAST 10 QUESTIONS")
    print("=" * 70)

    for q in questions[-10:]:

        print(
            f"{q['question_id']} | "
            f"{q['language']} | "
            f"{q['question']}"
        )


def check_specific_questions(
    questions
):

    question_map = {
        q["number"]: q
        for q in questions
    }

    test_numbers = [
        1,
        10,
        53,
        54,
        70,
        71,
        99,
        100,
        101,
        109,
        110,
        139,
        140,
        141,
        144,
        150,
        159,
        160,
        175,
        179,
        180,
        181,
        190,
        199,
        200
    ]

    print()
    print("=" * 70)
    print("SPECIFIC QUESTION CHECK")
    print("=" * 70)

    for number in test_numbers:

        if number in question_map:

            q = question_map[number]

            print()
            print(
                f"Q{number}:"
            )

            print(
                q["question"]
            )

        else:

            print()
            print(
                f"Q{number}: MISSING"
            )

def main():

    print("=" * 70)
    print("TaxAssist - Question Dataset Builder")
    print("=" * 70)

    if not INPUT_PDF.exists():

        print()
        print(
            "ERROR: PDF NOT FOUND"
        )

        print(
            f"Expected: {INPUT_PDF}"
        )

        raise FileNotFoundError(
            INPUT_PDF
        )


    text = extract_pdf_text(
        INPUT_PDF
    )


    text = clean_pdf_text(
        text
    )

    RAW_TEXT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RAW_TEXT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(text)

    print()
    print(
        f"Raw text saved to: "
        f"{RAW_TEXT_FILE}"
    )

    questions = extract_questions(
        text
    )


    questions = remove_duplicates(
        questions
    )


    questions.sort(
        key=lambda x: x["number"]
    )


    valid = validate_questions(
        questions
    )


    print_questions(
        questions
    )

    check_specific_questions(
        questions
    )

    if not valid:

        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)

        print(
            "Dataset validation FAILED."
        )

        print(
            "Questions were NOT saved."
        )

        print(
            "Check the extracted text and parser."
        )

        raise ValueError(
            "Expected exactly 200 valid questions."
        )

    save_json(
        questions,
        OUTPUT_JSON
    )


    print()
    print("=" * 70)
    print("QUESTION DATASET CREATED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Total questions : {len(questions)}"
    )

    print(
        f"Output          : {OUTPUT_JSON}"
    )

    print()
    print("1-200 all questions found")

    print(
        "Multi-line questions converted to one line"
    )

    print(
        "Bangla text preserved"
    )

    print(
        "English text preserved"
    )

    print(
        "Mixed text preserved"
    )

    print(
        "No duplicate question numbers"
    )

    print()
    print(
        "DONE"
    )

if __name__ == "__main__":

    main()