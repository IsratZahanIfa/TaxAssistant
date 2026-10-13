from pathlib import Path
from unicodeconverter import converter

INPUT_FILE = Path("data/Clean_Text/Income_Tax_Act_2023_raw.txt")
OUTPUT_FILE = Path("data/Clean_Text/Income_Tax_Act_2023_unicode.txt")


def repair_mojibake(text):
    """
    Repair common UTF-8 mojibake without changing normal English.
    """

    replacements = {
        "â€™": "’",
        "â€œ": "“",
        "â€": "”",
        "â€˜": "‘",
        "â€“": "–",
        "â€”": "—",
        "â€¦": "…",
        "â€”": "—",
        "Â": "",
        "ï‚¾": "—",
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    return text


def looks_like_bijoy(line):
    """
    Decide whether a line is likely to contain Bijoy Bangla.

    Typical Bijoy text contains sequences such as:
    evsjv, Avwg, AwZwi, msL¨v, KZ, cÖ, etc.
    """

    bijoy_patterns = [
        "evsjv",
        "Avwg",
        "AvZ",
        "AwZ",
        "msL",
        "KZ",
        "cÖ",
        "cÖK",
        "KZ©",
        "e¨",
        "wb",
        "‡",
        "ˆ",
        "¨",
        "©",
    ]

    return any(pattern in line for pattern in bijoy_patterns)


def convert_line(line):
    """
    Convert only Bijoy-containing lines.
    Preserve normal English lines.
    """

    if not line.strip():
        return line

    # Page markers and separators must remain unchanged
    if line.startswith("PAGE "):
        return line

    if set(line.strip()) <= {"="}:
        return line

    # Convert only likely Bijoy lines
    if looks_like_bijoy(line):
        try:
            converted = converter.convert_bijoy_to_unicode(line)
            return converted
        except Exception as e:
            print(f"Warning: conversion failed for line: {line[:80]}")
            print(f"Reason: {e}")
            return line

    # Normal English / other text
    return line


def main():

    print("=" * 70)
    print("TaxAssist - Bijoy/ANSI → Unicode Converter")
    print("=" * 70)

    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    print("Working...")

    if not INPUT_FILE.exists():
        print("ERROR: Input file does not exist.")
        return

    # IMPORTANT:
    # The raw extraction is UTF-8 text containing Bijoy-style characters.
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    lines = text.splitlines(keepends=True)

    converted_lines = []

    bijoy_count = 0
    english_count = 0

    for line in lines:

        if looks_like_bijoy(line):
            bijoy_count += 1
        else:
            english_count += 1

        converted = convert_line(line)

        # Repair mojibake punctuation after conversion
        converted = repair_mojibake(converted)

        converted_lines.append(converted)

    final_text = "".join(converted_lines)

    # Save explicitly as UTF-8
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        f.write(final_text)

    print()
    print("=" * 70)
    print("CONVERSION COMPLETED")
    print("=" * 70)

    print(f"Total lines      : {len(lines):,}")
    print(f"Bijoy lines      : {bijoy_count:,}")
    print(f"Other lines      : {english_count:,}")
    print()
    print(f"Unicode file:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()