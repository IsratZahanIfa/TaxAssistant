from pathlib import Path
import re

INPUT_FILE = Path("data/Clean_Text/Income_Tax_Act_2023_unicode.txt")
OUTPUT_FILE = Path("data/Clean_Text/Income_Tax_Act_2023_clean.txt")


def clean_line(line):
    # Remove trailing/leading whitespace
    line = line.strip()

    # Remove excessive spaces
    line = re.sub(r"[ \t]+", " ", line)

    return line


def main():
    print("=" * 70)
    print("TaxAssist - Tax Act Text Cleaning")
    print("=" * 70)

    if not INPUT_FILE.exists():
        print(f"ERROR: File not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned = []

    for line in lines:
        line = clean_line(line)

        # Preserve page separators
        if line.startswith("="):
            cleaned.append(line)
            continue

        # Preserve PAGE markers
        if re.match(r"^PAGE\s+\d+$", line, re.IGNORECASE):
            cleaned.append(line)
            continue

        # Keep non-empty lines
        if line:
            cleaned.append(line)

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(cleaned))

    print()
    print("Cleaning completed successfully!")
    print(f"Input lines : {len(lines):,}")
    print(f"Output lines: {len(cleaned):,}")
    print()
    print(f"Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()