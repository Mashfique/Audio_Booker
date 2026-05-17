#!/usr/bin/env python3
"""
TTS Text Cleaner
Cleans a raw PDF-extracted text file for audiobook TTS conversion
without altering original content.

Usage:
    python tts_cleaner.py input.txt output.txt
"""

import re
import sys
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

LIGATURES = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st",
}

KNOWN_ACRONYMS = {
    "DEVI", "LLC", "DNA", "RNA", "AI", "FBI", "CIA", "NASA",
    "USA", "UK", "US", "EU", "UN",
}

def to_title_case(text: str) -> str:
    """Convert ALL-CAPS text to Title Case, preserving known acronyms."""
    words = text.split()
    result = []
    for w in words:
        if w in KNOWN_ACRONYMS:
            result.append(w)
        else:
            result.append(w.capitalize())
    return " ".join(result)


# ── Main cleaner ──────────────────────────────────────────────────────────────

def clean(text: str) -> str:

    # ── 1. Remove Gutenberg header / copyright block / Table of Contents ────────
    # Strategy 1: standard Gutenberg delimiter (most reliable)
    gutenberg_start = re.search(
        r"\*{3}\s*START OF THE PROJECT GUTENBERG EBOOK[^\n]*\*{3}", text, re.IGNORECASE
    )
    if gutenberg_start:
        text = text[gutenberg_start.end():]
    else:
        # Strategy 2: look for a known first heading (Preface, Chapter, Adventure, etc.)
        first_heading = re.search(
            r"(?m)^(Preface|Chapter\s+[IVXLCDM\d]+|Adventure\s+[IVXLCDM\d]+|CHAPTER\s+[IVXLCDM\d]+|ADVENTURE\s+[IVXLCDM\d]+)\b",
            text,
        )
        if first_heading:
            text = text[first_heading.start():]

    # Also strip the Gutenberg footer
    gutenberg_end = re.search(
        r"\*{3}\s*END OF THE PROJECT GUTENBERG EBOOK[^\n]*\*{3}", text, re.IGNORECASE
    )
    if gutenberg_end:
        text = text[:gutenberg_end.start()]

    # ── 2. Remove footnote / citation markers ─────────────────────────────────
    text = re.sub(r"\[\d+\]", "", text)

    # ── 3. Ligatures ──────────────────────────────────────────────────────────
    for lig, rep in LIGATURES.items():
        text = text.replace(lig, rep)

    # ── 4. Normalize punctuation ──────────────────────────────────────────────
    text = text.replace("—", ", ")   # em-dash  →  ", "
    text = text.replace("–", "-")    # en-dash  →  hyphen
    text = text.replace("…", "...")  # … → ...
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("\xa0", " ")      # non-breaking space

    # ── 5. Fix known word-merge artifacts from PDF extraction ─────────────────
    text = text.replace("fourstep",   "four-step")
    text = text.replace("L L C",      "LLC")
    text = text.replace("cunningulus","cunnilingus")  # typo in original

    # ── 6. Fix headings split across blank lines ──────────────────────────────
    text = re.sub(
        r"The Four Elements of the Sex God\s*\n\s*\n\s*Method",
        "The Four Elements of the Sex God Method",
        text,
    )
    text = re.sub(
        r"DEVI GRAPHS[^\n]+\n\s*\n[^\n]*SEX GOD METHOD[^\n]*",
        "DEVI Graphs: An Easy Way to Understand the Sex God Method",
        text,
        flags=re.IGNORECASE,
    )

    # ── 7. Fix "Heading merged with body" (heading runs into first sentence) ───
    # "Introduction To The 2nd Edition Congratulations..." → split after heading
    text = re.sub(
        r"(Introduction To The 2nd Edition)\s+([A-Z])",
        r"\1\n\n\2",
        text,
    )

    # ── 8. Fix numbered ALL-CAPS sections embedded inside paragraphs ──────────
    # e.g. "They are: 1. PSYCHOLOGICAL OVER PHYSICAL STIMULATION A common..."
    # → "\n\n1. Psychological Over Physical Stimulation\n\nA common..."
    def split_numbered_allcaps(m):
        num   = m.group(1)
        caps  = m.group(2).strip()
        return f"\n\n{num}. {to_title_case(caps)}\n\n"

    text = re.sub(
        r"(\d+)\.\s+([A-Z][A-Z\s]{8,})\s+(?=[A-Z][a-z]|[A-Z] [a-z])",
        split_numbered_allcaps,
        text,
    )

    # ── 9. Fix "DOMINANCE EMOTION VARIETY IMMERSION" strung together ──────────
    text = re.sub(
        r"\bDOMINANCE\s+EMOTION\s+VARIETY\s+IMMERSION\b",
        "Dominance, Emotion, Variety, and Immersion",
        text,
    )

    # ── 10. Convert standalone ALL-CAPS lines (headings) to Title Case ─────────
    def fix_allcaps_line(m):
        line = m.group(1).strip()
        return f"\n\n{to_title_case(line)}\n\n"

    # Match lines between blank lines that are entirely uppercase
    text = re.sub(
        r"\n\n([A-Z][A-Z\s\-:\/]{3,}[A-Z])\n\n",
        fix_allcaps_line,
        text,
    )
    # Also catch single-word ALL-CAPS headings (e.g. "DOMINANCE", "PAIN")
    text = re.sub(
        r"\n\n([A-Z]{4,})\n\n",
        fix_allcaps_line,
        text,
    )

    # ── 11. Normalize whitespace ───────────────────────────────────────────────
    text = re.sub(r"[ \t]+",  " ",  text)   # multiple spaces → one
    text = re.sub(r" \n",     "\n", text)   # trailing spaces before newline
    text = re.sub(r"\n{3,}",  "\n\n", text) # 3+ blank lines → one blank line
    text = text.strip()

    return text


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python tts_cleaner.py input.txt output.txt")
        sys.exit(1)

    input_path  = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    original = input_path.read_text(encoding="utf-8")
    cleaned  = clean(original)
    output_path.write_text(cleaned, encoding="utf-8")

    print(f"Input :  {len(original):>10,} chars  |  {len(original.split()):>6,} words")
    print(f"Output:  {len(cleaned):>10,} chars  |  {len(cleaned.split()):>6,} words")
    print(f"Removed: {len(original)-len(cleaned):>10,} chars")
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
