#!/usr/bin/env python3
"""
Chapter Splitter
Splits a cleaned TTS text file into individual chapter .txt files.

Usage:
    python chapter_splitter.py input.txt output_folder/
"""

import re
import sys
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

# These repeat inside fantasy chapters and should NOT start a new file
SKIP_HEADINGS = {
    "Description", "Example", "Notes", "State", "Drive",
    "Guys", "Sex", "Final Words", "Dominance",
}

# ── Pre-processing fixes ──────────────────────────────────────────────────────

def fix_remaining_issues(text: str) -> str:
    """Fix issues not caught by tts_cleaner.py."""

    # Headings split across blank lines
    fixes = [
        ("Troubleshooting: Lack of Stamina and\n\nWeak Erection",
         "Troubleshooting: Lack of Stamina and Weak Erection"),
        ("Improving Your Testosterone and Sex\n\nDrive",
         "Improving Your Testosterone and Sex Drive"),
        ("The Continuously Orgasmic State\n\nState",
         "The Continuously Orgasmic State"),
        ("Final Words\n\nFinal Words",
         "Final Words"),
    ]
    for old, new in fixes:
        text = text.replace(old, new)

    # Remaining ALL-CAPS headings with special characters
    text = re.sub(
        r"LIMITING BELIEF #(\d+):\s*([A-Z][A-Z\s]+[A-Z])",
        lambda m: f"Limiting Belief Number {m.group(1)}: {m.group(2).title()}",
        text,
    )
    text = text.replace("DON'T LIVE IN FEAR", "Don't Live In Fear")
    text = re.sub(r"\n\nSEX\n\n", "\n\nSex\n\n", text)

    return text


# ── Heading detection ─────────────────────────────────────────────────────────

def is_heading(para: str) -> bool:
    """Return True if this paragraph looks like a chapter/section heading."""
    s = para.strip()
    if not s:
        return False
    if len(s) > 80:
        return False
    if "\n" in s:
        return False
    if s[-1] in ".!?,;":
        return False
    if re.match(r"^\d+\.", s):          # numbered list item
        return False
    if not s[0].isupper():
        return False
    if s in SKIP_HEADINGS:
        return False

    # Must be mostly Title Case — filters out sentence fragments
    words = s.split()
    cap_ratio = sum(1 for w in words if w[0].isupper()) / len(words)
    if cap_ratio < 0.5:
        return False

    return True


# ── Splitter ──────────────────────────────────────────────────────────────────

def split_chapters(input_path: str, output_dir: str):
    text = Path(input_path).read_text(encoding="utf-8")
    text = fix_remaining_issues(text)

    paragraphs = text.split("\n\n")

    chapters   = []        # list of (title, [paragraphs])
    current    = None      # (title, [paragraphs])

    for para in paragraphs:
        if is_heading(para):
            if current and len("\n\n".join(current[1]).split()) > 10:
                chapters.append(current)
            current = (para.strip(), [para])
        else:
            if current is not None:
                current[1].append(para)
            # text before first heading is silently dropped (copyright remnants)

    if current and len("\n\n".join(current[1]).split()) > 10:
        chapters.append(current)

    # ── Save files ────────────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#':>3}  {'File':55s}  {'Words':>6}")
    print("-" * 70)

    for i, (title, paras) in enumerate(chapters, 1):
        content    = "\n\n".join(paras).strip()
        safe_title = re.sub(r"[^\w\s-]", "", title)[:55].strip()
        safe_title = re.sub(r"\s+", "_", safe_title)
        filename   = f"{i:02d}_{safe_title}.txt"
        (out / filename).write_text(content, encoding="utf-8")
        words = len(content.split())
        print(f"{i:>3}. {filename:55s}  {words:>6,}")

    print("-" * 70)
    total_words = sum(len(open(f, encoding='utf-8').read().split())
                      for f in sorted(out.glob("*.txt")))
    print(f"{'Total: ' + str(len(chapters)) + ' chapters':>60s}  {total_words:>6,}")
    print(f"\nSaved to: {out.resolve()}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python chapter_splitter.py input.txt output_folder/")
        sys.exit(1)
    split_chapters(sys.argv[1], sys.argv[2])
