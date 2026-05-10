#!/usr/bin/env python3
"""
PDF to Audiobook — detects chapters, cleans text, converts each to an MP3.

Usage:
    python pdf_to_audiobook.py book.pdf

Requirements:
    pip install pdfplumber edge-tts
    ffmpeg on PATH  (winget install ffmpeg)

Voices:
    Run `edge-tts --list-voices` to see all options.
    Edit VOICE below to change the narrator.
"""

import asyncio
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("ERROR: pdfplumber not found. Run: pip install pdfplumber")
    sys.exit(1)

try:
    import edge_tts
except ImportError:
    print("ERROR: edge-tts not found. Run: pip install edge-tts")
    sys.exit(1)

if shutil.which("ffmpeg") is None:
    print("ERROR: ffmpeg not found. Install with: winget install ffmpeg")
    sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────────────

VOICE      = "en-US-AriaNeural"   # female US English — change to suit your book
RATE       = "+0%"                # speech speed: "+10%" faster, "-10%" slower
CHUNK_SIZE = 4000                 # chars per TTS request (keep under ~5000)
OUTPUT_DIR = Path("audiobook")

# Patterns that mark the start of a new chapter (case-insensitive, line start)
CHAPTER_PATTERNS = [
    r"^chapter\s+[\divxlcdm]+",   # Chapter 1 / Chapter IV
    r"^chapter\s+\w+",            # Chapter One
    r"^part\s+[\divxlcdm]+",      # Part 1 / Part II
    r"^part\s+\w+",               # Part One
    r"^\d+\.\s+[A-Z]",            # 1. Introduction
    r"^epilogue$",
    r"^prologue$",
    r"^introduction$",
    r"^preface$",
    r"^foreword$",
    r"^appendix",
    r"^conclusion$",
]


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    # Rejoin words broken by hyphen + newline ("impor-\ntant" → "important")
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Drop lines containing only digits (page numbers)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Collapse 3+ blank lines to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Strip non-printable control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def is_chapter_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80:
        return False
    return any(re.match(p, s, re.IGNORECASE) for p in CHAPTER_PATTERNS)


# ── PDF extraction ────────────────────────────────────────────────────────────

def find_repeated_lines(pdf, threshold: int = 5) -> set[str]:
    """Lines that appear on many pages are headers or footers — skip them."""
    counts: Counter = Counter()
    for page in pdf.pages:
        raw = page.extract_text() or ""
        seen = set()
        for line in raw.splitlines():
            s = line.strip()
            if s and s not in seen:
                counts[s] += 1
                seen.add(s)
    return {line for line, n in counts.items() if n >= threshold}


def extract_chapters(pdf_path: Path) -> list[dict]:
    """Return [{"title": str, "text": str}, …] — one entry per chapter."""
    chapters: list[dict] = []
    current: dict = {"title": "Front Matter", "text": ""}

    with pdfplumber.open(str(pdf_path)) as pdf:
        repeated = find_repeated_lines(pdf)

        for page in pdf.pages:
            raw = page.extract_text() or ""
            for line in raw.splitlines():
                stripped = line.strip()

                if stripped in repeated:
                    continue  # skip header / footer

                if is_chapter_heading(stripped):
                    if current["text"].strip():
                        chapters.append(current)
                    current = {"title": stripped, "text": ""}
                else:
                    current["text"] += line + "\n"

    if current["text"].strip():
        chapters.append(current)

    return chapters


# ── TTS helpers ───────────────────────────────────────────────────────────────

def split_into_chunks(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split text at sentence boundaries so no chunk exceeds ~size chars."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > size and current:
            chunks.append(current.strip())
            current = sentence + " "
        else:
            current += sentence + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks


async def _tts_chunk(text: str, path: Path):
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE)
    await communicate.save(str(path))


async def chapter_to_mp3(text: str, output_path: Path):
    text = clean_text(text)
    if not text:
        return

    chunks = split_into_chunks(text)

    if len(chunks) == 1:
        await _tts_chunk(chunks[0], output_path)
        return

    # Render each chunk to a temp file then concat with ffmpeg
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        parts: list[Path] = []

        for i, chunk in enumerate(chunks):
            part = tmp / f"part_{i:04d}.mp3"
            await _tts_chunk(chunk, part)
            parts.append(part)

        list_file = tmp / "list.txt"
        list_file.write_text(
            "\n".join(f"file '{p}'" for p in parts), encoding="utf-8"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(list_file), "-c", "copy", str(output_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_audiobook.py <book.pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)

    out_dir = OUTPUT_DIR / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[+] Book   : {pdf_path.name}")
    print(f"[+] Voice  : {VOICE}  (rate: {RATE})")
    print(f"[+] Output : {out_dir.resolve()}\n")

    print("[*] Extracting chapters…")
    chapters = extract_chapters(pdf_path)
    print(f"[+] Found {len(chapters)} chapter(s)\n")

    for i, ch in enumerate(chapters, 1):
        safe  = re.sub(r"[^\w\s-]", "", ch["title"])[:50].strip()
        fname = f"{i:02d}_{safe}.mp3"
        out   = out_dir / fname
        words = len(ch["text"].split())

        print(f"[{i}/{len(chapters)}] {ch['title']}  ({words:,} words)")
        await chapter_to_mp3(ch["text"], out)

        size_kb = out.stat().st_size // 1024 if out.exists() else 0
        print(f"    → {fname}  ({size_kb} KB)\n")

    print(f"[+] Done! {len(chapters)} MP3(s) in: {out_dir.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
