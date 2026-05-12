#!/usr/bin/env python3
"""
PDF to Audiobook — detects chapters, cleans text, converts each to an MP3.

Usage:
    python pdf_to_audiobook.py book.pdf                          # edge-tts (free, default)
    python pdf_to_audiobook.py book.pdf --engine xtts --voice ref.mp3   # XTTS v2 voice clone
    python pdf_to_audiobook.py book.pdf --stitch                 # convert + stitch into one MP3
    python pdf_to_audiobook.py --stitch-only <folder>            # stitch existing chapter folder

Engines:
    edge     — Microsoft neural TTS, free, no reference audio needed (default)
    xtts     — Coqui XTTS v2, clones a voice from --voice reference audio, runs locally
    elevenlabs — (coming soon)

Requirements:
    pip install pdfplumber edge-tts
    pip install TTS torch          # only needed for --engine xtts
    ffmpeg on PATH  (winget install ffmpeg)
"""

import argparse
import asyncio
import concurrent.futures
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


# ── Config ────────────────────────────────────────────────────────────────────

EDGE_VOICE      = "en-US-AriaNeural"  # run `edge-tts --list-voices` to browse
EDGE_RATE       = "+0%"               # "+10%" faster, "-10%" slower
EDGE_CHUNK_SIZE = 4000                # chars per edge-tts request

XTTS_CHUNK_WORDS = 200                # XTTS v2 works best under ~250 words per chunk
XTTS_REF_SECONDS = 15                 # seconds to trim from reference audio
XTTS_LANGUAGE    = "en"

OUTPUT_DIR = Path("audiobook")

CHAPTER_PATTERNS = [
    r"^chapter\s+[\divxlcdm]+",
    r"^chapter\s+\w+",
    r"^part\s+[\divxlcdm]+",
    r"^part\s+\w+",
    r"^\d+\.\s+[A-Z]",
    r"^epilogue$",
    r"^prologue$",
    r"^introduction$",
    r"^preface$",
    r"^foreword$",
    r"^appendix",
    r"^conclusion$",
]


# ── Text cleaning ─────────────────────────────────────────────────────────────

_KNOWN_ACRONYMS = {
    "AI", "ML", "UK", "US", "EU", "UN", "NATO", "FBI", "CIA", "NASA",
    "CEO", "CFO", "CTO", "HR", "IT", "PR", "ID", "OK", "TV", "PC",
    "USB", "PDF", "MP3", "TTS", "GPU", "CPU", "RAM", "API", "URL",
    "HTML", "CSS", "SQL", "GMT", "EST", "PST", "BC", "AD", "WWII", "WWI",
    "USA", "UK", "UAE", "PTSD", "DNA", "RNA", "VIP", "RSVP",
}

def _fix_allcaps(text: str) -> str:
    """Convert ALL-CAPS words that aren't real acronyms to Title Case so TTS reads them naturally."""
    def _replace(m: re.Match) -> str:
        w = m.group(0)
        return w if w in _KNOWN_ACRONYMS else w.capitalize()
    return re.sub(r"\b[A-Z]{3,}\b", _replace, text)


def clean_text(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)           # fix hyphenated line breaks
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # drop page numbers
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = _fix_allcaps(text)
    return text.strip()


def is_chapter_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80:
        return False
    return any(re.match(p, s, re.IGNORECASE) for p in CHAPTER_PATTERNS)


# ── PDF extraction ────────────────────────────────────────────────────────────

def find_repeated_lines(pdf, threshold: int = 5) -> set[str]:
    counts: Counter = Counter()
    for page in pdf.pages:
        raw = page.extract_text() or ""
        seen: set[str] = set()
        for line in raw.splitlines():
            s = line.strip()
            if s and s not in seen:
                counts[s] += 1
                seen.add(s)
    return {line for line, n in counts.items() if n >= threshold}


def extract_chapters(pdf_path: Path) -> list[dict]:
    chapters: list[dict] = []
    current: dict = {"title": "Front Matter", "text": ""}

    with pdfplumber.open(str(pdf_path)) as pdf:
        repeated = find_repeated_lines(pdf)
        for page in pdf.pages:
            raw = page.extract_text() or ""
            for line in raw.splitlines():
                stripped = line.strip()
                if stripped in repeated:
                    continue
                if is_chapter_heading(stripped):
                    if current["text"].strip():
                        chapters.append(current)
                    current = {"title": stripped, "text": ""}
                else:
                    current["text"] += line + "\n"

    if current["text"].strip():
        chapters.append(current)
    return chapters


# ── Shared audio helpers ──────────────────────────────────────────────────────

def split_by_chars(text: str, size: int) -> list[str]:
    """Split at sentence boundaries into chunks of ~size chars."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) > size and current:
            chunks.append(current.strip())
            current = s + " "
        else:
            current += s + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks


def split_by_words(text: str, max_words: int) -> list[str]:
    """Split at sentence boundaries into chunks of ~max_words words."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, count = [], [], 0
    for s in sentences:
        w = len(s.split())
        if count + w > max_words and current:
            chunks.append(" ".join(current))
            current, count = [s], w
        else:
            current.append(s)
            count += w
    if current:
        chunks.append(" ".join(current))
    return chunks


def ffmpeg_concat(parts: list[Path], output: Path):
    with tempfile.TemporaryDirectory() as tmp:
        lst = Path(tmp) / "list.txt"
        lst.write_text("\n".join(f"file '{p.resolve()}'" for p in parts), encoding="utf-8")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(lst), "-c", "copy", str(output)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )


def wav_to_mp3(wav: Path, mp3: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav), "-b:a", "192k", str(mp3)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    wav.unlink()


# ── Engine: edge-tts ──────────────────────────────────────────────────────────

async def _edge_chunk(text: str, path: Path):
    await edge_tts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE).save(str(path))


async def edge_chapter_to_mp3(text: str, output: Path):
    chunks = split_by_chars(text, EDGE_CHUNK_SIZE)
    if len(chunks) == 1:
        await _edge_chunk(chunks[0], output)
        return
    with tempfile.TemporaryDirectory() as tmp:
        parts = [Path(tmp) / f"p{i:04d}.mp3" for i in range(len(chunks))]
        for chunk, part in zip(chunks, parts):
            await _edge_chunk(chunk, part)
        ffmpeg_concat(parts, output)


# ── Engine: XTTS v2 ───────────────────────────────────────────────────────────

def _load_xtts(use_gpu: bool):
    """Import and load the XTTS v2 model. Called once, returned for reuse."""
    try:
        import torch
        from TTS.api import TTS as CoquiTTS
    except ImportError:
        print("ERROR: TTS or torch not found. Run: pip install TTS torch")
        sys.exit(1)

    if use_gpu:
        if not __import__("torch").cuda.is_available():
            print("[!] CUDA not available — falling back to CPU.")
            use_gpu = False
        else:
            vram = __import__("torch").cuda.get_device_properties(0).total_memory / 1e9
            if vram < 3.5:
                print(f"[!] GPU VRAM ({vram:.1f} GB) is too low for XTTS v2 — falling back to CPU.")
                use_gpu = False

    device = "cuda" if use_gpu else "cpu"
    print(f"[+] Loading XTTS v2 on {device.upper()} (first run downloads ~1.9 GB)…")
    tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print(f"[+] XTTS v2 loaded on {device.upper()}\n")
    return tts


def _prepare_reference(ref_audio: Path) -> Path:
    """Trim reference MP3 to XTTS_REF_SECONDS of WAV, saved next to original."""
    out_wav = ref_audio.with_stem(ref_audio.stem + "_ref_trimmed").with_suffix(".wav")
    if out_wav.exists():
        print(f"[+] Using cached reference: {out_wav.name}")
        return out_wav
    print(f"[+] Trimming reference audio to {XTTS_REF_SECONDS}s WAV…")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(ref_audio),
         "-ss", "3", "-t", str(XTTS_REF_SECONDS),
         "-ar", "22050", "-ac", "1", str(out_wav)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    print(f"[+] Reference saved: {out_wav.name}\n")
    return out_wav


def _xtts_chunk_sync(tts, text: str, ref_wav: str, out_wav: Path):
    """Synchronous XTTS inference for one chunk."""
    tts.tts_to_file(
        text=text,
        speaker_wav=ref_wav,
        language=XTTS_LANGUAGE,
        file_path=str(out_wav),
    )


async def xtts_chapter_to_mp3(tts, ref_wav: Path, text: str, output: Path):
    chunks = split_by_words(text, XTTS_CHUNK_WORDS)
    loop   = asyncio.get_event_loop()

    with tempfile.TemporaryDirectory() as tmp:
        parts: list[Path] = []
        for i, chunk in enumerate(chunks):
            wav  = Path(tmp) / f"p{i:04d}.wav"
            mp3  = Path(tmp) / f"p{i:04d}.mp3"
            print(f"      chunk {i+1}/{len(chunks)}…", end="\r")
            # Run blocking XTTS call in a thread so the event loop stays alive
            await loop.run_in_executor(
                None, _xtts_chunk_sync, tts, chunk, str(ref_wav), wav
            )
            wav_to_mp3(wav, mp3)
            parts.append(mp3)

        print()  # newline after \r progress
        if len(parts) == 1:
            shutil.copy(parts[0], output)
        else:
            ffmpeg_concat(parts, output)


# ── Engine: ElevenLabs (stub) ─────────────────────────────────────────────────

async def elevenlabs_chapter_to_mp3(text: str, output: Path, api_key: str, voice_id: str):
    raise NotImplementedError(
        "ElevenLabs engine is not implemented yet. Coming soon."
    )


# ── Stitching ─────────────────────────────────────────────────────────────────

def stitch_folder(folder: Path, output: Path):
    mp3s = sorted(folder.glob("*.mp3"))
    if not mp3s:
        print(f"ERROR: No MP3 files found in {folder}")
        sys.exit(1)
    print(f"\n[*] Stitching {len(mp3s)} chapter(s) → {output.name}")
    ffmpeg_concat(mp3s, output)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"[+] Saved: {output.resolve()}  ({size_mb:.1f} MB)")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="PDF to Audiobook converter")
    parser.add_argument("pdf", nargs="?",
                        help="Path to the PDF file")
    parser.add_argument("--engine", choices=["edge", "xtts", "elevenlabs"],
                        default="edge",
                        help="TTS engine to use (default: edge)")
    parser.add_argument("--voice", metavar="AUDIO",
                        help="Reference audio file for voice cloning (required for --engine xtts)")
    parser.add_argument("--gpu", action="store_true",
                        help="Try to run XTTS on GPU (default: CPU; MX110 2GB is too small)")
    parser.add_argument("--stitch", action="store_true",
                        help="Stitch all chapter MP3s into one file after converting")
    parser.add_argument("--stitch-only", metavar="FOLDER",
                        help="Skip conversion — stitch an existing chapter folder into one MP3")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found. Install with: winget install ffmpeg")
        sys.exit(1)

    # ── stitch-only mode ──────────────────────────────────────────────────────
    if args.stitch_only:
        folder = Path(args.stitch_only)
        if not folder.is_dir():
            print(f"ERROR: Folder not found: {folder}")
            sys.exit(1)
        stitch_folder(folder, folder.parent / f"{folder.name}_full.mp3")
        return

    # ── convert mode ─────────────────────────────────────────────────────────
    if not args.pdf:
        parser.print_help()
        sys.exit(1)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)

    if args.engine == "xtts" and not args.voice:
        print("ERROR: --engine xtts requires --voice <reference_audio.mp3>")
        sys.exit(1)

    out_dir = OUTPUT_DIR / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[+] Book   : {pdf_path.name}")
    print(f"[+] Engine : {args.engine}")
    print(f"[+] Output : {out_dir.resolve()}\n")

    # Prepare engine-specific resources
    tts_model = None
    ref_wav   = None

    if args.engine == "xtts":
        tts_model = _load_xtts(use_gpu=args.gpu)
        ref_wav   = _prepare_reference(Path(args.voice))

    print("[*] Extracting chapters…")
    chapters = extract_chapters(pdf_path)
    print(f"[+] Found {len(chapters)} chapter(s)\n")

    for i, ch in enumerate(chapters, 1):
        safe  = re.sub(r"[^\w\s-]", "", ch["title"])[:50].strip()
        fname = f"{i:02d}_{safe}.mp3"
        out   = out_dir / fname
        text  = clean_text(ch["text"])
        words = len(text.split())

        print(f"[{i}/{len(chapters)}] {ch['title']}  ({words:,} words)")

        if args.engine == "xtts":
            await xtts_chapter_to_mp3(tts_model, ref_wav, text, out)
        elif args.engine == "elevenlabs":
            await elevenlabs_chapter_to_mp3(text, out, api_key="", voice_id="")
        else:
            await edge_chapter_to_mp3(text, out)

        size_kb = out.stat().st_size // 1024 if out.exists() else 0
        print(f"    → {fname}  ({size_kb} KB)\n")

    print(f"[+] Done! {len(chapters)} MP3(s) in: {out_dir.resolve()}")

    if args.stitch:
        stitch_folder(out_dir, out_dir.parent / f"{pdf_path.stem}_full.mp3")


if __name__ == "__main__":
    asyncio.run(main())
