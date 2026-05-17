#!/usr/bin/env python3
"""
Whisper Transcribe
Transcribes MP3 files using WhisperX with word-level timestamps.
Outputs a .transcript.txt and a synced .html player per file.

Usage:
    python whisper_transcribe.py chapter_01.mp3
    python whisper_transcribe.py "C:\\path\\to\\mp3_folder"

Requirements:
    pip install whisperx
    ffmpeg on PATH
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import torch
    import whisperx
except ImportError:
    print("ERROR: whisperx not found. Run: pip install whisperx")
    sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_SIZE   = "base"    # tiny / base / small / medium — base works on 2GB VRAM
LANGUAGE     = "en"
BATCH_SIZE   = 16        # reduce if GPU runs out of memory
COMPUTE_TYPE = "float16" # float16 for GPU, int8 for CPU


# ── Transcription ─────────────────────────────────────────────────────────────

def load_model(device: str):
    compute = COMPUTE_TYPE if device == "cuda" else "int8"
    print(f"[+] Loading Whisper {MODEL_SIZE} on {device.upper()} ({compute})...")
    model = whisperx.load_model(MODEL_SIZE, device,
                                compute_type=compute, language=LANGUAGE)
    print("[+] Model loaded.\n")
    return model


def transcribe(model, mp3_path: Path, device: str) -> list[dict]:
    """Transcribe one MP3, return list of word dicts with timestamps."""
    print(f"[>] Transcribing: {mp3_path.name}")
    audio  = whisperx.load_audio(str(mp3_path))
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)

    print(f"    Aligning word timestamps...")
    align_model, metadata = whisperx.load_align_model(
        language_code=LANGUAGE, device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )

    # Flatten to list of word dicts: {word, start, end, score}
    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append({
                "word":  w.get("word", "").strip(),
                "start": round(w.get("start", 0), 3),
                "end":   round(w.get("end",   0), 3),
                "score": round(w.get("score", 1.0), 3),
            })
    return words


# ── Output: transcript .txt ───────────────────────────────────────────────────

def save_transcript(words: list[dict], out_path: Path):
    lines = []
    for w in words:
        flag = " ⚠" if w["score"] < 0.7 else ""
        lines.append(f"{w['start']:>8.2f}s  {w['word']}{flag}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    low = sum(1 for w in words if w["score"] < 0.7)
    print(f"    Transcript: {out_path.name}  ({len(words)} words, {low} flagged low-confidence)")


# ── Output: synced HTML player ────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body  {{ font-family: Georgia, serif; max-width: 860px; margin: 40px auto;
           padding: 0 20px; background: #1a1a2e; color: #e0e0e0; }}
  h2    {{ color: #a0c4ff; }}
  audio {{ width: 100%; margin: 16px 0; }}
  #text {{ line-height: 2.2; font-size: 1.15rem; }}
  .w    {{ cursor: pointer; padding: 1px 2px; border-radius: 3px;
           transition: background 0.1s; }}
  .w:hover  {{ background: #334; }}
  .active   {{ background: #4a90e2; color: #fff; border-radius: 4px; }}
  .low      {{ text-decoration: underline wavy #ff6b6b; }}
  #legend   {{ font-size: 0.85rem; color: #888; margin-top: 12px; }}
</style>
</head>
<body>
<h2>{title}</h2>
<audio id="audio" controls src="{audio_src}"></audio>
<div id="text">{word_spans}</div>
<p id="legend">
  <span style="background:#4a90e2;color:#fff;padding:2px 6px;border-radius:3px;">highlighted</span> = currently playing &nbsp;|&nbsp;
  <span style="text-decoration:underline wavy #ff6b6b;">underlined</span> = low confidence (may be mumbled/dropped)
</p>
<script>
const audio = document.getElementById('audio');
const words = {word_data};
let last = -1;

audio.addEventListener('timeupdate', () => {{
  const t = audio.currentTime;
  let idx = -1;
  for (let i = 0; i < words.length; i++) {{
    if (t >= words[i].s && t <= words[i].e) {{ idx = i; break; }}
  }}
  if (idx !== last) {{
    if (last >= 0) document.getElementById('w'+last)?.classList.remove('active');
    if (idx  >= 0) document.getElementById('w'+idx )?.classList.add('active');
    last = idx;
  }}
}});

document.querySelectorAll('.w').forEach(el => {{
  el.addEventListener('click', () => {{
    const s = parseFloat(el.dataset.s);
    audio.currentTime = s;
    audio.play();
  }});
}});
</script>
</body>
</html>"""


def save_html(words: list[dict], mp3_path: Path, out_path: Path):
    spans = []
    for i, w in enumerate(words):
        low_cls  = " low" if w["score"] < 0.7 else ""
        text     = w["word"].replace("&", "&amp;").replace("<", "&lt;")
        spans.append(
            f'<span id="w{i}" class="w{low_cls}" '
            f'data-s="{w["start"]}" data-e="{w["end"]}">{text}</span>'
        )
    word_spans = " ".join(spans)
    word_data  = json.dumps(
        [{"s": w["start"], "e": w["end"]} for w in words],
        separators=(",", ":"),
    )

    html = HTML_TEMPLATE.format(
        title      = mp3_path.stem,
        audio_src  = mp3_path.name,   # relative — HTML sits next to MP3
        word_spans = word_spans,
        word_data  = word_data,
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"    HTML player: {out_path.name}")


# ── Per-file processing ───────────────────────────────────────────────────────

def process_file(model, mp3_path: Path, device: str):
    words = transcribe(model, mp3_path, device)
    if not words:
        print(f"    [!] No words detected — skipping.")
        return

    save_transcript(words, mp3_path.with_suffix(".transcript.txt"))
    save_html(words, mp3_path, mp3_path.with_suffix(".html"))
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transcribe MP3s with WhisperX")
    parser.add_argument("input", help="MP3 file or folder of MP3s")
    parser.add_argument("--model", default=MODEL_SIZE,
                        choices=["tiny", "base", "small", "medium"],
                        help=f"Whisper model size (default: {MODEL_SIZE})")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if GPU is available")
    args = parser.parse_args()

    global MODEL_SIZE
    MODEL_SIZE = args.model

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("[!] Running on CPU — will be slower.\n")

    input_path = Path(args.input)
    if input_path.is_file():
        mp3s = [input_path]
    elif input_path.is_dir():
        mp3s = sorted(input_path.glob("*.mp3"))
        if not mp3s:
            print(f"ERROR: No MP3 files found in {input_path}")
            sys.exit(1)
        print(f"[+] Found {len(mp3s)} MP3(s) in {input_path}\n")
    else:
        print(f"ERROR: Not found: {input_path}")
        sys.exit(1)

    model = load_model(device)

    for mp3 in mp3s:
        process_file(model, mp3, device)

    print(f"[+] Done. {len(mp3s)} file(s) processed.")
    print(f"    Open the .html files in your browser to see the synced player.")


if __name__ == "__main__":
    main()
