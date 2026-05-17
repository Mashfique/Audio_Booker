# Audio Book Project — Session Context

**GitHub:** https://github.com/Mashfique/Audio_Booker.git
**Active branch:** `dev` (merges to `master` when stable)

---

## What this project does

A full pipeline to convert a PDF book into a voice-cloned audiobook:

1. **`audio_grabber.py`** — captures system audio (WASAPI loopback) when browser plays, saves clips as MP3, prevents PC sleep
2. **`pdf_to_audiobook.py`** — local multi-engine PDF→audiobook converter (edge-tts / XTTS v2 / ElevenLabs stub)
3. **`pdf_to_audiobook_colab.ipynb`** — Google Colab notebook using Chatterbox TTS (Resemble AI, 2025) for voice cloning on a free T4 GPU
4. **`tts_cleaner.py`** — cleans raw PDF-extracted text for TTS (removes TOC, footnotes, ALL-CAPS, ligatures, etc.)
5. **`chapter_splitter.py`** — splits a cleaned .txt file into individual chapter .txt files
6. **`whisper_transcribe.py`** — transcribes MP3s using WhisperX, outputs word-level .transcript.txt and synced HTML player

---

## All files in repo

| File | Purpose |
|------|---------|
| `audio_grabber.py` | WASAPI loopback recorder (Windows) |
| `pdf_to_audiobook.py` | Local PDF→audiobook, supports edge-tts / XTTS v2 / ElevenLabs |
| `pdf_to_audiobook_colab.ipynb` | Colab notebook — Chatterbox TTS voice cloning |
| `tts_cleaner.py` | Cleans raw PDF text for TTS |
| `chapter_splitter.py` | Splits cleaned .txt into per-chapter files |
| `whisper_transcribe.py` | WhisperX transcription → .transcript.txt + synced .html |
| `requirements.txt` | Local deps: pyaudiowpatch, numpy, pdfplumber, edge-tts |
| `SESSION.md` | This file — project context for resuming sessions |
| `project_overview_for_notebooklm.txt` | Full project explanation for NotebookLM |
| `.gitignore` | Ignores recordings/, __pycache__/, *.pyc, *.wav, .env |

---

## Full workflow (end-to-end)

```
PDF
 → tts_cleaner.py          — clean text (remove TOC, fix headings, ligatures, etc.)
 → chapter_splitter.py     — split into per-chapter .txt files
 → [manual review]         — open .txt files, fix anything Gemini/AI missed
 → Colab Cell 4: TXT_DIR   — point to folder of .txt files on Drive
 → Colab Cell 9            — Chatterbox TTS converts each chapter → MP3
 → Colab Cell 10           — (optional) stitch all chapters into one MP3
 → whisper_transcribe.py   — verify audio quality, flag low-confidence words
 → [future] audio repair   — re-run TTS on bad sentences, splice with ffmpeg
```

---

## Colab notebook — cell map (current)

| Cell | What it does |
|------|-------------|
| 1 | Check GPU + Python version |
| 2 | Install packages — **must restart + re-run after first run** |
| 3 | Mount Google Drive |
| 4 | CONFIG: `TXT_DIR`, `VOICE_REF`, `OUTPUT_DIR`, `REF_SECONDS`, `CHUNK_WORDS` |
| 5 | Helper functions (text cleaning, chapter loading, ffmpeg utils) |
| 6 | Trim reference audio → 15s WAV at 22050Hz mono |
| 7 | Load Chatterbox TTS model (cached to Drive via `HF_HOME`) |
| 8 | Load chapters from `.txt` files in `TXT_DIR` |
| 9 | **Main loop** — per chapter: split → TTS chunks → WAV → MP3 → Drive. Auto-resumes from last completed chapter. Overall progress bar. |
| 10 | (Optional) Stitch all chapter MP3s into one full audiobook |
| 11 | Install WhisperX |
| 12 | Transcribe a chapter MP3 → word timestamps + .transcript.txt |

### Cell 2 — critical install order (never change)
```python
%pip install -q chatterbox-tts pdfplumber
%pip install -q torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
%pip install -q "numpy==1.26.4" --force-reinstall
# → Runtime → Restart session → re-run Cell 2 → continue from Cell 3
```

### Why those pins
| Package | Reason |
|---------|--------|
| torchvision cu124 | chatterbox forces torch 2.6.0; cu121 caps at 0.20.1 |
| torchaudio removed | version conflicts; replaced with scipy.io.wavfile |
| numpy==1.26.4 | numba (via librosa, via chatterbox) needs numpy ≤ 2.0 |

### Cell 7 — model cache (saves re-downloading 1-2 GB each session)
```python
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/audiobook/hf_cache"
```
Drive must be mounted (Cell 3) before Cell 7.

### Cell 4 — current config (txt-based workflow)
```python
TXT_DIR    = "/content/drive/MyDrive/audiobook/chapters"
VOICE_REF  = "/content/drive/MyDrive/audiobook/reference.mp3"
OUTPUT_DIR = "/content/drive/MyDrive/audiobook/output"
REF_SECONDS = 15
CHUNK_WORDS = 200
```

---

## User setup
- **Local GPU**: MX110 (2 GB VRAM) — too small for Chatterbox/XTTS; use Colab
- **Colab GPU**: T4 (16 GB VRAM) — free tier (quota resets daily)
- **Kaggle**: free T4 alternative — 30 hrs/week, use when Colab quota is spent
- **Reference audio**: 10-min clean MP3, trimmed to 15s WAV from 3s offset

---

## Known issues fixed

| Problem | Fix Applied |
|---------|-------------|
| ALL-CAPS words read letter-by-letter (e.g. "A C T") | `_fix_allcaps()` in clean_text() — Title Case, preserves acronyms |
| torchaudio undefined symbol | Removed torchaudio; use scipy.io.wavfile |
| torchvision::nms does not exist | Switch to cu124 index for torchvision |
| Numba needs NumPy ≤ 2.0 | Pin numpy==1.26.4 after torchvision install |
| pydub / audioop missing (Python 3.13+) | Removed pydub; ffmpeg subprocess instead |
| Stuttering / mumbled words | Root cause: PDF extraction artifacts. Fixed via tts_cleaner.py |
| Session timeout loses progress | Cell 9 auto-resumes — skips already-converted chapters |
| Model re-downloads every session | HF_HOME pointed to Google Drive (Cell 7) |

---

## Text cleaning pipeline (tts_cleaner.py)

Fixes applied to raw PDF text before TTS:

| Fix | Detail |
|-----|--------|
| Remove copyright + TOC | Everything before first section heading |
| Footnote markers | `[1]`, `[2]` etc. removed |
| Ligatures | ﬁ→fi, ﬂ→fl, ﬀ→ff etc. |
| Em/en dashes | — → `, ` and – → `-` |
| Smart quotes | `"` `"` → `"`, `'` `'` → `'` |
| ALL-CAPS headings | `THE PERPETUAL LOVEMAKER` → `The Perpetual Lovemaker` |
| Numbered embedded sections | `1. PSYCHOLOGICAL OVER PHYSICAL...` split as heading |
| Merged heading + body | `Introduction To The 2nd Edition Congratulations...` split |
| Word merges | `fourstep` → `four-step`, `cunningulus` → `cunnilingus` |
| Split headings | `The Four Elements...\n\nMethod` → joined |

**Usage:**
```bash
python tts_cleaner.py input.txt output.txt
```

---

## Chapter splitter (chapter_splitter.py)

Splits cleaned .txt into 93 numbered chapter files.

**Usage:**
```bash
python chapter_splitter.py cleaned.txt output_folder/
```

**Current book:** 93 chapters, 60,983 words total
Located at: `C:\Users\kazim\Documents\Claude\chapters\`

---

## WhisperX transcription (whisper_transcribe.py)

Transcribes MP3s → word-level timestamps.
Outputs per MP3:
- `.transcript.txt` — each word with timestamp + ⚠ flag if confidence < 0.7
- `.html` — browser player: words highlight as audio plays, low-confidence underlined in red

**Usage:**
```bash
pip install whisperx
python whisper_transcribe.py chapter_01.mp3         # single file
python whisper_transcribe.py "path/to/mp3_folder"   # whole folder
python whisper_transcribe.py chapter_01.mp3 --cpu   # force CPU (MX110 fallback)
```

**Models:** tiny / base (default) / small / medium
base model needs ~1GB VRAM — should work on MX110

---

## Local script usage

```bash
# PDF → audiobook (free, Microsoft voice)
python pdf_to_audiobook.py book.pdf

# PDF → audiobook (voice cloning, needs GPU ≥ 4GB)
python pdf_to_audiobook.py book.pdf --engine xtts --voice reference.mp3 --gpu

# Stitch chapters after converting
python pdf_to_audiobook.py book.pdf --stitch

# Stitch an existing folder of chapter MP3s
python pdf_to_audiobook.py --stitch-only "audiobook/book/"
```

---

## Current status

- ✅ Full text pipeline built and tested (tts_cleaner + chapter_splitter)
- ✅ 93 chapter .txt files ready at `C:\Users\kazim\Documents\Claude\chapters\`
- ✅ Colab notebook updated: txt-based input, resume logic, progress bar, model cache
- ✅ WhisperX transcription script built (not yet tested)
- ✅ Repo pushed to GitHub: https://github.com/Mashfique/Audio_Booker.git
- ⏳ **Next: test full pipeline with a small PDF in Colab before converting all 93 chapters**
- 🔲 Audio repair tool (re-run TTS on bad sentences, splice with ffmpeg) — planned

---

## Planned next steps

1. **Test with small PDF in Colab** — confirm txt workflow, progress bar, resume, and model cache all work end-to-end
2. **Build audio repair tool** — use WhisperX confidence scores to flag bad sentences, re-run TTS, splice with ffmpeg
3. **Build synced HTML player** (whisper_transcribe.py already outputs this)
4. **ElevenLabs engine** — stub exists in pdf_to_audiobook.py, implement when needed
