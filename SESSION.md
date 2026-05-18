# Audio Book Project — Session Context

**GitHub:** https://github.com/Mashfique/Audio_Booker.git
**Active branch:** `dev` (merges to `master` when stable)

---

## What this project does

A full pipeline to convert a PDF book into a voice-cloned audiobook:

1. **`audio_grabber.py`** — captures system audio (WASAPI loopback) when browser plays, saves clips as MP3, prevents PC sleep
2. **`pdf_to_audiobook.py`** — local multi-engine PDF→audiobook converter (edge-tts / XTTS v2 / ElevenLabs stub)
3. **`pdf_to_audiobook_colab.ipynb`** — Google Colab notebook using Chatterbox TTS (Resemble AI, 2025) for voice cloning on a free T4 GPU
4. **`tts_cleaner.py`** — cleans raw PDF-extracted text for TTS (removes TOC, footnotes, ALL-CAPS, ligatures, Gutenberg headers)
5. **`chapter_splitter.py`** — splits a cleaned .txt file into individual chapter .txt files
6. **`whisper_transcribe.py`** — transcribes MP3s using WhisperX, outputs word-level .transcript.txt and synced HTML player
7. **`text_normalize.py`** — dependency-free normalizer (numbers→`<num>`, abbreviations, homophones, contractions) so source vs transcript compare fairly
8. **`chunk_score.py`** — per-chunk WER + classifies each mismatch DROPPED / MISREAD / ADDED / ARTIFACT with a phonetic gate
9. **`audiobook_pipeline.py`** — **autonomous, disconnect-resilient orchestrator**: generate → transcribe → score → auto-repair → splice → stitch, with per-chunk manifest checkpointing and rebuild-from-disk resume

---

## All files in repo

| File | Purpose |
|------|---------|
| `audio_grabber.py` | WASAPI loopback recorder (Windows) |
| `pdf_to_audiobook.py` | Local PDF→audiobook, supports edge-tts / XTTS v2 / ElevenLabs |
| `pdf_to_audiobook_colab.ipynb` | Colab notebook — Chatterbox TTS voice cloning |
| `tts_cleaner.py` | Cleans raw PDF text for TTS (+ Gutenberg header strip) |
| `chapter_splitter.py` | Splits cleaned .txt into per-chapter files |
| `whisper_transcribe.py` | WhisperX transcription → .transcript.txt + synced .html |
| `text_normalize.py` | Normalizer for fair source-vs-transcript compare (self-tested) |
| `chunk_score.py` | Per-chunk WER + DROPPED/MISREAD/ADDED/ARTIFACT classifier (self-tested) |
| `audiobook_pipeline.py` | Autonomous resilient orchestrator (self-tested, GPU-free) |
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
| 0 | Pull latest code from GitHub (clones `dev` branch) |
| 1 | Check GPU + Python version |
| 2 | Install packages (incl. `faster-whisper`) — **must restart + re-run after first run** |
| 3 | Mount Google Drive |
| 4 | CONFIG: `DRIVE_BASE`, `TXT_DIR`, `VOICE_REF`, `OUTPUT_DIR`, `FULL_OUTPUT_DIR`, `HF_CACHE_DIR`, `REF_SECONDS`, `CHUNK_WORDS`. Auto-creates all output dirs. |
| 4A | Prepare a new book — runs `tts_cleaner.py` + `chapter_splitter.py` on a raw .txt |
| 5 | Helper functions |
| 6 | Trim reference audio → 15s WAV at 22050Hz mono |
| 7 | Load Chatterbox TTS model (cached to Drive via `HF_HOME`) |
| **P** | **ONE-CLICK PIPELINE** — `AudiobookPipeline`: generate → faster-whisper transcribe → score → autonomous repair (best-of-3, ≤4 retries, target 0.98) → splice → stitch → QA report. Resumes after any disconnect. |
| **S** | Read-only status (safe in a 2nd Colab tab) |
| 8–13 | Legacy manual/debug tools (single-chapter convert, stitch, WhisperX inspect, manual diff). Not needed for one-click flow. |

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

## Autonomous pipeline architecture (audiobook_pipeline.py)

**Goal:** one-click upload→audiobook with a feedback loop that auto-irons out
dropped/misread content to a target accuracy, surviving frequent Colab
disconnects.

### Feedback loop (per chunk, not per chapter)
```
generate → faster-whisper transcribe → normalize both sides → score
  acc ≥ 0.98 (target)            → accept
  DROPPED / far MISREAD          → repair ↺ best-of-3, ≤4 retries
  phonetic-near / number / homophone → accept (Whisper artifact, no fix)
```
Escalation: best-of-N re-roll → (future) sentence isolation → flag for review.

### Why faster-whisper (not WhisperX) in the loop
WhisperX's alignment pulls torch and hits a circular-import when loaded
alongside Chatterbox (the `torch.fx` error we saw). The **per-chunk design
removes the need for word alignment** — each chunk is its own file, so we
only need its transcript text. `faster-whisper` (CTranslate2) gives that with
no torch conflict. WhisperX Cell 12 stays as a manual deep-inspect tool.

### Disconnect-resilience (5 mechanisms)
1. **Chunk-level checkpoint** — every accepted take is a real file on Drive
   immediately; a disconnect loses at most the one chunk in flight.
2. **Atomic manifest writes** — `.tmp`→fsync→`os.replace` + rolling `.bak`;
   `_load_json` auto-falls back to `.bak`.
3. **Rebuild-from-disk** — chunking is deterministic, so a wiped manifest is
   reconstructed from chunk files on Drive; existing audio is NOT regenerated.
4. **Edit-invalidation** — per-chunk `text_hash`; editing the source .txt
   regenerates only the changed chunks.
5. **Persistent repair budget** — cumulative attempt count in
   `pipeline_state.json`; cap survives unlimited reconnects.

**Monotonic guard:** a chunk's audio is replaced only by a strictly-better
take — the loop can improve or hold, never degrade good audio (critical with
non-deterministic Chatterbox).

### State layout on Drive (`DRIVE_BASE/pipeline/`)
```
pipeline_state.json            stage, chapter idx, repair budget, heartbeat
chapters/NN/manifest.json      per-chunk: id, text_hash, status, attempts,
                               best_accuracy, defects
chapters/NN/chunks/chunk_XXXX.mp3   accepted/best take (source of truth)
chapters/NN/NN_title.mp3       assembled when all chunks done
qa_report.html / .json         per-chapter accuracy + flagged-for-review
```

### Recovery after a disconnect
Re-run Cells **0,2,3,4,5,6,7 → P**. Cell P recomputes the resume point from
Drive and continues. Cell **S** shows status without touching the run.

### All three modules are self-tested (run locally, GPU-free)
`python text_normalize.py` · `python chunk_score.py` · `python audiobook_pipeline.py`

---

## Current status

- ✅ Text pipeline tested (tts_cleaner + chapter_splitter); Gutenberg header strip added
- ✅ First full Colab run completed end-to-end (Jekyll & Hyde, 11 chapters) — voice good, some TTS misreads observed (balderdash→boulder dash, Lanyon→Lainian)
- ✅ Cell 4 centralised config + auto-creates dirs; Cell 0 clones `dev`
- ✅ **Autonomous pipeline built + self-tested**: text_normalize.py, chunk_score.py, audiobook_pipeline.py
- ✅ Notebook one-click Cell P + status Cell S added; faster-whisper in Cell 2
- ✅ All pushed to GitHub `dev`: https://github.com/Mashfique/Audio_Booker.git
- ⏳ **Next: run Cell P end-to-end in Colab on a small book to validate autonomy + resume on real GPU**
- 🔲 Merge `dev` → `master` once Cell P is validated in Colab

---

## Planned next steps

1. **Validate Cell P in Colab** — small book; confirm repair loop reduces misreads, disconnect→resume works on real Drive, QA report is useful
2. **Tune thresholds** — phonetic-near cutoff, target accuracy, budget — based on real flagged counts
3. **Sentence-isolation repair strategy** — escalation step 2 (currently best-of-N re-roll only)
4. **Phonetic respelling dict** — auto-fix persistent proper-noun misreads (logged, reviewable)
5. **Merge dev → master**, set Cell 0 `BRANCH = "master"`
6. **ElevenLabs engine** — stub exists in pdf_to_audiobook.py, implement when needed
