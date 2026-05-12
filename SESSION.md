# Audio Book Project — Session Context

## What this project does
Two Python tools:
1. **`audio_grabber.py`** — captures system audio (WASAPI loopback) when browser plays, saves clips as MP3, prevents PC sleep
2. **`pdf_to_audiobook.py`** — converts a PDF into per-chapter MP3s using TTS (edge-tts free default, XTTS v2 voice cloning, ElevenLabs stub)
3. **`pdf_to_audiobook_colab.ipynb`** — Google Colab version using **Chatterbox TTS** (Resemble AI, 2025) for voice cloning on a free T4 GPU

---

## Files
| File | Purpose |
|------|---------|
| `audio_grabber.py` | WASAPI loopback recorder |
| `pdf_to_audiobook.py` | Local multi-engine PDF→audiobook converter |
| `pdf_to_audiobook_colab.ipynb` | Colab notebook (Chatterbox TTS, T4 GPU) |
| `requirements.txt` | Local deps: pyaudiowpatch, numpy, pdfplumber, edge-tts |
| `.gitignore` | Ignores recordings/, __pycache__/, *.pyc, *.wav, .env |

---

## Colab notebook — cell map
| Cell | What it does |
|------|-------------|
| 1 | Check GPU + Python version |
| 2 | Install packages (chatterbox-tts, pdfplumber, torchvision cu124, numpy 1.26.4) |
| 3 | Mount Google Drive |
| 4 | **CONFIG** — set PDF_PATH, VOICE_REF, OUTPUT_DIR, REF_SECONDS, CHUNK_WORDS |
| 5 | Helper functions (text cleaning, chapter extraction, ffmpeg utils) |
| 6 | Trim reference audio to WAV |
| 7 | Load Chatterbox TTS model (~1-2 GB download on first run) |
| 8 | Extract chapters from PDF, print word counts |
| 9 | **Main conversion loop** — chapter → chunks → WAV → MP3 → Drive |
| 10 | (Optional) Stitch all chapter MP3s into one full audiobook |

### Cell 2 — critical install order (do not change)
```python
%pip install -q chatterbox-tts pdfplumber
%pip install -q torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
%pip install -q "numpy==1.26.4" --force-reinstall
# → Runtime → Restart session → re-run Cell 2 → continue from Cell 3
```

### Why those pins
- `chatterbox-tts` forces **torch 2.6.0**; cu121 wheels cap at torch 2.5.x → need **cu124**
- `torchvision==0.21.0+cu124` matches torch 2.6.0 (cu121 caps at 0.20.1)
- `torchaudio` removed entirely — version conflicts; replaced with `scipy.io.wavfile`
- `numpy==1.26.4` — numba (via librosa, via chatterbox) needs numpy ≤ 2.0; cu124 torchvision pulls 2.4

---

## User setup
- **Local GPU**: MX110 (2 GB VRAM) — too small for XTTS/Chatterbox; use Colab
- **Colab GPU**: T4 (16 GB VRAM) — free tier, enable at Runtime → Change runtime type → T4 GPU
- **Reference audio**: 10-min clean MP3, trimmed to 15s WAV starting at 3s offset
- **Drive paths** (set in Cell 4):
  - PDF: `/content/drive/MyDrive/audiobook/book.pdf`
  - Voice ref: `/content/drive/MyDrive/audiobook/reference.mp3`
  - Output: `/content/drive/MyDrive/audiobook/output`

---

## Known issues fixed
| Problem | Fix |
|---------|-----|
| `A C T` spoken letter-by-letter | `_fix_allcaps()` in `clean_text()` — converts ALL-CAPS words to Title Case, preserves known acronyms (NASA, FBI, etc.) |
| `torchaudio` undefined symbol | Removed torchaudio entirely; use `scipy.io.wavfile` to write WAV |
| `torchvision::nms does not exist` | Switch torchvision install to cu124 index |
| `Numba needs NumPy ≤ 2.0` | Pin `numpy==1.26.4` after torchvision install |
| pydub / audioop missing (Python 3.13+) | Removed pydub; use ffmpeg subprocess directly |

---

## Current status (as of last session)
- Cell 9 running: converting **19 chapters** to MP3 with Chatterbox TTS voice cloning
- Was at **chapter 2/19** when user stepped away
- MP3s saving directly to Google Drive as each chapter completes
- After Cell 9 finishes → run **Cell 10** to stitch into one full MP3 (optional)

---

## Planned next steps
1. **Two-step text pipeline** (agreed, not yet implemented):
   - Add **Cell 8b**: extract chapters → save as `.txt` files to Drive for manual review
   - Improve `clean_text()`: fix ligatures (`ﬁ`→`fi`, `ﬂ`→`fl`), strip footnote numbers, normalize smart quotes/em-dashes
   - Modify Cell 9 to read from `.txt` files if present, fall back to PDF otherwise
   - *Reason*: stuttering likely caused by PDF extraction anomalies (ligatures, footnote markers, headers bleeding into paragraphs)

2. **ElevenLabs engine** — currently a stub in `pdf_to_audiobook.py`

3. **Merge dev branch** into master when stable (project is not a git repo yet locally — was discussed but skipped)

---

## Local script usage
```bash
# Free, no reference audio needed
python pdf_to_audiobook.py book.pdf

# Voice cloning (needs local GPU ≥ 4 GB VRAM)
python pdf_to_audiobook.py book.pdf --engine xtts --voice reference.mp3 --gpu

# Stitch chapters after converting
python pdf_to_audiobook.py book.pdf --stitch

# Stitch an existing folder of chapter MP3s
python pdf_to_audiobook.py --stitch-only audiobook/book/
```
