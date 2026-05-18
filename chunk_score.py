#!/usr/bin/env python3
"""
Chunk Scorer
Aligns the *source* text of a chunk against the *spoken* words WhisperX
transcribed from that chunk's audio, then classifies every discrepancy so the
autonomous repair loop fixes real defects and ignores noise.

Classification:
  DROPPED  - source words with no spoken counterpart        -> REAL defect
  MISREAD  - source word replaced by a phonetically DISTANT  -> REAL defect
             word (the TTS said the wrong thing)
  ADDED    - spoken words with no source counterpart         -> REAL defect
             (TTS repeated/hallucinated, or stray noise)
  ARTIFACT - source word replaced by a phonetically NEAR     -> NOT a defect
             word (audio is fine; Whisper just spelled it
             differently, e.g. "Lanyon" -> "Lainian")

Accuracy = 1 - (DROPPED + ADDED + MISREAD tokens) / source_token_count
ARTIFACT substitutions do NOT count against accuracy.

Everything here is dependency-free and deterministic.

Public API:
    score_chunk(source_text, spoken_words)        -> ChunkScore
    words_from_transcript_file(path)              -> list[str]
    words_from_whisperx_result(result)            -> list[str]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from text_normalize import normalize_tokens, tokenize


# ── Phonetic helpers (dependency-free) ────────────────────────────────────────

_SOUNDEX_MAP = {
    **{c: "1" for c in "bfpv"},
    **{c: "2" for c in "cgjkqsxz"},
    **{c: "3" for c in "dt"},
    **{c: "4" for c in "l"},
    **{c: "5" for c in "mn"},
    **{c: "6" for c in "r"},
}


def soundex(word: str) -> str:
    """Classic Soundex code (letter + 3 digits). Crude but cheap; used only as
    one of several 'phonetically near' signals."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    first = w[0]
    tail = []
    prev = _SOUNDEX_MAP.get(first, "")
    for ch in w[1:]:
        code = _SOUNDEX_MAP.get(ch, "")
        if ch in "hw":
            # h, w do not break a run
            continue
        if code and code != prev:
            tail.append(code)
        if ch not in "aeiouy":
            prev = code
        else:
            prev = ""
    return (first + "".join(tail) + "000")[:4].upper()


def levenshtein(a: str, b: str) -> int:
    """Standard edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]


def char_ratio(a: str, b: str) -> float:
    """1.0 == identical, 0.0 == nothing in common (by edit distance)."""
    if not a and not b:
        return 1.0
    m = max(len(a), len(b))
    return 1.0 - levenshtein(a, b) / m if m else 1.0


def phonetically_near(a: str, b: str) -> bool:
    """True when two words are close enough that the audio is almost certainly
    fine and the difference is a Whisper spelling choice, not a TTS misread.

    Combined signal (any one is enough):
      - identical after normalization
      - same Soundex code
      - high character similarity (>= 0.72)
      - same first letter AND moderate similarity (>= 0.60)
    """
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    r = char_ratio(a, b)
    if r >= 0.72:
        return True
    if a and b and a[0] == b[0] and r >= 0.60:
        return True
    sa, sb = soundex(a), soundex(b)
    if sa and sa == sb:
        return True
    return False


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Defect:
    kind: str          # DROPPED | MISREAD | ADDED
    source: str        # original source words (display)
    spoken: str        # what was transcribed
    src_start: int     # source token index range (within chunk, normalized)
    src_end: int


@dataclass
class ChunkScore:
    accuracy: float                       # 0.0 - 1.0 (ARTIFACT not penalized)
    wer: float                            # raw word error rate (all subs count)
    n_source: int                         # normalized source token count
    defects: list[Defect] = field(default_factory=list)
    n_artifact: int = 0                   # near-miss subs ignored (Whisper)

    @property
    def passed(self) -> bool:
        return self.accuracy >= getattr(self, "_target", 0.98)

    def meets(self, target: float) -> bool:
        return self.accuracy >= target

    def summary(self) -> str:
        d = sum(1 for x in self.defects if x.kind == "DROPPED")
        m = sum(1 for x in self.defects if x.kind == "MISREAD")
        a = sum(1 for x in self.defects if x.kind == "ADDED")
        return (f"acc={self.accuracy:.3f} wer={self.wer:.3f} "
                f"DROPPED={d} MISREAD={m} ADDED={a} "
                f"artifact_ignored={self.n_artifact}")


# ── Transcript extraction helpers ─────────────────────────────────────────────

_TRANSCRIPT_LINE = re.compile(r"^\s*[\d.]+s\s+(.+?)(?:\s+⚠)?\s*$")


def words_from_transcript_file(path) -> list[str]:
    """Parse the '<time>s  word' format written by whisper_transcribe.py /
    notebook Cell 12 into a flat list of spoken words."""
    from pathlib import Path
    out: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _TRANSCRIPT_LINE.match(line)
        if m:
            out.append(m.group(1).strip())
    return out


def words_from_whisperx_result(result: dict) -> list[str]:
    """Flatten a WhisperX aligned result dict into spoken word strings."""
    out: list[str] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            t = (w.get("word") or "").strip()
            if t:
                out.append(t)
    return out


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_chunk(source_text: str, spoken_words) -> ChunkScore:
    """Compare a chunk's source text against the words transcribed from its
    audio. `spoken_words` may be a list of strings or a raw transcript string.
    """
    if isinstance(spoken_words, str):
        spoken_words = tokenize(spoken_words)

    # Normalized token streams drive the alignment (false-positive suppression).
    src_norm = normalize_tokens(source_text)
    spk_norm = normalize_tokens(" ".join(spoken_words))

    n_source = max(1, len(src_norm))
    matcher = SequenceMatcher(None, src_norm, spk_norm, autojunk=False)

    defects: list[Defect] = []
    err_tokens = 0          # accuracy-affecting (DROPPED + ADDED + MISREAD)
    raw_sub = raw_del = raw_ins = 0
    n_artifact = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        if tag == "delete":
            raw_del += (i2 - i1)
            err_tokens += (i2 - i1)
            defects.append(Defect(
                "DROPPED", " ".join(src_norm[i1:i2]), "", i1, i2))

        elif tag == "insert":
            raw_ins += (j2 - j1)
            err_tokens += (j2 - j1)
            defects.append(Defect(
                "ADDED", "", " ".join(spk_norm[j1:j2]), i1, i1))

        elif tag == "replace":
            raw_sub += max(i2 - i1, j2 - j1)
            src_span = src_norm[i1:i2]
            spk_span = spk_norm[j1:j2]

            # First test the span as a whole: joining handles N<->M word
            # splits/merges ("balderdash" <-> "boulder dash",
            # "cannot" <-> "can not"). If the joined forms are phonetically
            # near, the audio is fine and Whisper just split/spelled it
            # differently -> whole span is an ARTIFACT, no defect.
            if phonetically_near("".join(src_span), "".join(spk_span)):
                n_artifact += max(len(src_span), len(spk_span))
            else:
                # Genuinely different: pair word-by-word, leftovers drop/add.
                k = min(len(src_span), len(spk_span))
                for off in range(k):
                    s, t = src_span[off], spk_span[off]
                    if phonetically_near(s, t):
                        n_artifact += 1                  # audio fine, ignore
                    else:
                        err_tokens += 1
                        defects.append(Defect(
                            "MISREAD", s, t, i1 + off, i1 + off + 1))
                if len(src_span) > k:                     # extra source -> drop
                    extra = src_span[k:]
                    err_tokens += len(extra)
                    defects.append(Defect(
                        "DROPPED", " ".join(extra), "", i1 + k, i2))
                elif len(spk_span) > k:                   # extra spoken -> add
                    extra = spk_span[k:]
                    err_tokens += len(extra)
                    defects.append(Defect(
                        "ADDED", "", " ".join(extra), i2, i2))

    wer = (raw_sub + raw_del + raw_ins) / n_source
    accuracy = max(0.0, 1.0 - err_tokens / n_source)
    return ChunkScore(
        accuracy=round(accuracy, 4),
        wer=round(wer, 4),
        n_source=len(src_norm),
        defects=defects,
        n_artifact=n_artifact,
    )


# ── CLI / self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        from pathlib import Path
        src = Path(sys.argv[1]).read_text(encoding="utf-8")
        spoken = words_from_transcript_file(sys.argv[2])
        sc = score_chunk(src, spoken)
        print(sc.summary())
        for d in sc.defects:
            print(f"  [{d.kind:8s}] '{d.source}'  ->  '{d.spoken}'")
        sys.exit(0)

    # Self-test on realistic cases from the Jekyll & Hyde run.
    tests = [
        # name, source, spoken, expected dominant defect kind(s)
        ("clean match",
         "He blew out his candle and set forth.",
         "He blew out his candle and set forth.",
         set()),
        ("near word-split (artifact, ignored by design)",
         "Such unscientific balderdash, added the doctor.",
         "Such unscientific boulder dash added the doctor.",
         set()),                                   # ~0.73 near -> artifact
        ("misread (genuinely far)",
         "And then he conceived a spark of hope.",
         "And then he conceived a spark of soap.",
         {"MISREAD"}),
        ("artifact (near, proper noun)",
         "the great Doctor Lanyon had his house",
         "the great Doctor Lainian had his house",
         set()),                                   # near -> ignored
        ("dropped words",
         "He gave his friend a few seconds to recover his composure.",
         "He gave his friend to recover.",
         {"DROPPED"}),
        ("number style differs (no defect)",
         "It provided in case of the decease of Henry Jekyll in 1851.",
         "It provided in case of the decease of Henry Jekyll in "
         "eighteen fifty-one.",
         set()),
        ("homophone (no defect)",
         "He could not bear to see their faces there.",
         "He could not bare to see there faces their.",
         set()),
    ]

    all_ok = True
    for name, src, spk, expect in tests:
        sc = score_chunk(src, spk)
        kinds = {d.kind for d in sc.defects}
        ok = kinds == expect
        all_ok &= ok
        print(f"[{'OK ' if ok else 'FAIL'}] {name}")
        print(f"       {sc.summary()}")
        if not ok:
            print(f"       expected kinds={expect or '{}'}  got={kinds or '{}'}")
            for d in sc.defects:
                print(f"         [{d.kind}] '{d.source}' -> '{d.spoken}'")
    print("\nAll passed." if all_ok else "\nSome checks FAILED.")
