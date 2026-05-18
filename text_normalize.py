#!/usr/bin/env python3
"""
Text Normalizer
Normalizes text so the *source* sent to TTS and the *transcript* produced by
WhisperX can be compared fairly. Without this, harmless differences (case,
punctuation, "Dr." vs "doctor", "1851" vs "eighteen fifty-one", "their" vs
"there") generate hundreds of false mismatches and the autonomous repair loop
chases noise instead of real defects.

Design goals:
  - Dependency-free and deterministic (same input -> same output, always).
  - Collapse number/currency runs to a single <num> token so different reading
    styles never flag, while a *dropped* number still shows as a deletion.
  - Expand contractions and abbreviations on both sides.
  - Canonicalize common homophones (Whisper often picks the wrong spelling
    for correctly-spoken audio).

Public API:
    tokenize(text)            -> list[str]   raw word tokens (display)
    normalize_tokens(text)    -> list[str]   canonical tokens (comparison)
    normalize_text(text)      -> str         canonical tokens joined by space
"""

from __future__ import annotations

import re

# ── Tokenization ──────────────────────────────────────────────────────────────
# A token is a run of letters/digits with optional internal apostrophes or
# hyphens (so "don't", "well-dressed", "1851" stay intact). Pure punctuation is
# dropped.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[''\-][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Split text into raw word tokens. Used for the human-readable diff."""
    return _TOKEN_RE.findall(text)


# ── Contractions ──────────────────────────────────────────────────────────────
# Expanded on both sides because TTS may say "do not" while Whisper writes
# "don't" (or vice versa). Mapped to the expanded form.
_CONTRACTIONS = {
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "you'd": "you would", "he's": "he is", "he'll": "he will",
    "he'd": "he would", "she's": "she is", "she'll": "she will",
    "she'd": "she would", "it's": "it is", "it'll": "it will",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "we'd": "we would", "they're": "they are", "they've": "they have",
    "they'll": "they will", "they'd": "they would", "that's": "that is",
    "that'll": "that will", "that'd": "that would", "there's": "there is",
    "there'll": "there will", "here's": "here is", "what's": "what is",
    "what're": "what are", "who's": "who is", "who'll": "who will",
    "where's": "where is", "when's": "when is", "why's": "why is",
    "how's": "how is", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "hasn't": "has not",
    "haven't": "have not", "hadn't": "had not", "doesn't": "does not",
    "don't": "do not", "didn't": "did not", "won't": "will not",
    "wouldn't": "would not", "shan't": "shall not", "shouldn't": "should not",
    "can't": "cannot", "couldn't": "could not", "mustn't": "must not",
    "mightn't": "might not", "needn't": "need not", "let's": "let us",
    "o'clock": "oclock", "ma'am": "maam",
}

# ── Abbreviations ─────────────────────────────────────────────────────────────
# Title/word abbreviations a TTS engine speaks in full.
_ABBREV = {
    "mr": "mister", "mrs": "missus", "ms": "miss", "dr": "doctor",
    "st": "saint", "mt": "mount", "jr": "junior", "sr": "senior",
    "vs": "versus", "etc": "etcetera", "no": "number", "fig": "figure",
    "vol": "volume", "ch": "chapter", "pg": "page", "approx": "approximately",
    "dept": "department", "govt": "government", "prof": "professor",
}

# ── Homophones / common Whisper spelling swaps ────────────────────────────────
# Map every member of a homophone group to one canonical token. This prevents
# false mismatches when the audio is correct but Whisper chose a different
# spelling. Kept conservative — only true sound-alikes.
_HOMOPHONE_GROUPS = [
    ("there", "their", "they're"),
    ("to", "too", "two"),
    ("your", "you're"),
    ("its", "it's"),
    ("hear", "here"),
    ("no", "know"),
    ("knew", "new"),
    ("not", "knot"),
    ("one", "won"),
    ("be", "bee"),
    ("by", "buy", "bye"),
    ("for", "four", "fore"),
    ("see", "sea"),
    ("son", "sun"),
    ("rite", "right", "write", "wright"),
    ("road", "rode", "rowed"),
    ("threw", "through"),
    ("weather", "whether"),
    ("which", "witch"),
    ("plain", "plane"),
    ("peace", "piece"),
    ("principal", "principle"),
    ("aloud", "allowed"),
    ("brake", "break"),
    ("cite", "sight", "site"),
    ("days", "daze"),
    ("flour", "flower"),
    ("hour", "our"),
    ("meat", "meet", "mete"),
    ("pair", "pare", "pear"),
    ("rain", "reign", "rein"),
    ("steal", "steel"),
    ("tail", "tale"),
    ("waist", "waste"),
    ("wait", "weight"),
    ("way", "weigh"),
    ("wood", "would"),
    ("aisle", "isle"),
    ("ate", "eight"),
    ("bare", "bear"),
    ("board", "bored"),
    ("cell", "sell"),
    ("dear", "deer"),
    ("fair", "fare"),
    ("heal", "heel"),
    ("him", "hymn"),
    ("hole", "whole"),
    ("mail", "male"),
    ("mind", "mined"),
    ("none", "nun"),
    ("oar", "or", "ore"),
    ("poor", "pour", "pore"),
    ("scene", "seen"),
    ("some", "sum"),
    ("stair", "stare"),
    ("toe", "tow"),
    ("vain", "vane", "vein"),
    ("ware", "wear", "where"),
    ("week", "weak"),
]
_HOMOPHONE = {}
for _group in _HOMOPHONE_GROUPS:
    _canon = _group[0]
    for _w in _group:
        _HOMOPHONE[_w] = _canon

# ── Number / currency detection ───────────────────────────────────────────────
# Any maximal run of number-ish tokens (digits, cardinals, ordinals, "hundred",
# "thousand", "and" inside a number, currency words) collapses to a single
# <num> token on BOTH sides. Different reading styles ("1851" vs "eighteen
# fifty-one" vs "one thousand eight hundred and fifty-one") then never flag,
# while a fully *dropped* number still shows as a deletion (source has <num>,
# transcript does not).
_ONES = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
}
_TENS = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
         "ninety"}
_SCALES = {"hundred", "thousand", "million", "billion", "trillion"}
_ORDINALS = {
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth", "twentieth", "thirtieth", "fortieth", "fiftieth",
    "sixtieth", "seventieth", "eightieth", "ninetieth", "hundredth",
    "thousandth", "millionth",
}
_CURRENCY = {"dollar", "dollars", "cent", "cents", "pound", "pounds",
             "penny", "pence", "shilling", "shillings", "pence",
             "euro", "euros", "franc", "francs"}
_NUMBER_WORDS = _ONES | _TENS | _SCALES | _ORDINALS | _CURRENCY

_NUM_PLACEHOLDER = "<num>"


def _is_number_token(tok: str) -> bool:
    """True if a (lowercased) token participates in a number expression."""
    if any(c.isdigit() for c in tok):
        return True
    base = tok.replace("-", "").replace("'", "")
    if base in _NUMBER_WORDS:
        return True
    # ordinals written as digits: 1st, 2nd, 3rd, 21st ...
    if re.fullmatch(r"\d+(st|nd|rd|th)", tok):
        return True
    return False


# ── Core normalization ────────────────────────────────────────────────────────

def _expand(tokens: list[str]) -> list[str]:
    """Lowercase, expand contractions + abbreviations, canonicalize homophones."""
    out: list[str] = []
    for raw in tokens:
        t = raw.lower()
        # strip a trailing standalone period style artifact already handled by
        # tokenizer; abbreviations arrive without their dot.
        if t in _CONTRACTIONS:
            out.extend(_CONTRACTIONS[t].split())
            continue
        if t in _ABBREV:
            out.append(_ABBREV[t])
            continue
        out.append(t)
    return out


def _collapse_numbers(tokens: list[str]) -> list[str]:
    """Replace every maximal run of number-ish tokens with one <num> token.

    'and' is absorbed only when sandwiched between number tokens
    (e.g. 'two hundred and fifty') so prose 'and' is untouched.
    """
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        if _is_number_token(tokens[i]):
            j = i + 1
            while j < n:
                if _is_number_token(tokens[j]):
                    j += 1
                elif (tokens[j] == "and" and j + 1 < n
                      and _is_number_token(tokens[j + 1])):
                    j += 2
                else:
                    break
            out.append(_NUM_PLACEHOLDER)
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return out


def normalize_tokens(text: str) -> list[str]:
    """Full canonical token list used for source-vs-transcript comparison."""
    toks = _expand(tokenize(text))
    toks = _collapse_numbers(toks)
    toks = [_HOMOPHONE.get(t, t) for t in toks]
    # Drop any now-empty tokens defensively.
    return [t for t in toks if t]


def normalize_text(text: str) -> str:
    """Canonical tokens joined by single spaces (handy for logging/debug)."""
    return " ".join(normalize_tokens(text))


# ── CLI / self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) >= 2:
        src = Path(sys.argv[1]).read_text(encoding="utf-8")
        print(normalize_text(src))
        sys.exit(0)

    # Quick self-test demonstrating false-positive suppression.
    samples = [
        ("Dr. Lanyon paid £1851 to Mr. Hyde.",
         "doctor lanyon paid <num> to mister hyde"),
        ("They're going to their house over there.",
         "they are going to there house over there"),
        ("It was 1851; he won one race.",
         "it was <num> he one <num> race"),
        ("I don't know whether the weather will hold.",
         "i do not no weather the weather will hold"),
    ]
    ok = True
    for raw, expected in samples:
        got = normalize_text(raw)
        flag = "OK " if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"[{flag}] {raw}")
        print(f"       got: {got}")
        if got != expected:
            print(f"       exp: {expected}")
    print("\nAll passed." if ok else "\nSome checks FAILED.")
