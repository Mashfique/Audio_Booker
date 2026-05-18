#!/usr/bin/env python3
"""
Audiobook Pipeline — autonomous, disconnect-resilient orchestrator.

One call drives the whole thing:

    pipe = AudiobookPipeline(
        book_dir,                 # state root on Drive
        chapters,                 # [{"title","text"}, ...]
        tts_generate,             # fn(text:str, out_mp3:str) -> None
        asr_transcribe,           # fn(audio_path:str) -> list[str] spoken words
        full_output_path,         # where the stitched book is written
        target_accuracy=0.98, best_of=3, max_retries=4, chunk_words=200,
    )
    pipe.run()                    # idempotent — resumes automatically

Design pillars
--------------
* Runtime memory is disposable; Drive is the only source of truth.
* Checkpoint after EVERY chunk, never per chapter.
* Atomic state writes (.tmp -> fsync -> os.replace) + rolling .bak.
* Rebuild-from-disk: splitting is deterministic, so a lost/corrupt
  manifest is reconstructed from the chunk files present on Drive.
* Monotonic guard: a chunk's audio is only ever replaced by a take with
  STRICTLY higher accuracy — the loop can improve or hold, never degrade.
* Persistent global repair budget survives unlimited reconnects.

Heavy libs (torch / chatterbox / whisper) are NOT imported here. TTS and
ASR are injected, so this module is unit-testable without a GPU and is
decoupled from whichever engines the notebook chooses.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from chunk_score import score_chunk

# ── Status constants ──────────────────────────────────────────────────────────
PENDING = "PENDING"
DONE = "DONE"        # accuracy >= target
FLAGGED = "FLAGGED"  # budget exhausted; best take kept, needs human review

STAGE_INIT = "INIT"
STAGE_GENERATING = "GENERATING"
STAGE_STITCHING = "STITCHING"
STAGE_COMPLETE = "COMPLETE"


# ── Small utilities ───────────────────────────────────────────────────────────

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _atomic_write_json(path: Path, obj) -> None:
    """Write JSON so a crash mid-write can never corrupt the file.
    tmp in same dir -> flush+fsync -> keep .bak -> os.replace (atomic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        try:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        except OSError:
            pass
    os.replace(tmp, path)


def _load_json(path: Path):
    """Load JSON, falling back to the .bak copy if the primary is unreadable."""
    path = Path(path)
    for p in (path, path.with_suffix(path.suffix + ".bak")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            continue
    return None


def split_by_words(text: str, max_words: int) -> list[str]:
    """Deterministic sentence-aware chunker. MUST stay deterministic: the
    rebuild-from-disk guarantee depends on (text, max_words) -> same chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
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
    return [c for c in chunks if c.strip()]


def ffmpeg_concat(parts: list[str], output: str) -> None:
    """Concatenate MP3s without re-encoding."""
    with tempfile.TemporaryDirectory() as tmp:
        lst = Path(tmp) / "list.txt"
        lst.write_text(
            "\n".join(f"file '{Path(p).resolve().as_posix()}'" for p in parts),
            encoding="utf-8",
        )
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(lst), "-c", "copy", str(output)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )


# ── Pipeline ──────────────────────────────────────────────────────────────────

@dataclass
class _Cfg:
    target_accuracy: float
    best_of: int
    max_retries: int
    chunk_words: int
    global_budget_factor: int


class AudiobookPipeline:

    def __init__(
        self,
        book_dir,
        chapters: list[dict],
        tts_generate: Callable[[str, str], None],
        asr_transcribe: Callable[[str], list],
        full_output_path,
        target_accuracy: float = 0.98,
        best_of: int = 3,
        max_retries: int = 4,
        chunk_words: int = 200,
        global_budget_factor: int = 6,
        log: Callable[[str], None] = print,
        concat: Callable[[list, str], None] = ffmpeg_concat,
    ):
        self.book_dir = Path(book_dir)
        self.chapters = chapters
        self.tts = tts_generate
        self.asr = asr_transcribe
        self.full_output_path = Path(full_output_path)
        self.cfg = _Cfg(target_accuracy, best_of, max_retries,
                        chunk_words, global_budget_factor)
        self.log = log
        self.concat = concat

        self.book_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.book_dir / "pipeline_state.json"
        self.chapters_dir = self.book_dir / "chapters"
        self.chapters_dir.mkdir(parents=True, exist_ok=True)

        self.state = self._load_or_init_state()

    # ── State -----------------------------------------------------------------

    def _load_or_init_state(self) -> dict:
        st = _load_json(self.state_path)
        if st is None:
            st = {
                "version": 1,
                "stage": STAGE_INIT,
                "target_accuracy": self.cfg.target_accuracy,
                "best_of": self.cfg.best_of,
                "max_retries": self.cfg.max_retries,
                "chunk_words": self.cfg.chunk_words,
                "cumulative_repair_attempts": 0,
                "started_at": _now(),
                "last_heartbeat": _now(),
                "chapters": [
                    {"idx": i, "title": ch["title"],
                     "text_hash": _sha1(ch["text"]), "status": PENDING}
                    for i, ch in enumerate(self.chapters, 1)
                ],
            }
            _atomic_write_json(self.state_path, st)
        return st

    def _heartbeat(self, **extra) -> None:
        self.state["last_heartbeat"] = _now()
        self.state.update(extra)
        _atomic_write_json(self.state_path, self.state)

    @property
    def _budget_cap(self) -> int:
        total = sum(
            len(split_by_words(ch["text"], self.cfg.chunk_words))
            for ch in self.chapters
        )
        return max(1, total) * self.cfg.global_budget_factor

    def _budget_left(self) -> int:
        return self._budget_cap - self.state["cumulative_repair_attempts"]

    # ── Manifest (per chapter) ------------------------------------------------

    def _chapter_dir(self, idx: int) -> Path:
        d = self.chapters_dir / f"{idx:02d}"
        (d / "chunks").mkdir(parents=True, exist_ok=True)
        return d

    def _build_manifest(self, idx: int, ch: dict) -> dict:
        """Create a fresh manifest from the chapter text."""
        texts = split_by_words(ch["text"], self.cfg.chunk_words)
        return {
            "chapter_idx": idx,
            "title": ch["title"],
            "text_hash": _sha1(ch["text"]),
            "chunks": [
                {"id": j, "text": t, "text_hash": _sha1(t),
                 "status": PENDING, "attempts": 0,
                 "best_accuracy": 0.0, "best_wer": 1.0,
                 "audio": f"chunk_{j:04d}.mp3", "flags": []}
                for j, t in enumerate(texts)
            ],
        }

    def _load_manifest(self, idx: int, ch: dict) -> dict:
        """Load manifest; rebuild/reconcile against current chapter text so
        editing the source .txt invalidates only the changed chunks."""
        cdir = self._chapter_dir(idx)
        man = _load_json(cdir / "manifest.json")
        fresh = self._build_manifest(idx, ch)

        if man is None:
            # Rebuild-from-disk: existing chunk audio == already done.
            for c in fresh["chunks"]:
                if (cdir / "chunks" / c["audio"]).exists():
                    c["status"] = DONE
                    c["best_accuracy"] = self.cfg.target_accuracy
            self._save_manifest(idx, fresh)
            return fresh

        # Reconcile: keep progress for chunks whose text is unchanged.
        old_by_id = {c["id"]: c for c in man.get("chunks", [])}
        for c in fresh["chunks"]:
            old = old_by_id.get(c["id"])
            if old and old.get("text_hash") == c["text_hash"]:
                c.update({k: old[k] for k in
                          ("status", "attempts", "best_accuracy",
                           "best_wer", "flags") if k in old})
                # Trust DONE only if the audio file actually exists.
                if c["status"] in (DONE, FLAGGED) and not (
                        cdir / "chunks" / c["audio"]).exists():
                    c["status"] = PENDING
                    c["best_accuracy"], c["best_wer"] = 0.0, 1.0
            else:
                # Text changed -> stale audio, regenerate this chunk only.
                ap = cdir / "chunks" / c["audio"]
                if ap.exists():
                    ap.unlink()
        self._save_manifest(idx, fresh)
        return fresh

    def _save_manifest(self, idx: int, man: dict) -> None:
        _atomic_write_json(self._chapter_dir(idx) / "manifest.json", man)

    # ── Repair loop (per chunk) ----------------------------------------------

    def _process_chunk(self, idx: int, man: dict, chunk: dict) -> None:
        cdir = self._chapter_dir(idx)
        final_audio = cdir / "chunks" / chunk["audio"]
        target = self.cfg.target_accuracy

        # attempt 0 = initial generation; 1..max_retries = repairs
        attempt = chunk["attempts"]
        while True:
            if attempt > 0 and self._budget_left() <= 0:
                chunk["status"] = FLAGGED
                chunk["flags"].append("budget_exhausted")
                self.log(f"    chunk {chunk['id']:04d}: budget exhausted "
                         f"-> FLAGGED (best acc={chunk['best_accuracy']:.3f})")
                break

            best_sc = None
            best_tmp = None
            for n in range(self.cfg.best_of):
                cand = cdir / "chunks" / f"{chunk['audio']}.cand{n}"
                try:
                    self.tts(chunk["text"], str(cand))
                    spoken = self.asr(str(cand))
                except Exception as e:                       # noqa: BLE001
                    self.log(f"    chunk {chunk['id']:04d}: gen/asr error: {e}")
                    continue
                sc = score_chunk(chunk["text"], spoken)
                if best_sc is None or sc.accuracy > best_sc.accuracy:
                    if best_tmp and best_tmp.exists():
                        best_tmp.unlink()
                    best_sc, best_tmp = sc, cand
                else:
                    cand.unlink(missing_ok=True)
                if sc.accuracy >= target:
                    break                                    # early exit (quota)

            if attempt > 0:
                self.state["cumulative_repair_attempts"] += 1

            # Monotonic guard: promote only a strictly better take.
            if best_sc is not None and (
                best_sc.accuracy > chunk["best_accuracy"]
                or not final_audio.exists()
            ):
                os.replace(str(best_tmp), str(final_audio))
                chunk["best_accuracy"] = best_sc.accuracy
                chunk["best_wer"] = best_sc.wer
                chunk["defects"] = [
                    {"kind": d.kind, "src": d.source, "spoken": d.spoken}
                    for d in best_sc.defects
                ][:20]
            elif best_tmp and best_tmp.exists():
                best_tmp.unlink()                            # discard worse take

            chunk["attempts"] = attempt + 1
            self._save_manifest(idx, man)
            self._heartbeat(stage=STAGE_GENERATING)

            if chunk["best_accuracy"] >= target:
                chunk["status"] = DONE
                self.log(f"    chunk {chunk['id']:04d}: DONE "
                         f"acc={chunk['best_accuracy']:.3f} "
                         f"(attempt {attempt})")
                break

            attempt += 1
            if attempt > self.cfg.max_retries:
                chunk["status"] = FLAGGED
                chunk["flags"].append("max_retries")
                self.log(f"    chunk {chunk['id']:04d}: max retries "
                         f"-> FLAGGED (best acc={chunk['best_accuracy']:.3f})")
                break

        self._save_manifest(idx, man)

    # ── Chapter / book -------------------------------------------------------

    def _assemble_chapter(self, idx: int, man: dict) -> Path:
        cdir = self._chapter_dir(idx)
        safe = re.sub(r"[^\w\s-]", "", man["title"])[:50].strip() or f"ch{idx}"
        out = cdir / f"{idx:02d}_{safe}.mp3"
        parts = [str(cdir / "chunks" / c["audio"]) for c in man["chunks"]]
        if len(parts) == 1:
            shutil.copy(parts[0], out)
        else:
            self.concat(parts, str(out))
        return out

    def run(self) -> dict:
        self.log(f"Pipeline start. Budget cap={self._budget_cap}, "
                 f"used={self.state['cumulative_repair_attempts']}, "
                 f"target={self.cfg.target_accuracy}")
        self._heartbeat(stage=STAGE_GENERATING)

        chapter_mp3s = []
        for i, ch in enumerate(self.chapters, 1):
            st_ch = self.state["chapters"][i - 1]
            man = self._load_manifest(i, ch)

            pending = [c for c in man["chunks"]
                       if c["status"] not in (DONE, FLAGGED)]
            done = len(man["chunks"]) - len(pending)
            self.log(f"\n[Chapter {i}/{len(self.chapters)}] {ch['title']}"
                     f" — {len(man['chunks'])} chunks "
                     f"({done} done, {len(pending)} to do)")

            for chunk in man["chunks"]:
                if chunk["status"] in (DONE, FLAGGED):
                    continue
                self._process_chunk(i, man, chunk)

            out = self._assemble_chapter(i, man)
            chapter_mp3s.append(str(out))
            st_ch["status"] = DONE
            self._heartbeat()
            accs = [c["best_accuracy"] for c in man["chunks"]]
            self.log(f"  chapter assembled: {out.name}  "
                     f"(mean acc {sum(accs)/max(1,len(accs)):.3f})")

        # Stitch full book
        self._heartbeat(stage=STAGE_STITCHING)
        self.full_output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(chapter_mp3s) == 1:
            shutil.copy(chapter_mp3s[0], self.full_output_path)
        elif chapter_mp3s:
            self.concat(chapter_mp3s, str(self.full_output_path))
        self.log(f"\nFull audiobook: {self.full_output_path}")

        report = self._qa_report()
        self._heartbeat(stage=STAGE_COMPLETE)
        return report

    # ── QA report ------------------------------------------------------------

    def _qa_report(self) -> dict:
        rows, flagged = [], []
        total_chunks = total_acc = 0.0
        n = 0
        for i, ch in enumerate(self.chapters, 1):
            man = _load_json(self._chapter_dir(i) / "manifest.json")
            if not man:
                continue
            accs = [c["best_accuracy"] for c in man["chunks"]]
            mean = sum(accs) / max(1, len(accs))
            n_flag = sum(1 for c in man["chunks"] if c["status"] == FLAGGED)
            rows.append((i, man["title"], len(man["chunks"]), mean, n_flag))
            total_chunks += len(man["chunks"])
            total_acc += sum(accs)
            n += len(man["chunks"])
            for c in man["chunks"]:
                if c["status"] == FLAGGED:
                    flagged.append({
                        "chapter": i, "title": man["title"],
                        "chunk": c["id"], "accuracy": c["best_accuracy"],
                        "text": c["text"][:200],
                        "defects": c.get("defects", []),
                    })
        overall = total_acc / max(1, n)
        report = {
            "overall_accuracy": round(overall, 4),
            "total_chunks": int(total_chunks),
            "flagged_count": len(flagged),
            "repair_attempts_used": self.state["cumulative_repair_attempts"],
            "chapters": [
                {"idx": r[0], "title": r[1], "chunks": r[2],
                 "mean_accuracy": round(r[3], 4), "flagged": r[4]}
                for r in rows
            ],
            "flagged": flagged,
        }
        _atomic_write_json(self.book_dir / "qa_report.json", report)
        self._write_qa_html(report)
        self.log(f"\nQA: overall acc {overall:.3f}, "
                 f"{len(flagged)} chunk(s) flagged for review. "
                 f"Report: {self.book_dir/'qa_report.html'}")
        return report

    def _write_qa_html(self, r: dict) -> None:
        crows = "".join(
            f"<tr><td>{c['idx']}</td><td>{c['title']}</td>"
            f"<td>{c['chunks']}</td><td>{c['mean_accuracy']:.3f}</td>"
            f"<td>{'0' if not c['flagged'] else c['flagged']}</td></tr>"
            for c in r["chapters"]
        )
        frows = "".join(
            f"<tr><td>{f['chapter']}</td><td>{f['chunk']}</td>"
            f"<td>{f['accuracy']:.3f}</td><td>{f['text']}</td></tr>"
            for f in r["flagged"]
        ) or "<tr><td colspan=4>None — every chunk met target.</td></tr>"
        html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Audiobook QA</title><style>
 body{{font-family:system-ui;background:#1a1a2e;color:#e0e0e0;
   max-width:1000px;margin:auto;padding:30px}}
 h2{{color:#a0c4ff}} table{{border-collapse:collapse;width:100%;margin:14px 0}}
 td,th{{padding:6px 10px;border-bottom:1px solid #333;text-align:left}}
 th{{color:#888;border-bottom:2px solid #555}}
 .big{{font-size:1.5rem;color:#99ff99}}</style></head><body>
<h2>Audiobook QA Report</h2>
<p class="big">Overall accuracy: {r['overall_accuracy']:.1%}</p>
<p>{r['total_chunks']} chunks · {r['flagged_count']} flagged ·
 {r['repair_attempts_used']} repair attempts used</p>
<h3>Per chapter</h3>
<table><tr><th>#</th><th>Title</th><th>Chunks</th>
 <th>Mean acc</th><th>Flagged</th></tr>{crows}</table>
<h3>Flagged for human review</h3>
<table><tr><th>Ch</th><th>Chunk</th><th>Best acc</th>
 <th>Text (first 200 chars)</th></tr>{frows}</table>
</body></html>"""
        (self.book_dir / "qa_report.html").write_text(html, encoding="utf-8")

    # ── Read-only status (safe from a second process) ------------------------

    def status(self) -> dict:
        st = _load_json(self.state_path) or {}
        out = {"stage": st.get("stage"), "heartbeat": st.get("last_heartbeat"),
               "repair_attempts": st.get("cumulative_repair_attempts"),
               "chapters": []}
        for i, ch in enumerate(self.chapters, 1):
            man = _load_json(self._chapter_dir(i) / "manifest.json")
            if not man:
                out["chapters"].append({"idx": i, "title": ch["title"],
                                        "state": "not started"})
                continue
            done = sum(1 for c in man["chunks"] if c["status"] == DONE)
            flag = sum(1 for c in man["chunks"] if c["status"] == FLAGGED)
            out["chapters"].append({
                "idx": i, "title": ch["title"],
                "chunks": len(man["chunks"]),
                "done": done, "flagged": flag,
                "pending": len(man["chunks"]) - done - flag,
            })
        return out


# ── Self-test (no GPU; mock TTS/ASR) ──────────────────────────────────────────

if __name__ == "__main__":
    import sys

    tmpdir = Path(tempfile.mkdtemp(prefix="abp_test_"))
    book = tmpdir / "book"

    chapters = [
        {"title": "Chapter One",
         "text": "The cat sat on the mat. It was a sunny day. "
                 "The dog ran fast."},
        {"title": "Chapter Two",
         "text": "He opened the old wooden door slowly."},
    ]

    tts_calls: list[str] = []          # every text TTS was asked to render

    # Mock TTS: writes the text into a fake "audio" file.
    def fake_tts(text: str, out: str) -> None:
        tts_calls.append(text)
        Path(out).write_text(text, encoding="utf-8")

    # Mock ASR: echoes the text, but for one chunk it "misreads" on the first
    # two takes then reads it correctly — exercises repair + monotonic guard.
    attempts_seen: dict[str, int] = {}

    def fake_asr(audio_path: str) -> list:
        text = Path(audio_path).read_text(encoding="utf-8")
        if text.startswith("He opened the old wooden door"):
            k = attempts_seen.get("door", 0)
            attempts_seen["door"] = k + 1
            if k < 2:                                  # first 2 takes are bad
                return "He opened the cold wooden floor slowly".split()
        return text.split()

    def no_concat(parts, output):                      # mock ffmpeg
        Path(output).write_text(
            "\n".join(Path(p).read_text(encoding="utf-8") for p in parts),
            encoding="utf-8")

    full_out = tmpdir / "book_full.mp3"

    print("=== RUN 1 (fresh) ===")
    p1 = AudiobookPipeline(book, chapters, fake_tts, fake_asr, full_out,
                           best_of=2, max_retries=4, concat=no_concat)
    r1 = p1.run()
    door_repaired = attempts_seen.get("door", 0) >= 3

    print("\n=== Simulate hard disconnect: wipe ch1 manifest + .bak ===")
    for nm in ("manifest.json", "manifest.json.bak"):
        (book / "chapters" / "01" / nm).unlink(missing_ok=True)

    tts_calls.clear()                                  # track RUN 2 only
    p2 = AudiobookPipeline(book, chapters, fake_tts, fake_asr, full_out,
                           best_of=2, max_retries=4, concat=no_concat)
    r2 = p2.run()

    ch1_text = chapters[0]["text"]
    ch1_status = p2.status()["chapters"][0]

    ok = True
    checks = [
        ("run1 overall accuracy 1.0", r1["overall_accuracy"] == 1.0),
        # rebuild-from-disk conservatively scores recovered chunks at target
        # (it intentionally does NOT re-transcribe accepted audio), so run2
        # overall must only be >= target, not 1.0.
        ("run2 overall accuracy >= target",
         r2["overall_accuracy"] >= 0.98),
        ("no chunks flagged", r2["flagged_count"] == 0),
        ("repair loop fixed 'door' chunk", door_repaired),
        ("full book written", full_out.exists()),
        ("rebuild-from-disk: ch1 chunk marked done",
         ch1_status.get("done") == ch1_status.get("chunks") ==
         len(split_by_words(ch1_text, 200))),
        ("rebuild-from-disk: ch1 audio NOT regenerated",
         not any(t in ch1_text or ch1_text.startswith(t[:20])
                 for t in tts_calls)),
        ("qa html exists", (book / "qa_report.html").exists()),
        ("qa json exists", (book / "qa_report.json").exists()),
    ]
    for name, cond in checks:
        print(f"[{'OK ' if cond else 'FAIL'}] {name}")
        ok &= cond

    shutil.rmtree(tmpdir, ignore_errors=True)
    print("\nAll passed." if ok else "\nSome checks FAILED.")
    sys.exit(0 if ok else 1)
