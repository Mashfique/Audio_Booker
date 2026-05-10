#!/usr/bin/env python3
"""
Audio Grabber — captures system audio (WASAPI loopback) when playback is
detected in the browser, saves each segment as an MP3, and keeps the PC
from sleeping while running.

Requirements:
    pip install pyaudiowpatch pydub numpy
    ffmpeg must be on PATH (https://ffmpeg.org/download.html)
"""

import ctypes
import os
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    print("ERROR: pyaudiowpatch not found. Run: pip install pyaudiowpatch")
    sys.exit(1)

try:
    from pydub import AudioSegment
except ImportError:
    print("ERROR: pydub not found. Run: pip install pydub")
    sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR         = Path("recordings")   # folder where MP3s are saved
CHUNK              = 1024                 # frames per buffer
SILENCE_THRESHOLD  = 200                  # RMS below this = silence (raise if noisy)
SILENCE_DURATION   = 4.0                  # seconds of silence before ending a clip
MIN_CLIP_DURATION  = 2.0                  # clips shorter than this are discarded
MP3_BITRATE        = "192k"


# ── Sleep prevention ──────────────────────────────────────────────────────────

_ES_CONTINUOUS     = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _prevent_sleep():
    ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
    print("[+] Sleep prevention ON")


def _allow_sleep():
    ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
    print("[+] Sleep prevention OFF")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rms(data: bytes) -> float:
    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(arr ** 2))) if len(arr) else 0.0


def _save_mp3(frames: list[bytes], sample_rate: int, channels: int) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = OUTPUT_DIR / f"clip_{ts}.wav"
    mp3_path = OUTPUT_DIR / f"clip_{ts}.mp3"

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)          # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    audio = AudioSegment.from_wav(str(wav_path))
    audio.export(str(mp3_path), format="mp3", bitrate=MP3_BITRATE)
    wav_path.unlink()

    duration = len(audio) / 1000
    size_kb  = mp3_path.stat().st_size // 1024
    print(f"    Saved: {mp3_path.name}  ({duration:.1f}s, {size_kb} KB)")
    return mp3_path


# ── Main grabber ──────────────────────────────────────────────────────────────

class AudioGrabber:
    def __init__(self):
        self.pa            = pyaudio.PyAudio()
        self._recording    = False
        self._frames: list[bytes] = []
        self._lock         = threading.Lock()

    # ------------------------------------------------------------------
    def _find_loopback(self) -> dict:
        """Return info dict for the default WASAPI loopback device."""
        try:
            device = self.pa.get_default_wasapi_loopback()
        except Exception as exc:
            print(f"ERROR finding loopback device: {exc}")
            print("Make sure audio is routed through your default output device.")
            sys.exit(1)

        if device is None:
            print("ERROR: No WASAPI loopback device found.")
            sys.exit(1)

        return device

    # ------------------------------------------------------------------
    def run(self):
        device      = self._find_loopback()
        sample_rate = int(device["defaultSampleRate"])
        channels    = device["maxInputChannels"]

        print(f"\n[+] Loopback device : {device['name']}")
        print(f"[+] Sample rate     : {sample_rate} Hz | Channels: {channels}")
        print(f"[+] Output folder   : {OUTPUT_DIR.resolve()}")
        print(f"[+] Silence trigger : {SILENCE_DURATION}s  |  RMS threshold: {SILENCE_THRESHOLD}")
        print("\n[*] Listening for browser audio… Press Ctrl+C to stop.\n")

        silence_start:  float | None = None
        clip_start:     float | None = None

        # ------------------------------------------------------------------
        def callback(in_data, frame_count, time_info, status):
            nonlocal silence_start, clip_start

            level    = _rms(in_data)
            audible  = level >= SILENCE_THRESHOLD

            if audible:
                if not self._recording:
                    self._recording = True
                    self._frames    = []
                    clip_start      = time.time()
                    print(f"[>] Audio detected (RMS={level:.0f}) — recording…")
                silence_start = None
                self._frames.append(in_data)
            else:
                if self._recording:
                    self._frames.append(in_data)          # buffer during silence gap

                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= SILENCE_DURATION:
                        duration              = time.time() - clip_start
                        self._recording       = False
                        frames_snap           = self._frames[:]
                        self._frames          = []
                        silence_start         = None

                        if duration >= MIN_CLIP_DURATION:
                            print(f"[=] Silence — saving clip ({duration:.1f}s)…")
                            threading.Thread(
                                target=_save_mp3,
                                args=(frames_snap, sample_rate, channels),
                                daemon=True,
                            ).start()
                        else:
                            print(f"[!] Clip too short ({duration:.1f}s) — discarded.")

            return (None, pyaudio.paContinue)

        # ------------------------------------------------------------------
        stream = self.pa.open(
            format             = pyaudio.paInt16,
            channels           = channels,
            rate               = sample_rate,
            input              = True,
            input_device_index = device["index"],
            frames_per_buffer  = CHUNK,
            stream_callback    = callback,
        )

        stream.start_stream()

        try:
            while stream.is_active():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[!] Stopping…")
        finally:
            # Save any in-progress recording
            if self._recording and self._frames and clip_start is not None:
                duration = time.time() - clip_start
                if duration >= MIN_CLIP_DURATION:
                    print(f"[=] Saving in-progress clip ({duration:.1f}s)…")
                    _save_mp3(self._frames, sample_rate, channels)

            stream.stop_stream()
            stream.close()
            self.pa.terminate()
            _allow_sleep()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _prevent_sleep()
    AudioGrabber().run()
