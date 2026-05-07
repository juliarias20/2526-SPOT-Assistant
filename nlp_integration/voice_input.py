"""
voice_input.py
--------------
Push-to-talk speech-to-text input for run.py.

Drop-in replacement for input() in the REPL when --voice is active.
Uses faster-whisper for fully local, offline transcription.

Usage (from run.py):
    python run.py --voice

Push-to-talk flow:
    1. Press Enter to start recording
    2. Speak your command
    3. Press Enter again to stop
    4. Transcript is displayed — confirm, retype, or press Enter to execute

Install (run once):
    pip install faster-whisper sounddevice numpy

No internet required after install — model weights are cached locally.
Run cache_models.py to pre-download the Whisper model before going offline.
"""

import sys
import os
import threading
import time
from pathlib import Path
from typing import Optional

# ── Availability checks ───────────────────────────────────────────────────────
try:
    import numpy as np
    import sounddevice as sd
    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False

VOICE_AVAILABLE = _AUDIO_AVAILABLE and _WHISPER_AVAILABLE

# ── Config ────────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "base.en"   # base.en: ~145MB, fast on CPU, English-only
SAMPLE_RATE        = 16000       # Hz — Whisper expects 16kHz
MAX_RECORD_SEC     = 15          # Safety ceiling — stops recording after this

# Store Whisper weights inside the repo rather than the HF Hub cache.
# This path is independent of HF_HUB_OFFLINE so offline mode doesn't block it.
_HERE = Path(__file__).parent
WHISPER_CACHE_DIR = _HERE / "models" / "whisper"


class VoiceInput:
    """
    Manages Whisper model loading and push-to-talk recording.

    Load once at startup, then call .listen() in place of input().
    Falls back to typed input automatically on any error.
    """

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        if not VOICE_AVAILABLE:
            _print_missing_deps()
            self.ready = False
            return

        print(f"[voice] Loading Whisper '{model_size}' model from {WHISPER_CACHE_DIR}...")
        try:
            WHISPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # download_root bypasses HF Hub lookup entirely — works fully offline
            # as long as cache_models.py has been run at least once while online.
            self._model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                download_root=str(WHISPER_CACHE_DIR),
            )
            self.ready = True
            print(f"[voice] Ready — press Enter to start/stop recording.")
        except Exception as e:
            print(f"[voice] Failed to load Whisper model: {e}")
            print(f"[voice] Falling back to typed input.")
            self.ready = False

    def listen(self, prompt: str = "  SPOT> ") -> str:
        """
        Push-to-talk input. Displays prompt, waits for Enter, records until
        Enter again, transcribes, and returns the transcript string.

        Falls back to a normal input() call on any error so the demo
        never crashes.
        """
        if not self.ready:
            return input(prompt)

        sys.stdout.write(prompt)
        sys.stdout.write("[voice: press Enter to record] ")
        sys.stdout.flush()

        try:
            input()  # wait for first Enter
        except (EOFError, KeyboardInterrupt):
            raise

        print("  🎤 Recording... (press Enter to stop)")

        # ── Record audio in a background thread ──────────────────────────────
        frames: list[np.ndarray] = []
        stop_event = threading.Event()

        def _record():
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            ) as stream:
                deadline = time.time() + MAX_RECORD_SEC
                while not stop_event.is_set() and time.time() < deadline:
                    chunk, _ = stream.read(SAMPLE_RATE // 10)  # 100ms chunks
                    frames.append(chunk.copy())
                if time.time() >= deadline:
                    print(f"\n  [voice] Max recording time ({MAX_RECORD_SEC}s) reached.")

        record_thread = threading.Thread(target=_record, daemon=True)
        record_thread.start()

        try:
            input()  # wait for second Enter
        except (EOFError, KeyboardInterrupt):
            stop_event.set()
            record_thread.join(timeout=1)
            raise

        stop_event.set()
        record_thread.join(timeout=2)

        if not frames:
            print("  [voice] No audio captured — falling back to typed input.")
            return input(prompt)

        audio = np.concatenate(frames, axis=0).flatten()
        duration = len(audio) / SAMPLE_RATE

        if duration < 0.3:
            print("  [voice] Recording too short — falling back to typed input.")
            return input(prompt)

        # ── Transcribe ────────────────────────────────────────────────────────
        print(f"  [voice] Transcribing ({duration:.1f}s)...")
        try:
            segments, _ = self._model.transcribe(
                audio,
                language="en",
                beam_size=5,
                vad_filter=True,          # suppress silence segments
                vad_parameters={
                    "min_silence_duration_ms": 300,
                },
            )
            transcript = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            print(f"  [voice] Transcription error: {e} — falling back to typed input.")
            return input(prompt)

        if not transcript:
            print("  [voice] Nothing detected — falling back to typed input.")
            return input(prompt)

        # ── Confirm ───────────────────────────────────────────────────────────
        print(f"\n  🗣  Heard: \"{transcript}\"")
        confirm = input("  Execute? [Enter=yes / type correction / 's' to skip]: ").strip()

        if confirm.lower() == "s":
            return ""                    # empty → REPL skips and loops
        elif confirm == "":
            return transcript            # confirmed as-is
        else:
            return confirm               # operator typed a correction


def _print_missing_deps() -> None:
    missing = []
    if not _AUDIO_AVAILABLE:
        missing.append("sounddevice numpy")
    if not _WHISPER_AVAILABLE:
        missing.append("faster-whisper")
    print("[voice] Missing dependencies:")
    for pkg in missing:
        print(f"         pip install {pkg}")
    print("[voice] Falling back to typed input.\n")


def check_microphone() -> bool:
    """Quick sanity check — returns True if a default input device exists."""
    if not _AUDIO_AVAILABLE:
        return False
    try:
        info = sd.query_devices(kind="input")
        return info is not None
    except Exception:
        return False