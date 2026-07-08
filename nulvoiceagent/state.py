"""Publish the agent's current phase to a tiny JSON file so an OPTIONAL avatar or
status widget can show what the agent is doing. Nothing in the framework requires
a reader — this just writes the file; wire up whatever you like (a desktop avatar,
a status bar, an OBS overlay) to poll it, or ignore it entirely.

States: "idle" | "listening" | "thinking" | "talking". Written atomically (temp
+ os.replace) so a reader never sees a torn file. Each turn runs in one
short-lived process that blocks through record -> brain -> play, so we write the
phase synchronously and drop back to "idle" when the turn ends. "talking" also
carries an `until` epoch (now + audio duration) as a backstop: if this process is
killed mid-speech, a reader can still revert to idle on its own.
"""
from __future__ import annotations
import io
import json
import os
import tempfile
import time
import wave

from . import config

STATE_FILE = config.CACHE_DIR / "state.json"


def write(state: str, until: float | None = None, mouth: float | None = None,
          level: float | None = None) -> None:
    obj: dict = {"state": state, "ts": time.time()}
    if until is not None:
        obj["until"] = until
    if mouth is not None:
        obj["mouth"] = mouth
    if level is not None:
        obj["level"] = level
    try:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(config.CACHE_DIR), prefix=".state.")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(tmp, STATE_FILE)
    except OSError:
        pass


def idle() -> None:
    write("idle")


def listening(level: float | None = None) -> None:
    # `level` (0..1 mic loudness) lets a reader animate a voice-input meter to the
    # live mic level while listening.
    write("listening", level=level)


def thinking() -> None:
    write("thinking")


def talking(duration: float) -> None:
    write("talking", until=time.time() + max(0.0, duration) + 0.5)


def wav_duration(wav: bytes) -> float:
    """Seconds of audio in a WAV blob (best-effort; 0 on parse failure)."""
    try:
        with wave.open(io.BytesIO(wav), "rb") as w:
            fr = w.getframerate() or 24000
            return w.getnframes() / float(fr)
    except (wave.Error, OSError, ValueError):
        return 0.0
