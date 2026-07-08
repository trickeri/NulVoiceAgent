"""Text-to-speech out — POST to kokoromodel, play the WAV through PipeWire."""
from __future__ import annotations
import json
import os
import subprocess
import tempfile
import time
import urllib.request

from . import config, state

# Seconds the last speak() spent on synth (the kokoromodel POST), i.e. the latency
# before any sound — NOT the playback duration. The CLI reads this to log a
# per-turn `timing:` line. Reset on every speak().
last_synth_secs = 0.0


def synth(text: str, voice: str | None = None, pitch: str | None = None,
          speed: str | None = None) -> bytes:
    """Return WAV bytes for `text` from kokoromodel, using the active profile's
    voice/pitch/speed unless overridden."""
    payload: dict = {"text": text}
    v = voice if voice is not None else (config.VOICE or None)
    p = pitch if pitch is not None else (config.PITCH or None)
    s = speed if speed is not None else (config.SPEED or None)
    if v:
        payload["voice"] = v
    if p:
        payload["pitch"] = p
    if s:
        payload["speed"] = s
    req = urllib.request.Request(
        config.KOKORO_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _sink_present(name: str) -> bool:
    if not name or not _has("pw-cli"):
        return False
    try:
        out = subprocess.run(["pw-cli", "ls", "Node"], capture_output=True,
                             text=True, timeout=3).stdout
        return f'node.name = "{name}"' in out
    except (OSError, subprocess.SubprocessError):
        return False


def play(wav: bytes) -> None:
    """Play WAV via pw-play. Dry to the default sink unless config.TTS_SINK names a
    present PipeWire sink (e.g. your own effects chain), in which case it routes
    there; falls back to the default sink if that sink isn't present."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        f.write(wav)
        f.flush()
        if _has("pw-play"):
            cmd = ["pw-play"]
            if config.TTS_SINK and _sink_present(config.TTS_SINK):
                cmd += ["--target", config.TTS_SINK]
            cmd.append(f.name)
        else:
            cmd = ["paplay", f.name]
        subprocess.run(cmd, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def speak(text: str, voice: str | None = None, pitch: str | None = None,
          speed: str | None = None) -> None:
    global last_synth_secs
    text = (text or "").strip()
    if not text:
        return
    _s = time.monotonic()
    wav = synth(text, voice=voice, pitch=pitch, speed=speed)
    last_synth_secs = time.monotonic() - _s
    # Publish "talking" state (with the audio duration as a backstop) so an optional
    # avatar/status reader can animate while we speak, then drop back to idle.
    # play() blocks until the audio finishes, so this brackets the whole reply.
    state.talking(state.wav_duration(wav))
    try:
        play(wav)
    finally:
        state.idle()


def cue(kind: str) -> None:
    """Play the optional start/stop cue to the DEFAULT sink. No-op unless
    config.CUE_START/STOP point at existing sound files."""
    if not config.CUES:
        return
    path = config.CUE_START if kind == "start" else config.CUE_STOP
    if not path or not os.path.exists(path):
        return
    try:
        # Non-blocking (spawn) so recording starts immediately and the cue can't
        # delay/clip speech or hold the process before the stop handler is installed.
        subprocess.Popen(["pw-play", path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def _has(binary: str) -> bool:
    import shutil
    return shutil.which(binary) is not None
