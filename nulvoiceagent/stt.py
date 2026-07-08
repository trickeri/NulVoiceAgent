"""Speech-to-text in — capture the mic with parec, endpoint on silence, POST to
NulSpeech2Text. parec (PulseAudio/PipeWire) records 16 kHz mono s16le, the rate
the STT server expects.
"""
from __future__ import annotations
import io
import math
import re
import signal
import subprocess
import urllib.request
import wave

from . import config

_FRAME = 320            # samples per 20 ms frame at 16 kHz
_BYTES = _FRAME * 2     # s16le
_PREROLL = 10           # frames (~200 ms) kept before onset so the first syllable isn't clipped

_manual_stop = False


def _request_stop(_signum, _frame):
    global _manual_stop
    _manual_stop = True


def record_manual(max_secs: float | None = None, level_cb=None) -> bytes:
    """Record until a SIGUSR1 arrives (manual toggle stop) or max_secs as a safety
    cap. NO silence endpointing — the talker decides when it ends. Returns WAV
    bytes. After it returns, SIGUSR1 is ignored so a stray press during the
    follow-on processing can't kill the process.

    `level_cb`, if given, is called ~10×/s with the current mic loudness (0..1) so
    a UI (the taskbar voice indicator) can react to the mic while recording."""
    global _manual_stop
    _manual_stop = False
    max_secs = config.MANUAL_MAX_SECS if max_secs is None else max_secs
    signal.signal(signal.SIGUSR1, _request_stop)

    cmd = ["parec", "--format=s16le", f"--rate={config.SAMPLE_RATE}",
           "--channels=1", "--latency-msec=30"]
    if config.MIC_SOURCE:
        cmd.append(f"--device={config.MIC_SOURCE}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frames = bytearray()
    elapsed = 0.0
    frame_dt = _FRAME / config.SAMPLE_RATE
    nframe = 0
    try:
        # parec streams continuously, so each read returns within ~20 ms — the
        # loop re-checks _manual_stop every frame and exits promptly on SIGUSR1.
        while elapsed < max_secs and not _manual_stop:
            chunk = proc.stdout.read(_BYTES)
            if len(chunk) < _BYTES:
                break
            elapsed += frame_dt
            frames += chunk
            # Publish a mic level ~every 5 frames (~100 ms) so the indicator's bars
            # dance, without re-writing the state file at the full 50 Hz frame rate.
            if level_cb is not None:
                nframe += 1
                if nframe % 5 == 0:
                    try:
                        level_cb(_level(chunk))
                    except Exception:  # noqa: BLE001 — a UI hiccup must not stop recording
                        pass
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    return _wrap_wav(bytes(frames)) if frames else b""


def record(max_secs: float | None = None, onset_timeout: float | None = None,
           trail_silence: float | None = None) -> bytes:
    """Record one utterance with a hysteresis noise gate, return WAV bytes.

    Two thresholds (dBFS), like the OBS gate on this mic: the gate OPENS once the
    level sustains above GATE_OPEN_DB for ~GATE_ATTACK (so music/room noise below
    it never starts a capture), then stays open until the level holds below
    GATE_CLOSE_DB for `trail_silence` (end of turn). A short pre-roll is kept so
    the onset isn't clipped. Returns b"" if nothing crossed the open threshold.
    """
    max_secs = config.GATE_MAX_SECS if max_secs is None else max_secs
    onset_timeout = config.GATE_ONSET_TIMEOUT if onset_timeout is None else onset_timeout
    trail_silence = config.GATE_TRAIL if trail_silence is None else trail_silence
    open_db, close_db = config.GATE_OPEN_DB, config.GATE_CLOSE_DB

    cmd = ["parec", "--format=s16le", f"--rate={config.SAMPLE_RATE}",
           "--channels=1", "--latency-msec=30"]
    if config.MIC_SOURCE:
        cmd.append(f"--device={config.MIC_SOURCE}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frames = bytearray()
    preroll: list[bytes] = []
    started = False
    open_run = 0.0       # time sustained above OPEN before we commit onset
    silent_run = 0.0     # time held below CLOSE after onset
    elapsed = 0.0
    frame_dt = _FRAME / config.SAMPLE_RATE
    try:
        while elapsed < max_secs:
            chunk = proc.stdout.read(_BYTES)
            if len(chunk) < _BYTES:
                break
            elapsed += frame_dt
            db = _dbfs(chunk)
            if not started:
                if db >= open_db:
                    open_run += frame_dt
                    preroll.append(chunk)
                    if open_run >= config.GATE_ATTACK:
                        started = True
                        frames += b"".join(preroll)   # include pre-roll + onset
                        preroll.clear()
                else:
                    open_run = 0.0
                    preroll.append(chunk)
                    if len(preroll) > _PREROLL:
                        preroll.pop(0)
                    if elapsed >= onset_timeout:
                        break  # nobody spoke
            else:
                frames += chunk
                if db < close_db:
                    silent_run += frame_dt
                    if silent_run >= trail_silence:
                        break
                else:
                    # Leaky, not a hard reset: a stray music spike loses some
                    # accumulated quiet instead of zeroing it, so end-of-turn
                    # still triggers when you've actually stopped over music.
                    silent_run = max(0.0, silent_run - frame_dt * config.GATE_LEAK)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
    return _wrap_wav(bytes(frames)) if started else b""


def _rms(pcm: bytes) -> float:
    import array
    a = array.array("h")
    a.frombytes(pcm)
    if not a:
        return 0.0
    return (sum(v * v for v in a) / len(a)) ** 0.5


def _dbfs(pcm: bytes) -> float:
    """Frame level in dBFS (0 = full scale), matching how OBS reads the gate."""
    rms = _rms(pcm)
    if rms < 1.0:
        return -120.0
    return 20.0 * math.log10(rms / 32768.0)


def _level(pcm: bytes) -> float:
    """Frame loudness as 0..1 for the taskbar voice bars: map -60 dBFS -> 0 (quiet)
    up to -10 dBFS -> 1 (loud speech), clamped."""
    return max(0.0, min(1.0, (_dbfs(pcm) + 60.0) / 50.0))


def _wrap_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(config.SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def transcribe(wav_bytes: bytes) -> str:
    """POST WAV to NulSpeech2Text, return the transcript (stdlib multipart)."""
    boundary = "----nulvoiceagent"
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\""
        f"\r\n\r\ntext\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode(),
        wav_bytes,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    req = urllib.request.Request(
        config.STT_URL, data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return _normalize(resp.read().decode("utf-8", "replace").strip())


# STT near-always hears "pane" as "pain" (and you never mean the literal word).
# Translate it at the transcription chokepoint so the route grammar, the chat
# brain, and the command classifier all see "pane".
_PAIN = re.compile(r"\bpains\b|\bpain\b", re.IGNORECASE)

# STT often mangles "Claude" into "clawed"/"claud"/"clode" and glues a trailing
# pane number on ("Claude2" = "Claude 2"). Normalize it at the same chokepoint so
# pane routing, the chat brain, and the command classifier all see "Claude".
# Enumerated + word-anchored so real words (cloud, clawed, …) are left untouched.
# Add your own app-name fixups the same way if your STT mishears a tool you route
# to a lot.
_CLAUDE = re.compile(r"\b(?:clawed|clawde|clawd|claud|clode|claude)(\d+)?\b", re.IGNORECASE)


def _claude_sub(m: re.Match) -> str:
    return "Claude" + (f" {m.group(1)}" if m.group(1) else "")


# Whisper sometimes prepends a list bullet / dash / ellipsis to an utterance
# ("- Name this pane …", "... name this pane …") — a transcription artifact, never
# spoken. It silently broke the start-anchored route + intent grammars (they fell
# through to chat, where Hermes ad-libbed "Named it" without registering). Strip
# any leading punctuation/bullet run here so every downstream consumer sees the
# real first word.
_LEAD_JUNK = re.compile(r"^[\s\-‐-―*•·.,:;!?]+")


def _normalize(text: str) -> str:
    text = _LEAD_JUNK.sub("", text or "")
    text = _CLAUDE.sub(_claude_sub, text)

    def repl(m: re.Match) -> str:
        word = m.group(0)
        out = "panes" if word.lower().endswith("s") else "pane"
        return out.capitalize() if word[:1].isupper() else out
    return _PAIN.sub(repl, text)


def listen(**kw) -> str:
    """Record from the mic and return the transcript ('' if nothing was said)."""
    wav = record(**kw)
    return transcribe(wav) if wav else ""
