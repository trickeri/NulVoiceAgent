"""Text-to-speech out — POST to kokoromodel, play the WAV through PipeWire."""
from __future__ import annotations
import array
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import wave

from . import config, state

# Seconds the last speak() spent on synth (kokoromodel POST + optional harmony
# layer), i.e. the latency before any sound — NOT the playback duration. The CLI
# reads this to log a per-turn `timing:` line. Reset on every speak().
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
        wav = resp.read()
    if config.LAYER_PITCH:
        try:
            wav = _layer(wav, float(config.LAYER_PITCH), float(config.LAYER_GAIN or "0.7"))
        except (ValueError, OSError):
            pass
    return wav


def _layer(wav: bytes, semitones: float, gain: float) -> bytes:
    """Mix in a copy pitched `semitones` (e.g. -7 = a perfect fifth below) with a
    complementary EQ split: the lead is highpassed (config.LAYER_LEAD_HPF) so the
    pitched copy owns the lows; the copy is lowpassed (config.LAYER_DOWN_LPF) so the
    lead owns the highs. Pitch via rubberband; filter + mix via ffmpeg (falls back
    to an unfiltered python mix if ffmpeg is unavailable, and to the dry lead if
    rubberband is unavailable)."""
    rb = shutil.which("rubberband")
    if not rb or not semitones:
        return wav
    ff = shutil.which("ffmpeg")
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "base.wav")
        low = os.path.join(d, "low.wav")
        with open(base, "wb") as f:
            f.write(wav)
        subprocess.run([rb, "-p", str(semitones), "-q", base, low], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if ff:
            out = os.path.join(d, "out.wav")
            lead_f = f"highpass=f={config.LAYER_LEAD_HPF}" if config.LAYER_LEAD_HPF else "anull"
            down_pre = f"lowpass=f={config.LAYER_DOWN_LPF}," if config.LAYER_DOWN_LPF else ""
            fc = (f"[0:a]{lead_f}[lead];"
                  f"[1:a]{down_pre}volume={gain}[low];"
                  f"[lead][low]amix=inputs=2:normalize=0[m]")
            try:
                subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y",
                                "-i", base, "-i", low, "-filter_complex", fc,
                                "-map", "[m]", "-ar", "24000", "-ac", "1", out],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                with open(out, "rb") as f:
                    return f.read()
            except (subprocess.CalledProcessError, OSError):
                pass  # fall back to the unfiltered python mix below

        with wave.open(io.BytesIO(wav), "rb") as w:
            params = w.getparams()
            b = array.array("h"); b.frombytes(w.readframes(w.getnframes()))
        with wave.open(low, "rb") as w:
            lo = array.array("h"); lo.frombytes(w.readframes(w.getnframes()))

    n = min(len(b), len(lo))
    mixed = array.array("h", bytes(2 * n))
    for i in range(n):
        s = int(b[i] + gain * lo[i])
        mixed[i] = -32768 if s < -32768 else 32767 if s > 32767 else s
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(params.nchannels)
        w.setsampwidth(params.sampwidth)
        w.setframerate(params.framerate)
        w.writeframes(mixed.tobytes())
    return buf.getvalue()


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
