"""Text-to-speech out — POST to kokoromodel, play the WAV through PipeWire."""
from __future__ import annotations
import array
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
import wave

from . import config, state, visemes

# Seconds the last speak() spent on synth (kokoromodel POST + optional harmony
# layer), i.e. the latency before any sound — NOT the playback duration. The CLI
# reads this to log a per-turn `timing:` line. Reset on every speak().
last_synth_secs = 0.0

# Per-phoneme alignment from the last synth (torch backend only): list of
# {ph,start,end}. Empty when the backend returns no alignment (onnx). Read by speak()
# to build a viseme track for lip-sync.
_last_align: list[dict] = []


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
    global _last_align
    _last_align = []
    with urllib.request.urlopen(req, timeout=60) as resp:
        wav = resp.read()
        hdr = resp.headers.get("X-Kokoro-Align")   # per-phoneme timing (torch backend)
    if hdr:
        try:
            import base64
            _last_align = json.loads(base64.b64decode(hdr))
        except (ValueError, TypeError):
            _last_align = []
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


def _mouth_envelope(wav: bytes, hop: float = 0.025) -> tuple[list[float], float]:
    """Per-window mouth-openness 0..1 from a WAV's short-time loudness, so a reader
    can drive real amplitude lip-sync (mouth tracks the syllables actually spoken).
    Silence -> ~0, loud syllables -> ~1; normalized to the clip's own loud level and
    lightly smoothed (quick open, softer close) so it tracks speech without chattering.
    Returns (levels, hop_secs); empty list on parse failure so callers fall back."""
    try:
        with wave.open(io.BytesIO(wav), "rb") as w:
            ch, sw, fr = w.getnchannels(), w.getsampwidth(), (w.getframerate() or 24000)
            raw = w.readframes(w.getnframes())
    except (wave.Error, OSError, ValueError):
        return [], hop
    if sw != 2:                                   # only 16-bit PCM
        return [], hop
    a = array.array("h"); a.frombytes(raw)
    if ch > 1:                                     # downmix to mono
        a = a[0::ch]
    win = max(1, int(fr * hop))
    lv: list[float] = []
    for i in range(0, len(a), win):
        seg = a[i:i + win]
        if not seg:
            break
        rms = (sum(v * v for v in seg) / len(seg)) ** 0.5
        db = -120.0 if rms < 1.0 else 20.0 * math.log10(rms / 32768.0)
        lv.append(max(0.0, min(1.0, (db + 50.0) / 38.0)))   # -50 dBFS->0 .. -12 dBFS->1
    if not lv:
        return [], hop
    srt = sorted(lv); p95 = srt[int(0.95 * (len(srt) - 1))]  # normalize to loud level
    if p95 > 0.05:
        lv = [min(1.0, x / p95) for x in lv]
    lv = [x ** 1.3 for x in lv]                     # gamma: deepen dips -> more articulation
    out: list[float] = []; m = 0.0                 # asymmetric smoothing
    for x in lv:
        m += (0.6 if x > m else 0.32) * (x - m)
        out.append(round(m, 3))
    return out, hop


def _drive_mouth(env: list[float], hop: float, dur: float, until: float,
                 stop: threading.Event, track=None) -> None:
    """Publish the mouth envelope (amplitude) — and, when a viseme `track` is given,
    the current viseme shape — into the state file in sync with playback (~33 Hz)
    until the audio ends or `stop` is set. Runs in a thread alongside play()."""
    if not env:
        return
    start = time.monotonic()
    while not stop.is_set():
        t = time.monotonic() - start
        if t > dur + 0.05:
            break
        i = int(t / hop)
        m = env[i] if i < len(env) else 0.0
        vis = visemes.viseme_at(track, t) if track else None
        state.write("talking", until=until, mouth=m, viseme=vis)
        stop.wait(0.03)


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
    # avatar/status reader can animate while we speak, then drop back to idle. play()
    # blocks until the audio finishes, so this brackets the whole reply. A background
    # thread streams the real syllable amplitude as `mouth` 0..1 for live lip-sync.
    dur = state.wav_duration(wav)
    until = time.time() + max(0.0, dur) + 0.5
    env, hop = _mouth_envelope(wav)
    track = visemes.build_track(_last_align) if _last_align else None
    state.write("talking", until=until, mouth=(env[0] if env else None),
                viseme=(visemes.viseme_at(track, 0.0) if track else None))
    stop = threading.Event()
    driver = None
    if env:
        driver = threading.Thread(target=_drive_mouth,
                                  args=(env, hop, dur, until, stop, track), daemon=True)
        driver.start()
    try:
        play(wav)
    finally:
        stop.set()
        if driver is not None:
            driver.join(timeout=0.3)
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
