"""Ghost chain — an offline stereo FX chain for a TTS line (profile GHOST > 0).

Built for Slyn (a Shade Shifter): a dark voice whose words swell in out of their
own reversed echo. Kokoro hands us the whole sentence before playback, which is
what makes reverse reverb possible at all (it needs the future to build the swell).

    upper  lead -> reverse reverb (reverse, convolve a noise tail, reverse back)
           + a short ordinary room; separate random tails per side = stereo
           -> 48 kHz, highpass LAYER_LEAD_HPF, EQ_BANDS peaks, high shelf SHELF_DB @ SHELF_HZ,
           aexciter (EXCITE @ EXCITE_HZ: harmonics up to 20 kHz; Kokoro stops at 12)
    lower  lead pitched LAYER_PITCH (rubberband), lowpass LAYER_DOWN_LPF,
           x LAYER_GAIN, dry
    master LR4 crossover at MONO_BELOW: lows summed to mono, highs stereo;
           LUFS set: gentle compression -> gain to LUFS -> light limiter (-1 dBFS);
           else peak = dry lead peak + OUT_GAIN_DB (capped below clipping)

process() returns (wav, leadin): the GHOST_LEADIN seconds of silence added up front
(room for the first word's swell) shift every timing in the line by that much.
"""
from __future__ import annotations
import io
import os
import re
import subprocess
import tempfile
import wave

import numpy as np

from . import config

_SR = 48000


def _read(p: str) -> tuple[int, np.ndarray]:
    with wave.open(p) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
    return sr, a.reshape(-1, ch)


def _write(p, sr: int, a: np.ndarray) -> None:
    a = np.clip(a, -1.0, 1.0)
    with wave.open(p, "wb") as w:
        w.setnchannels(a.shape[1]); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((a * 32767).astype(np.int16).tobytes())


def _tail(rng, sr: int, seconds: float, pre: float = 0.0) -> np.ndarray:
    """Exponentially decaying noise impulse response (-60 dB at `seconds`)."""
    n = max(1, int(sr * seconds))
    h = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / sr / seconds)
    h[: int(sr * pre)] = 0
    return h / np.sqrt((h ** 2).sum())


def _conv(a: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = len(a) + len(h) - 1
    f = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, f) * np.fft.rfft(h, f), f)[: len(a)]


def _lowpass(a: np.ndarray, sr: int, hz: float) -> np.ndarray:
    """Zero-phase 24 dB/oct lowpass (Butterworth magnitude, applied in the FFT)."""
    if hz <= 0:
        return a
    n = 1 << (len(a) - 1).bit_length()
    f = np.fft.rfftfreq(n, 1 / sr)
    return np.fft.irfft(np.fft.rfft(a, n) / np.sqrt(1 + (f / hz) ** 8), n)[: len(a)]


def _norm(a: np.ndarray) -> np.ndarray:
    m = np.abs(a).max()
    return a / m if m > 0 else a


def _ff(*args: str) -> str:
    return subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *args], check=True,
                          capture_output=True, text=True).stderr


def _loudness(path: str) -> float:
    """Integrated loudness (LUFS) of a file, via ffmpeg's ebur128 summary."""
    err = _ff("-i", path, "-af", "ebur128", "-f", "null", "-")
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", err)
    return float(m[-1]) if m else -70.0


def _to_lufs(src: str, dst: str) -> None:
    """Loud without pumping: a gentle compressor evens the line out (slow enough
    to ride phrases, not syllables), gain lands it on config.LUFS, and a limiter
    with lookahead only catches the odd stray peak."""
    comp = os.path.join(os.path.dirname(dst), "comp.wav")
    _ff("-y", "-i", src, "-af",
        "acompressor=threshold=-24dB:ratio=2.5:attack=20:release=250:knee=4", comp)
    gain = config.LUFS - _loudness(comp)
    _ff("-y", "-i", comp, "-af",
        f"volume={gain:.2f}dB,alimiter=limit=0.891:attack=5:release=120:level=0", dst)


def process(wav: bytes) -> tuple[bytes, float]:
    with wave.open(io.BytesIO(wav)) as w:
        sr = w.getframerate()
        lead = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
    peak = np.abs(lead).max()
    if peak <= 0:
        return wav, 0.0
    lead_in = int(sr * config.GHOST_LEADIN)
    tail = int(sr * 0.9)                                  # room for the room reverb to ring out
    dry = np.concatenate([np.zeros(lead_in), lead, np.zeros(tail)])

    rng = np.random.default_rng(7)                        # same ghost every time for the same line
    upper = np.empty((len(dry), 2))
    for c in range(2):
        rev = _lowpass(_conv(dry[::-1], _tail(rng, sr, config.GHOST_DECAY))[::-1], sr,
                       config.GHOST_LPF)
        room = _conv(dry, _tail(rng, sr, 0.9, 0.012))
        upper[:, c] = config.GHOST_DRY * dry + config.GHOST * _norm(rev) * peak + config.GHOST_ROOM * _norm(room) * peak
    upper *= 10 ** (config.UPPER_GAIN_DB / 20)
    fade = min(len(dry), int(sr * 0.12))                 # the first swell is cut off by the lead-in
    upper[:fade] *= np.linspace(0, 1, fade)[:, None]

    with tempfile.TemporaryDirectory() as d:
        lead_p, low_p = os.path.join(d, "lead.wav"), os.path.join(d, "low.wav")
        up_p, mix_p = os.path.join(d, "up.wav"), os.path.join(d, "mix.wav")
        _write(lead_p, sr, dry[:, None])
        _write(up_p, sr, upper * 0.5)                     # headroom for the sum
        subprocess.run(["rubberband", "-p", str(config.LAYER_PITCH or 0), "-q", lead_p, low_p],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        hp = f"highpass=f={config.LAYER_LEAD_HPF}," if config.LAYER_LEAD_HPF else ""
        lp = f"lowpass=f={config.LAYER_DOWN_LPF}," if config.LAYER_DOWN_LPF else ""
        eq = ""
        for band in filter(None, (b.strip() for b in config.EQ_BANDS.split(","))):
            hz, db = (float(v) for v in band.split(":"))
            eq += f"equalizer=f={hz:g}:t=o:w=1:g={db:g},"
        shelf = f"highshelf=f={config.SHELF_HZ:g}:g={config.SHELF_DB:g}," if config.SHELF_DB else ""
        excite = (f"aexciter=amount={config.EXCITE:g}:drive=6:freq={config.EXCITE_HZ:g}:ceil=20000,"
                  if config.EXCITE > 0 else "")
        fc = (f"[0:a]aresample={_SR},{hp}{eq}{shelf}{excite}anull[u];"
              f"[1:a]aresample={_SR},{lp}volume={float(config.LAYER_GAIN or 0.7) * 0.5:g},"
              f"pan=stereo|c0=c0|c1=c0[l];"
              f"[u][l]amix=inputs=2:normalize=0")
        if config.MONO_BELOW > 0:
            f0 = f"{config.MONO_BELOW:g}"
            fc += (f",asplit[a][b];"
                   f"[a]lowpass=f={f0},lowpass=f={f0},pan=mono|c0=0.5*c0+0.5*c1,"
                   f"pan=stereo|c0=c0|c1=c0[lo];"
                   f"[b]highpass=f={f0},highpass=f={f0}[hi];"
                   f"[lo][hi]amix=inputs=2:normalize=0")
        fc += "[m]"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", up_p, "-i", low_p, "-filter_complex", fc,
                        "-map", "[m]", "-ar", str(_SR), mix_p], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if config.LUFS < 0:
            loud_p = os.path.join(d, "loud.wav")
            _to_lufs(mix_p, loud_p)
            osr, mix = _read(loud_p)
        else:
            osr, mix = _read(mix_p)
            mix *= min(peak * 10 ** (config.OUT_GAIN_DB / 20), 0.95) / max(np.abs(mix).max(), 1e-9)
    buf = io.BytesIO()
    _write(buf, osr, mix)
    return buf.getvalue(), lead_in / sr
