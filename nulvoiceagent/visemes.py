"""Map kokoromodel's per-phoneme alignment to a timed viseme track for avatar
lip-sync. Only used when the TTS backend returns alignment (the PyTorch kokoro
engine, KK_ENGINE=torch); otherwise the avatar falls back to amplitude-only.

Reduced viseme set (a common lip-sync set the avatar has art for):
    rest  (lips together)      oo (rounded, small)     oh (rounded, open)
    ah    (wide open)          eh (spread / front)     teeth (slight open, tongue/teeth)

Phonemes come from misaki (Kokoro's g2p): mostly IPA, with a few ASCII stand-ins
for diphthongs/flaps (O=oʊ, I=aɪ, A=aʊ, T=ɾ flap, W=... ). Stress marks (ˈ ˌ) hold
the previous shape; spaces/punctuation are pauses (rest)."""
from __future__ import annotations

# single-phoneme -> viseme
_MAP = {
    # open
    "ɑ": "ah", "a": "ah", "ʌ": "ah", "æ": "ah", "ɐ": "ah", "A": "ah", "I": "ah",
    # rounded open
    "ɔ": "oh", "o": "oh", "O": "oh", "Q": "oh",
    # rounded small
    "u": "oo", "ʊ": "oo", "w": "oo", "W": "oo",
    # spread / front / neutral
    "ɛ": "eh", "e": "eh", "ɪ": "eh", "i": "eh", "y": "eh", "j": "eh",
    "ə": "eh", "ɚ": "eh", "ɜ": "eh", "ɝ": "eh",
    # lips together
    "m": "rest", "b": "rest", "p": "rest",
}
# consonants that read as a slight open / tongue-teeth
_TEETH = set("tdszʃʒʧʤθðnlɹrkgŋhfvT")
_PAUSE = set(".,!?;:—-…\"'()")


def viseme_for(ph: str):
    """Viseme name for one phoneme, or None to hold the previous shape."""
    ph = (ph or "").strip()
    if ph == "":
        return "rest"                 # inter-word space -> mouth closes
    if ph in ("ˈ", "ˌ", "ː"):
        return None                   # stress / length marks -> hold current shape
    if ph in _PAUSE:
        return "rest"
    if ph in _MAP:
        return _MAP[ph]
    if ph[0] in _TEETH:
        return "teeth"
    return "teeth"                    # safe neutral default (slight open)


def build_track(align: list[dict]) -> list[tuple[float, float, str]]:
    """Turn [{ph,start,end},...] into merged [(start, end, viseme), ...], resolving
    'hold' markers to the previous viseme and coalescing consecutive same visemes."""
    track: list[tuple[float, float, str]] = []
    prev = "rest"
    for a in align or []:
        v = viseme_for(a.get("ph", ""))
        if v is None:
            v = prev
        prev = v
        s, e = float(a.get("start", 0.0)), float(a.get("end", 0.0))
        if e <= s:
            continue
        if track and track[-1][2] == v and abs(track[-1][1] - s) < 0.02:
            track[-1] = (track[-1][0], e, v)      # extend the run
        else:
            track.append((s, e, v))
    return track


def viseme_at(track: list[tuple[float, float, str]], t: float) -> str:
    """The viseme active at playback time `t` (seconds); 'rest' outside any span."""
    if not track:
        return "rest"
    lo, hi = 0, len(track) - 1
    while lo <= hi:                    # binary search the sorted spans
        mid = (lo + hi) // 2
        s, e, v = track[mid]
        if t < s:
            hi = mid - 1
        elif t >= e:
            lo = mid + 1
        else:
            return v
    return "rest"
