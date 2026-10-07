"""Publish what the agent is saying, word-timed, to a tiny JSON file so an OPTIONAL
caption reader (e.g. the desktop speech-bubble overlay) can print it along with
the voice. Like state.json, nothing in the framework needs a reader.

caption.json, written once per speak() just before playback starts:
    {"id": "<pid>-<ms>", "start": <epoch audio starts>, "until": <epoch it ends>,
     "sentences": [{"s": sec, "e": sec,
                    "words": [{"t": "there,", "ws": " ", "s": sec, "e": sec}, ...]}]}
Times are seconds from `start`. Joining t+ws over a sentence's words gives its
text. Punctuation rides on the word before it. When playback ends it's rewritten
as {"id": ..., "active": false} (only if it's still ours, so an overlapping
speaker isn't blanked).

Word times come from kokoromodel's X-Kokoro-Words header (torch engine). Without
them (onnx engine) the words are spread over the audio by character count, which
is close enough for a caption."""
from __future__ import annotations
import json
import os
import re
import tempfile
import time

from . import config

CAPTION_FILE = config.CACHE_DIR / "caption.json"

# A sentence ends on a word ending in . ! ? … (optionally followed by quotes/brackets).
_END = re.compile(r"[.!?…][\"')\]]*$")
# Past this many words a sentence is split (preferring a comma) so the bubble
# stays one readable line-group.
MAX_WORDS = 22


def _merge_punct(tokens: list[dict]) -> list[dict]:
    """Fold untimed / pure-punctuation tokens into the preceding word."""
    out: list[dict] = []
    for tk in tokens:
        t = str(tk.get("t", ""))
        s, e = tk.get("s"), tk.get("e")
        punct = not any(c.isalnum() for c in t)
        if out and (punct or s is None) and not out[-1]["ws"]:
            out[-1]["t"] += t
            out[-1]["ws"] = tk.get("ws", "") or ""
            if e is not None:
                out[-1]["e"] = max(out[-1]["e"], float(e))
            continue
        if not t.strip():
            continue
        s = float(s) if s is not None else (out[-1]["e"] if out else 0.0)
        e = float(e) if e is not None else s
        out.append({"t": t, "ws": tk.get("ws", "") or "", "s": s, "e": max(s, e)})
    return out


def _even_words(text: str, dur: float) -> list[dict]:
    """No timings: spread whitespace-split words over `dur` by character count."""
    parts = text.split()
    if not parts:
        return []
    lead, tail = min(0.25, dur * 0.05), min(0.3, dur * 0.05)
    span = max(0.1, dur - lead - tail)
    total = sum(len(p) + 1 for p in parts)
    out, t = [], lead
    for p in parts:
        d = span * (len(p) + 1) / total
        out.append({"t": p, "ws": " ", "s": round(t, 3), "e": round(t + d, 3)})
        t += d
    return out


def _sentences(words: list[dict]) -> list[dict]:
    groups: list[list[dict]] = [[]]
    for w in words:
        groups[-1].append(w)
        if _END.search(w["t"]):
            groups.append([])
    out: list[list[dict]] = []
    for g in groups:
        while len(g) > MAX_WORDS:              # split an over-long sentence
            cut = max((i for i in range(MAX_WORDS // 2, MAX_WORDS)
                       if g[i - 1]["t"].endswith((",", ";", ":", "—"))), default=MAX_WORDS)
            out.append(g[:cut]); g = g[cut:]
        if g:
            out.append(g)
    return [{"s": g[0]["s"], "e": g[-1]["e"], "words": g} for g in out]


def build(text: str, words: list[dict] | None, dur: float) -> list[dict]:
    ws = _merge_punct(words) if words else []
    if not ws:
        ws = _even_words(text, dur)
    for w in ws:
        w["ws"] = " " if w["ws"] else ""
    return _sentences(ws)


def _write(obj: dict) -> None:
    try:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(config.CACHE_DIR), prefix=".caption.")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(tmp, CAPTION_FILE)
    except OSError:
        pass


def show(text: str, words: list[dict] | None, dur: float) -> str | None:
    """Publish the caption for audio starting now; returns its id (for clear())."""
    if not config.CAPTION:
        return None
    sentences = build(text, words, dur)
    if not sentences:
        return None
    start = time.time() + config.CAPTION_DELAY
    cid = f"{os.getpid()}-{int(start * 1000)}"
    _write({"id": cid, "start": start, "until": start + dur + 0.3,
            "sentences": sentences})
    return cid


def clear(cid: str | None) -> None:
    if not cid:
        return
    try:
        cur = json.loads(CAPTION_FILE.read_text(encoding="utf-8"))
        if cur.get("id") != cid:
            return                              # someone else is speaking now
    except (OSError, ValueError):
        pass
    _write({"id": cid, "active": False})
