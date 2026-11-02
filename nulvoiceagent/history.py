"""Short-term voice conversation history for the chat brain.

Every hotkey press is a fresh process, and many turns never reach the brain at
all: deterministic pane routes, intents, summaries and voice answers are handled
before chat. So when Trickery says "that message I just sent — forward it to 3D
model Claude", the brain had no idea what "that message" was and had to ask a
Claude pane to dig through the log.

This module keeps a rolling JSONL of recent turns (what he said, what happened)
and renders the last N of them as a [SYSTEM] hint for every chat turn. The brain
can quote a past utterance VERBATIM by id, `[[ask: <pane> | #<id>]]`, so that a
forwarded message is the exact original text, not the model's paraphrase.
"""
from __future__ import annotations

import json
import re
import time

from . import config

PATH = config.CACHE_DIR / "history.jsonl"
KEEP = 200            # lines kept on disk
HINT_TURNS = 20       # entries shown to the brain
HINT_MAX_AGE = 3 * 3600
ENTRY_CHARS = 1500    # per-entry cap in the hint (substitution uses the full text)
HINT_CHARS = 12000    # total cap for the hint body


def _read() -> list[dict]:
    try:
        lines = PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out


def record(kind: str, text: str) -> int | None:
    """Append one entry. kind: you | agent | route | ask | action | intent | answer.
    Returns the entry id (ids only grow). Never raises: history must never break
    a voice turn."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        rows = _read()
        nid = (rows[-1].get("id", 0) + 1) if rows else 1
        rows.append({"id": nid, "ts": time.time(), "kind": kind, "text": text})
        rows = rows[-KEEP:]
        tmp = PATH.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                       encoding="utf-8")
        tmp.replace(PATH)
        return nid
    except OSError:
        return None


def recent(n: int = HINT_TURNS, max_age: float = HINT_MAX_AGE) -> list[dict]:
    cutoff = time.time() - max_age
    return [r for r in _read() if r.get("ts", 0) >= cutoff][-n:]


def get(entry_id: int) -> dict | None:
    for r in reversed(_read()):
        if r.get("id") == entry_id:
            return r
    return None


_LABEL = {"you": "Trickery said", "agent": "you replied", "route": "router",
          "ask": "forwarded", "action": "action", "intent": "intent",
          "answer": "answer"}


def hint(exclude_last_you: bool = True) -> str:
    """[SYSTEM] block listing recent turns, oldest first. The current utterance
    is already recorded by the caller, so it's dropped to avoid duplication."""
    rows = recent()
    if exclude_last_you and rows and rows[-1].get("kind") == "you":
        rows = rows[:-1]
    if not rows:
        return ""
    lines, total = [], 0
    for r in reversed(rows):                       # newest first while budgeting
        t = r["text"]
        if len(t) > ENTRY_CHARS:
            t = t[:ENTRY_CHARS] + " …[truncated]"
        line = (f"#{r['id']} [{time.strftime('%H:%M:%S', time.localtime(r['ts']))}] "
                f"{_LABEL.get(r['kind'], r['kind'])}: {t}")
        if total + len(line) > HINT_CHARS:
            break
        lines.append(line)
        total += len(line)
    lines.reverse()
    return (
        "\n\n[SYSTEM] Recent voice history (oldest first; speech-to-text, so names may "
        "be mis-heard). Lines marked 'router' are what the deterministic pane router "
        "said, e.g. a message that failed to route. Use this when Trickery refers "
        "back to something (\"that message\", \"what I just said\", \"send it to …\", "
        "\"do that again\"). To forward an earlier message of his EXACTLY as he said "
        "it, write the request part of the ask directive as just its id, e.g. "
        "[[ask: <pane name> | #12]]. The original text is substituted for you, minus "
        "any leading \"tell <pane> to\". Don't paraphrase a message you're asked to "
        "forward. Never ask a pane to search logs for something listed here.\n"
        + "\n".join(lines)
    )


# A leading "can you tell <pane> uh to um" routing phrase, stripped when an
# utterance is forwarded by id so the pane receives only the request itself.
_ROUTE_PREFIX = re.compile(
    r"^(?:hey\s+|okay\s+|so\s+)*(?:can you\s+|could you\s+|would you\s+|please\s+)?"
    r"(?:tell|ask|send|have|route|message|paste)\s+.{1,60}?"
    r"(?:\s+(?:uh|um|er)\b)?[\s,]+(?:to|that)\s+(?:(?:uh|um|er)\b[\s,]*)*",
    re.IGNORECASE)


def resolve_ref(msg: str) -> str:
    """If an [[ask:]] request is a history reference ('#12'), return the verbatim
    original text (routing prefix removed); otherwise return msg unchanged."""
    m = re.fullmatch(r"\s*#\s*(\d+)\s*", msg or "")
    if not m:
        return msg
    r = get(int(m.group(1)))
    if not r:
        return msg
    text = r["text"]
    if r.get("kind") == "you":
        stripped = _ROUTE_PREFIX.sub("", text, count=1).strip()
        if stripped:
            text = stripped[0].upper() + stripped[1:]
    return text
