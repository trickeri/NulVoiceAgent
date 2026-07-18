"""Deterministic *parameterized* intents — spoken commands that carry ARGUMENTS
the allowlist classifier can't pass (it only picks a bare action name).

This is an EXTENSION POINT and ships empty. The command allowlist (actions.conf)
covers name-only commands ("lock the screen"); reach for an intent only when a
command needs a value pulled out of speech (a size, a filename, a number).

Matched before either brain (like dispatch.try_route for panes): pure regex, no
LLM cost. The safety guarantee is preserved because the command TEMPLATE is
hardcoded in your parser and only validated values are interpolated — there is no
free-form command for the model (or speech) to inject.

To add one: write a parser `(_re, builder)` where the builder returns an
Intent(say, argv) or None, and append it to _PARSERS. Example (commented) below.
"""
from __future__ import annotations

import re
import subprocess
from typing import Callable, Optional


class Intent:
    def __init__(self, say: str, argv: list[str]):
        self.say = say      # short spoken confirmation
        self.argv = argv    # exact argv to run (list form — never shell-joined)


# A parser inspects the utterance and returns an Intent or None.
Parser = Callable[[str], Optional[Intent]]


# --- example (disabled) ------------------------------------------------------
# Uncomment and adapt. This one would map "set the volume to 40 [percent]" to a
# concrete `wpctl` call, interpolating ONLY a validated integer 0..100.
#
# _RE_VOLUME = re.compile(r"\b(?:set |change )?volume (?:to |at )?(\d{1,3})\b", re.I)
#
# def _volume(text: str) -> Optional[Intent]:
#     m = _RE_VOLUME.search(text)
#     if not m:
#         return None
#     pct = max(0, min(100, int(m.group(1))))
#     return Intent(f"Volume {pct} percent.",
#                   ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{pct/100:.2f}"])


# --- OBS clip: "clip that" -> save the replay buffer + speak confirmation ----- #
# The MONEY intent: a missed "clip" = a lost highlight. Kept deterministic (no
# LLM) so it's instant and never gets talked-about-instead-of-done by the chat
# brain (which otherwise hallucinates "the player isn't running"). Grammar is
# forgiving of ASR jitter: the utterance must START with a clip trigger, and
# after it only throwaway filler may remain — so "clip that's the same" and
# "clip that real quick dude" fire, but "clip the audio in Kdenlive" (real
# content words left over) falls through to chat.
def _clip_norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).split())


_RE_CLIP_HEAD = re.compile(
    r"^(?:hey\s+)?(?:"
    r"clip"
    r"|(?:make|mark|take|save|grab|drop|log)\s+(?:a\s+|the\s+|that\s+|this\s+)?clip"
    r")\b",
    re.IGNORECASE)
# After the trigger, ONLY these throwaway words may remain. Any other word
# ("audio", "kdenlive", "in", "from") means it's chat, not a clip command.
_CLIP_FILLER = {
    "that", "thats", "this", "it", "here", "now", "the", "a", "s",
    "stream", "moment", "clip", "same", "one", "of", "up", "there", "right",
    "real", "quick", "fast", "quickly", "please", "too", "though",
    "dude", "man", "bro", "buddy", "okay", "ok",
}
# Splits an inline title off the command: "clip that TITLE the galaxy shader bug"
# / "clip that CALL IT ...", "... TITLED ...", "... NAME IT ...". Everything after
# the keyword (from the ORIGINAL text, so casing/spacing is preserved) becomes the
# per-clip title; the part before must still be a bare clip command.
_RE_CLIP_TITLE = re.compile(
    r"\b(?:title[ds]?|call(?:ed)?(?:\s+it|\s+this|\s+that)?|name[ds]?(?:\s+it|\s+this|\s+that)?)\b",
    re.IGNORECASE)


def try_clip(text: str) -> Optional[Intent]:
    raw = (text or "").strip()
    # Peel off an optional inline title before checking the clip grammar.
    title = ""
    body = raw
    tm = _RE_CLIP_TITLE.search(raw)
    if tm:
        body = raw[:tm.start()]
        title = raw[tm.end():].strip(" \t,.;:!?-\"'").strip()
    norm = _clip_norm(body)
    m = _RE_CLIP_HEAD.match(norm)
    if not m:
        return None
    # Everything after the trigger (minus the title) must be pure filler, else chat.
    if any(w not in _CLIP_FILLER for w in norm[m.end():].split()):
        return None
    # Unlike the other (pure) parsers, this one EXECUTES the marker tool inline so
    # the spoken confirmation carries the LIVE timecode + saved-clip note. Fast
    # local obs-websocket call; timeout covers the replay-save poll (~3.2s).
    argv = ["obs-clip-marker"]
    if title:
        argv += ["--title", title]
    try:
        out = subprocess.run(argv, capture_output=True, text=True,
                             timeout=9).stdout.strip()
    except Exception:  # noqa: BLE001 — never crash the mic flow over the marker tool
        out = ""
    # Already executed above; empty argv tells run() there's nothing left to do,
    # and __main__ still speaks intent.say (the marker's confirmation line).
    return Intent(out or "I couldn't mark the clip.", [])


# Registered parsers, tried in order.
_PARSERS: list[Parser] = [
    try_clip,
    # _volume,
]


def match(text: str) -> Optional[Intent]:
    """Return the first matching Intent for `text`, or None."""
    text = (text or "").strip()
    if not text:
        return None
    for parser in _PARSERS:
        try:
            intent = parser(text)
        except Exception:  # noqa: BLE001 — a bad parser must never break the mic flow
            intent = None
        if intent is not None:
            return intent
    return None


def run(intent: Intent) -> bool:
    """Run an Intent's argv (detached). Returns True if it launched."""
    if not intent.argv:
        return False
    try:
        subprocess.Popen(intent.argv)  # argv form: no shell, nothing to inject
        return True
    except OSError:
        return False
