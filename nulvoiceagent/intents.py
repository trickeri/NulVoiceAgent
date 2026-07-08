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


# Registered parsers, tried in order. Ships empty — add your own above.
_PARSERS: list[Parser] = [
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
