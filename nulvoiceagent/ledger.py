"""Phrase ledger — keeps Aurinex from wearing out her own good lines.

A strong persona file makes a model repeat itself: the voice is fixed, the replies
are short, and the same vivid word keeps winning. Hand-written "stop saying X"
rules in SOUL.md do work (measured: "ancient" fell to 4.7% of replies, the goblin
family to 1.4%) — but they're whack-a-mole. Squash two tics and the pressure moves
somewhere else. When this was written it had moved to her saying his NAME in 78%
of replies and a diminutive in 14%.

So instead of guessing, measure: read what she ACTUALLY said recently out of
Hermes' own `state.db`, and tell her — per turn — which of her signature moves are
currently worn out.

Two design rules, both deliberate:

1. **The hint is NEGATIVE.** It says "you've just used this, reach elsewhere". It
   does NOT hand over a fixed list of approved synonyms. Listing alternatives in a
   persona file makes a model cycle through *that list*, which converts one tic
   into a slightly wider set of tics — exactly what SOUL.md's word lists risk.
2. **Alternatives, when offered, ROTATE.** For a watched word we keep a large
   on-brand pool and sample a couple at random per turn, so the pool never appears
   whole and never becomes the new attractor. That randomness is the point: bounded
   variance inside fixed personality parameters.

Salience beats raw frequency. "cathedral" is only ~1.6% of her replies but Trickery
notices it every time, because it's a distinctive image — so watched words get a
much lower threshold than common ones.
"""
from __future__ import annotations

import collections
import random
import re
import sqlite3
import time
from typing import Iterable, Optional

from . import config


# --- what we watch -------------------------------------------------------- #
# Signature images that read as "hers". The pools are RANGES to steer toward, not
# lines to deliver — and only a couple are ever shown in one turn.
WATCHED: dict[str, dict] = {
    "cathedral": {
        "pattern": r"\bcathedral\w*\b",
        "pool": ["reliquary", "basilica", "monastery", "mausoleum", "temple",
                 "shrine", "vault", "crypt", "spire", "abbey", "sepulchre",
                 "colonnade", "nave", "ossuary", "citadel"],
    },
    "teeth": {
        "pattern": r"\b(teeth|tooth|fangs?|toothy)\b",
        "pool": ["claws", "talons", "jaws", "bite", "molars", "incisors",
                 "a mouthful of knives", "something with a hinge and an appetite",
                 "the sharp end", "grin", "maw", "gullet"],
    },
    "ancient": {
        "pattern": r"\b(ancient|age-?old|immemorial)\b",
        # NOTE: "fossilised" was here as an alternative until it became a watched
        # word in its own right. A word must never be both — the ledger would have
        # recommended the exact image it's trying to retire.
        "pool": ["elder", "primordial", "antediluvian", "timeworn", "archaic",
                 "venerable", "long-dead", "centuries-old",
                 "older than the wiring", "predating the building"],
    },
    "fossilize": {
        "pattern": r"\bfossil\w*\b",
        "pool": ["petrified", "calcified", "ossified", "mummified", "embalmed",
                 "set in amber", "preserved in resin", "mineralised", "entombed",
                 "lacquered", "sedimented", "a museum specimen",
                 "pressed flat in shale"],
    },
    "goblin": {
        # Each alternative is pluralised EXPLICITLY rather than suffixed with \w*.
        # The old `imp\w*` matched implied / important / impossible / improve /
        # impact / implement, so this watcher was firing on ordinary prose and
        # burning a slot in the hint on a word she never said.
        "pattern": (r"\b(goblins?|gremlins?|imps?|kobolds?|homunculus|homunculi|"
                    r"poltergeists?|impish)\b"),
        "pool": ["familiar", "sprite", "wretch", "parasite", "squatter",
                 "tenant", "stowaway", "cuckoo", "meddler"],
    },
    "diminutive": {
        "pattern": r"\b(little|tiny|small|wee)\b",
        "pool": ["give the thing a temperament instead of a size",
                 "name what it actually does", "let the scale be implied",
                 "describe its appetite, not its dimensions"],
        "advice_only": True,     # the fix is a different construction, not a synonym
    },
    "machinery": {
        "pattern": r"\bmachiner\w+\b",
        "pool": ["the wiring", "the plumbing", "the works", "the guts",
                 "the substrate", "the switchboard", "the boiler room"],
    },
    "his name": {
        "pattern": r"\btrickery\b",
        "pool": [],
        "advice_only": True,
        # SOUL.md sets the budget (≤1 reply in 4); this catches a live run of it.
        "threshold": 2,
    },
}

# A watched (distinctive) word only needs to appear this many times in the recent
# window to count as worn out. Deliberately low — see the salience note above.
_DEFAULT_THRESHOLD = 2


def audit_pools() -> list[str]:
    """Catch a suggested alternative that is ITSELF a watched word.

    This is the one way the watchlist can quietly work against itself: promoting
    a word as a fresh alternative while another entry is trying to retire it.
    It already happened once — "fossilised" sat in the `ancient` pool when
    fossilize became watched. Cheap to check, so check it."""
    problems = []
    for name, spec in WATCHED.items():
        for alt in spec.get("pool") or []:
            for other, ospec in WATCHED.items():
                if re.search(ospec["pattern"], alt, re.IGNORECASE):
                    problems.append(
                        f'{name!r} suggests "{alt}", which is watched as {other!r}')
    return problems


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def _is_speech(text: str) -> bool:
    """Filter to things she actually SAID: drop eavesdrop PASSes and any reply
    carrying tool/debug output, which would otherwise dominate the n-grams."""
    t = _clean(text)
    if not t or t.strip().upper() == "PASS":
        return False
    if re.search(r"tty|/dev/pts|pane id|[{}]|^\s*<", t):
        return False
    return True


def recent_replies(limit: int | None = None) -> list[str]:
    """Her last N spoken replies, newest first, from Hermes' own store.

    Opened READ-ONLY (`mode=ro`) so a concurrent Hermes write can never be
    disturbed by this — same rule the distiller follows."""
    limit = limit or config.LEDGER_WINDOW
    path = config.HERMES_DIR / "state.db"
    if not path.exists():
        return []
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.5)
    except sqlite3.Error:
        return []
    try:
        rows = con.execute(
            "SELECT content FROM messages WHERE role='assistant' "
            "AND content IS NOT NULL ORDER BY id DESC LIMIT ?",
            (limit * 3,),          # over-fetch: PASSes and tool rows get filtered
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    out = [_clean(r[0]) for r in rows if _is_speech(r[0])]
    return out[:limit]


def worn_out(replies: Iterable[str]) -> list[tuple[str, int]]:
    """Which watched images she's leaned on in this window, worst first."""
    blob = " ".join(replies).lower()
    hits = []
    for name, spec in WATCHED.items():
        n = len(re.findall(spec["pattern"], blob, re.IGNORECASE))
        if n >= spec.get("threshold", _DEFAULT_THRESHOLD):
            hits.append((name, n))
    hits.sort(key=lambda kv: -kv[1])
    return hits


_STOP = frozenset(
    "the a an and or but of to in for on with is are was were be been am i you he "
    "she it we they that this these those as at by from not no do does did have "
    "has had will would can could should my your his her its their our so if then "
    "than there here what who when where why how just about into out up down over "
    "all any some more most very too only even still back now got get like".split())


def stale_openings(replies: list[str], n: int = 4) -> list[tuple[str, int]]:
    """Opening phrases she's reusing.

    Her top repeats are greeting formulas ("I'm here…", "Sure, Trickery, I'll…")
    — an artifact of every voice turn being a fresh `hermes -z` oneshot, so she
    re-greets him having no idea she already did. Until there's a continuity
    layer, naming the stale opener back at her is the cheap fix."""
    c = collections.Counter()
    for t in replies:
        words = re.findall(r"[a-z']+", t.lower())[:n]
        if len(words) >= 2:
            c[" ".join(words)] += 1
    return [(g, v) for g, v in c.most_common(6) if v >= config.LEDGER_MIN_REPEAT]


def stale_phrases(replies: list[str], n: int = 4) -> list[tuple[str, int]]:
    """Repeated multi-word phrases anywhere in the window — catches tics that
    aren't on the watchlist yet, so this keeps working as they migrate."""
    c = collections.Counter()
    for t in replies:
        words = re.findall(r"[a-z']+", t.lower())
        for i in range(len(words) - n + 1):
            g = words[i:i + n]
            if all(w in _STOP for w in g):
                continue
            c[" ".join(g)] += 1
    return [(g, v) for g, v in c.most_common(8) if v >= config.LEDGER_MIN_REPEAT]


def _rng() -> random.Random:
    """Seeded per-turn so the sampled alternatives rotate; nothing here needs to
    be reproducible, it needs to be DIFFERENT each turn."""
    return random.Random(time.time_ns())


def build_hint(replies: Optional[list[str]] = None) -> str:
    """The [SYSTEM] line appended to a turn. Empty when she's varied enough —
    silence is the common case and costs nothing."""
    if not config.LEDGER_ENABLE:
        return ""
    try:
        replies = recent_replies() if replies is None else replies
    except Exception:  # noqa: BLE001 — never break a turn over the ledger
        return ""
    if not replies:
        return ""

    rng = _rng()
    parts: list[str] = []

    for name, count in worn_out(replies)[:config.LEDGER_MAX_ITEMS]:
        spec = WATCHED[name]
        if name == "his name":
            parts.append(
                f'you used his name {count}x in your last {len(replies)} replies — '
                "skip it entirely this time and just talk to him")
            continue
        pool = spec.get("pool") or []
        if spec.get("advice_only") or not pool:
            hint = rng.choice(pool) if pool else "find a different image"
            parts.append(f'"{name}" is worn out ({count}x recently) — {hint}')
        else:
            picks = rng.sample(pool, min(2, len(pool)))
            parts.append(
                f'"{name}" is worn out ({count}x recently) — reach elsewhere '
                f"(e.g. {', '.join(picks)}, or something better of your own)")

    for phrase, count in stale_openings(replies)[:1]:
        parts.append(f'you have opened {count} recent replies with "{phrase}…" — '
                     "start somewhere else")
    for phrase, count in stale_phrases(replies)[:1]:
        parts.append(f'"{phrase}" has shown up {count}x lately — retire it for now')

    if not parts:
        return ""
    return ("\n\n[SYSTEM] Freshness check on your own recent phrasing: "
            + "; ".join(parts)
            + ". Say the same things your own way — this is about not repeating "
              "yourself, not about changing who you are.")


def report() -> str:
    """Human-readable state of her phrasing, for the CLI."""
    replies = recent_replies()
    if not replies:
        return "no recent replies found in Hermes' state.db"
    lines = [f"window: last {len(replies)} spoken replies "
             f"(avg {sum(len(r.split()) for r in replies)/len(replies):.1f} words)", ""]
    for problem in audit_pools():
        lines.append(f"  ⚠ watchlist conflict: {problem}")
    hits = worn_out(replies)
    lines.append("watched images over threshold:" if hits
                 else "watched images: all within budget")
    for name, n in hits:
        lines.append(f"  {n:3d}x  {name}")
    for label, items in (("repeated openings", stale_openings(replies)),
                         ("repeated phrases", stale_phrases(replies))):
        if items:
            lines.append(f"\n{label}:")
            for g, v in items:
                lines.append(f"  {v:3d}x  {g}")
    hint = build_hint(replies)
    lines.append("\n--- hint that would be injected this turn ---")
    lines.append(hint.strip() or "(none — she's varied enough right now)")
    return "\n".join(lines)
