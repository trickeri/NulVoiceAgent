"""Route mode — speak to a *named* terminal pane and have your words land there.

You focus a tmux pane, name it by voice ("name this Krita Claude"), then later
say "tell Krita Claude to add a double jump" and the payload is pasted into that
pane and submitted. Pure tmux: a pane's `%id` is a stable handle addressable with
no window focus at all, so this works even if the pane is on another desktop or
the window is unfocused.

Targets are DATA: `~/.config/nulvoiceagent/targets.json`
    { "<spoken name>": {"kind": "tmux", "pane": "%3", "added": <epoch>} }

Parsing is deterministic (no LLM round-trip): the grammar below maps a transcript
to one of three intents — name / dispatch / manage — and fuzzy-matches the spoken
target against the registry keys. Names are arbitrary spoken phrases, so both
sides are normalized (lowercase, alnum-only, single-spaced) before matching.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from difflib import SequenceMatcher
from typing import Optional

from . import config

# --------------------------------------------------------------------------- #
# result type — a parsed intent the CLI layer executes + speaks                #
# --------------------------------------------------------------------------- #
class Result:
    def __init__(self, ok: bool, say: str):
        self.ok = ok
        self.say = say


# --------------------------------------------------------------------------- #
# registry                                                                     #
# --------------------------------------------------------------------------- #
def _load_file() -> dict:
    try:
        data = json.loads(config.TARGETS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load() -> dict:
    """The registry, reconciled against tmux. The source of truth is the live
    per-pane `@va_name` option (set by register()): we rebuild name->pane from the
    panes that actually exist right now, so names for panes that were closed or
    that died on a REBOOT/server-restart simply vanish (their %id is gone) — the
    on-disk targets.json can never carry stale rows across a reboot again. Only if
    there's no tmux server do we fall back to the on-disk file."""
    cp = _tmux("list-panes", "-a", "-F", "#{pane_id}\t#{@va_name}")
    if cp.returncode != 0:
        return _load_file()          # no server running — nothing live to sync to
    old = _load_file()
    targets: dict = {}
    for line in cp.stdout.splitlines():
        pane, _, name = line.partition("\t")
        name = name.strip()
        if not name:
            continue
        added = old.get(name, {}).get("added") if isinstance(old.get(name), dict) else None
        targets[name] = {"kind": "tmux", "pane": pane, "added": added or int(time.time())}
    if targets != old:                # persist the cleaned-up view
        save(targets)
    return targets


def save(targets: dict) -> None:
    config.TARGETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.TARGETS_FILE.write_text(json.dumps(targets, indent=2), encoding="utf-8")


def register(name: str, pane: str) -> None:
    targets = load()
    targets[name] = {"kind": "tmux", "pane": pane, "added": int(time.time())}
    save(targets)
    # Persist the label in a per-pane USER OPTION (@va_name) that the border
    # reads — programs (Claude Code, the zsh prompt) constantly overwrite
    # pane_title via OSC escape sequences, but nothing touches @va_name, so the
    # name sticks. (-T too as a best-effort fallback for tools reading the title.)
    _tmux("set", "-p", "-t", pane, "@va_name", name)
    _tmux("select-pane", "-t", pane, "-T", name)


def forget(name: str) -> bool:
    targets = load()
    key = _match_key(name, targets)
    if key is None:
        return False
    pane = targets[key].get("pane")
    del targets[key]
    save(targets)
    if pane:
        _tmux("set", "-pu", "-t", pane, "@va_name")   # clear the label option
    return True


def reset() -> int:
    """Wipe ALL pane names: empty the registry and clear every live pane's label
    option + title (so borders fall back to "pane N"). Returns how many were cleared."""
    n = len(load())
    save({})
    cp = _tmux("list-panes", "-a", "-F", "#{pane_id}")
    for pane in cp.stdout.split():
        _tmux("set", "-pu", "-t", pane, "@va_name")
        _tmux("select-pane", "-t", pane, "-T", "")
    return n


def prune() -> list[str]:
    """Drop registry entries whose pane is no longer alive (e.g. after a pane is
    closed). Idempotent — reconciles the registry against live panes. Returns the
    names removed. The closed pane's title dies with the pane, so nothing else to do."""
    targets = load()
    cp = _tmux("list-panes", "-a", "-F", "#{pane_id}")
    alive = set(cp.stdout.split())
    dead = [name for name, d in targets.items() if d.get("pane") not in alive]
    if dead:
        for name in dead:
            del targets[name]
        save(targets)
    return dead


def name_for_pane(pane: Optional[str]) -> Optional[str]:
    """The spoken name for a tmux pane id (e.g. '%5'), or None. Prefers the live
    per-pane @va_name option (what the border shows + what `register()` sets),
    falling back to a reverse lookup in the registry. Used by the done announcer
    to say WHICH named pane finished."""
    if not pane:
        return None
    cp = _tmux("display-message", "-p", "-t", pane, "#{@va_name}")
    name = cp.stdout.strip() if cp.returncode == 0 else ""
    if name:
        return name
    for n, d in load().items():
        if d.get("pane") == pane:
            return n
    return None


# --------------------------------------------------------------------------- #
# fuzzy name matching                                                          #
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


# Generic words shared across many pane names — too weak to identify a pane on
# their own (saying just "claude" must not single one out).
_GENERIC_TOKENS = {"claude", "pane", "terminal", "window", "tab", "bot", "agent"}


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _looks_like_pane_name(s: str) -> bool:
    """True if the phrase is clearly addressing a pane — its last word is a
    generic pane token ("Build Claude", "the build pane"). Lets dispatch report
    an unknown pane instead of silently dropping to chat."""
    toks = _norm(s).split()
    return bool(toks) and toks[-1] in _GENERIC_TOKENS


# Pronouns / ordinary objects of "tell"/"send" that are NOT pane names, so
# "tell me a joke" / "send me the link" never get reported as an unknown pane.
_CHAT_TARGETS = {"me", "us", "him", "her", "them", "you", "it", "this", "that",
                 "myself", "yourself", "everyone", "everybody", "someone",
                 "somebody", "anyone", "anybody", "all"}


def _resembles_pane(s: str, targets: dict) -> bool:
    """Loose check used ONLY to decide whether an UNRESOLVED dispatch target is a
    (mis-heard) pane worth reporting — vs ordinary chat ("tell me a joke"). NEVER
    used to auto-route. True when the phrase ends in a generic pane token, or is a
    plausible mishearing of a registered pane name (close ratio)."""
    nt = _norm(s)
    if not nt:
        return False
    toks = nt.split()
    if toks[-1] in _CHAT_TARGETS:
        return False
    if _looks_like_pane_name(s):
        return True
    for key in targets:
        kt = _norm(key).split()
        if _ratio(_norm(key), nt) >= 0.55 or _ratio(kt[0] if kt else "", toks[0]) >= 0.6:
            return True
    return False


# Last dispatched/attempted message, persisted (each hotkey press is a fresh
# process) so a follow-up "send it to <name>" can re-target it. TTL'd so an old
# message isn't resent much later.
_STASH = config.CACHE_DIR / "last-payload.json"
_STASH_TTL = 600.0  # seconds


def _stash_payload(text: str) -> None:
    try:
        _STASH.write_text(json.dumps({"text": text, "ts": time.time()}), encoding="utf-8")
    except OSError:
        pass


def _read_stash() -> str:
    try:
        d = json.loads(_STASH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if time.time() - float(d.get("ts", 0)) > _STASH_TTL:
        return ""
    return (d.get("text") or "").strip()


def ambiguous_matches(spoken: str, targets: dict) -> list[str]:
    """Registered keys a spoken phrase could equally mean because it's a shared
    token-prefix of several ("rice" -> Rice Claude, Rice Codex) — used to ASK which
    one instead of guessing. Empty when there's an exact match or fewer than two
    candidates share the prefix (i.e. _match_key can resolve it)."""
    want = _norm(spoken)
    want = re.sub(r"^(?:the|this|that|my|a)\s+", "", want)
    if not want:
        return []
    if any(_norm(k) == want for k in targets):
        return []                       # an exact match exists — not ambiguous
    wt = want.split()
    hits = []
    for key in targets:
        kt = _norm(key).split()
        # the whole spoken phrase is a leading-token subset of the name, e.g.
        # "rice" -> ["rice","claude"], "rice codex" -> exact (handled above)
        if kt[:len(wt)] == wt or (len(wt) == 1 and want in kt):
            hits.append(key)
    return hits if len(hits) >= 2 else []


def or_join(items) -> str:
    items = list(items)
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


def leading_pane_match(text: str, targets: Optional[dict] = None):
    """If the utterance LEADS with a registered pane name, return (key, payload).
    Tolerant per-token (ratio >= 0.85) so a singular/plural or minor STT variance on
    the name still routes — "stream command restart" reaches the pane named "Stream
    Commands" instead of falling through to the Nightbot 'stream command' intent.
    Longest names are tried first so a specific name beats a shorter prefix."""
    targets = load() if targets is None else targets
    toks = _norm(re.sub(r"^\s*hey\s+", "", text or "")).split()
    if not toks:
        return None
    for key in sorted(targets, key=lambda k: len(_norm(k).split()), reverse=True):
        kt = _norm(key).split()
        if not kt or len(kt) > len(toks):
            continue
        if all(_ratio(a, b) >= 0.85 for a, b in zip(kt, toks[:len(kt)])):
            payload = " ".join(toks[len(kt):])
            payload = re.sub(r"^(?:to|that)\b\s*", "", payload).strip()
            return key, payload
    return None


def _match_key(spoken: str, targets: dict) -> Optional[str]:
    """Best registry key for a spoken name, or None if nothing is close enough.

    Names share a generic suffix ("Build Claude", "Audio Claude", "Krita
    Claude"), so a plain full-string ratio over-credits the shared "claude" and
    would route "Build Claude" to "Audio Claude" — dangerous, since dispatch
    pastes commands into a pane. So the DISCRIMINATOR (the leading word, after a
    dropped article) must agree, unless the whole thing is near-identical or one
    clearly contains the other."""
    want = _norm(spoken)
    want = re.sub(r"^(?:the|this|that|my|a)\s+", "", want)  # drop a leading article
    if not want:
        return None
    wt = want.split()
    # Exact match wins GLOBALLY, before any containment/fuzzy — otherwise a
    # containment hit on an earlier-iterated key ("Krita Claude") steals a request
    # that has an exact match later ("Krita Claude 2").
    for key in targets:
        if _norm(key) == want:
            return key
    # A numeric discriminator must agree exactly: never conflate "Krita Claude",
    # "Krita Claude 2", "Krita Claude 3" via containment/fuzzy.
    want_nums = [t for t in wt if t.isdigit()]
    contain_hits = []
    best, best_score, second_score = None, 0.0, 0.0
    for key in targets:
        nk = _norm(key)
        kt = nk.split()
        if [t for t in kt if t.isdigit()] != want_nums:
            continue
        # containment only when the SHORTER side is a full multi-token phrase or a
        # distinctive (>=4 char) single word — not a generic fragment like "claude"
        short, long = (want, nk) if len(want) <= len(nk) else (nk, want)
        if short and short in long and (" " in short or len(short) >= 4) and short not in _GENERIC_TOKENS:
            contain_hits.append(key)
        full = _ratio(nk, want)
        first = _ratio(kt[0] if kt else "", wt[0] if wt else "")
        # accept only if near-identical overall, or the discriminator word agrees
        score = full if (full >= 0.8 or (first >= 0.7 and full >= 0.5)) else 0.0
        if score > best_score:
            best, best_score, second_score = key, score, best_score
        elif score > second_score:
            second_score = score
    # Containment is confident ONLY when exactly one key contains the phrase. When
    # the phrase is a shared prefix of several ("rice" -> Rice Claude AND Rice
    # Codex), it's ambiguous — DON'T arbitrarily return the first-iterated one (the
    # old bug: "rice" always hit whichever was registered first). The caller reports
    # the ambiguity (ambiguous_matches) so the discriminator can be asked for.
    if len(contain_hits) == 1:
        return contain_hits[0]
    if len(contain_hits) >= 2:
        return None
    # Fuzzy: reject a near-tie between the top two candidates — a phrase that scores
    # equally against two panes ("rice" vs both rice panes) has no clear winner.
    if best_score >= 0.6 and best_score - second_score >= 0.06:
        return best
    return None


# --------------------------------------------------------------------------- #
# tmux plumbing                                                                #
# --------------------------------------------------------------------------- #
def _tmux(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(("tmux", *args), capture_output=True, text=True)


def active_pane() -> Optional[str]:
    """The pane the user is actually looking at.

    Prefer the client the window manager currently has focused — tmux flags it
    `focused` when `focus-events on` and the terminal reports focus (Ghostty
    does). That tracks the window you're *looking at* even if you just clicked
    into it without typing, so "name this pane X" names the right one when several
    Ghostty windows are open. Fall back to the most recently active client when
    none reports focus (focus events unavailable). Never use a bare
    `display-message -p` with no client — it picks an arbitrary (possibly
    detached) session."""
    cp = _tmux("list-clients", "-F", "#{client_activity}\t#{client_flags}\t#{client_name}")
    rows = [ln.split("\t", 2) for ln in cp.stdout.splitlines() if ln.count("\t") >= 2]
    if not rows:
        return None
    focused = [r for r in rows if "focused" in r[1].split(",")]
    pool = focused or rows
    pool.sort(key=lambda r: int(r[0] or 0), reverse=True)  # newest wins if tied
    client = pool[0][2]
    dm = _tmux("display-message", "-p", "-t", client, "#{pane_id}")
    pane = dm.stdout.strip()
    return pane or None


def pane_alive(pane: str) -> bool:
    cp = _tmux("list-panes", "-a", "-F", "#{pane_id}")
    return pane in cp.stdout.split()


# "This pane is busy" = Claude Code is mid-turn. The reliable signal on the current
# build is the live spinner status line, whose shape is
#     <glyph> <Verb>… (<n>s · <hint>)
# e.g. "✻ Channelling… (10s · thinking)", "✽ … (16s · ↓ 25 tokens · thought for 13s)".
# The "…(<n>s" elapsed-seconds counter is present for the whole turn and gone the
# instant it finishes — and it does NOT appear in idle chrome or completed output
# (those show "Thought for 13s", "⎿ …" etc., never "…(<n>s"). The glyph and verb
# rotate, so we anchor on the timer, not them. We OR in the older "(esc to
# interrupt)" footer so this also works on builds/tool-run states that show it
# (and so it's robust if the spinner line momentarily isn't captured). NB: the
# bypass-permissions prompt says "Esc to cancel" — deliberately not matched.
_RE_BUSY = re.compile(r"…\s*\(\s*\d+\s*s\b|esc\b[^.\n]*\binterrupt\b", re.IGNORECASE)


def pane_busy(pane: str) -> bool:
    """True when the target TUI is mid-run — detected by Claude Code's
    "(esc to interrupt)" footer in the pane's visible content. Best-effort: on any
    tmux/capture failure we report not-busy so a dispatch still goes through."""
    cp = _tmux("capture-pane", "-p", "-t", pane)
    if cp.returncode != 0:
        return False
    return bool(_RE_BUSY.search(cp.stdout))


def send(pane: str, payload: str) -> bool:
    """Bracketed-paste `payload` into `pane`, then submit with Enter. Bracketed
    paste (-p) makes a TUI like Claude Code treat a multiline payload as one paste
    rather than executing each line; the separate Enter submits it.

    If the pane is mid-run (pane_busy), send Escape FIRST to interrupt the current
    turn, so a follow-up message takes over immediately instead of queuing behind
    the running turn — matching the user telling it the next thing while it's still
    going. The Escape is sent as a real keypress (not pasted), then we wait
    DISPATCH_INTERRUPT_DELAY for Claude Code to return to an empty prompt."""
    if not pane_alive(pane):
        return False
    if pane_busy(pane):
        _tmux("send-keys", "-t", pane, "Escape")
        time.sleep(config.DISPATCH_INTERRUPT_DELAY)
    if _tmux("set-buffer", "-b", "va_dispatch", "--", payload).returncode != 0:
        return False
    _tmux("paste-buffer", "-p", "-d", "-b", "va_dispatch", "-t", pane)
    # Let the target TUI finish ingesting the bracketed paste before we submit.
    # Claude Code debounces input right after a paste (so a multiline paste isn't
    # run line-by-line); an Enter sent inside that window is swallowed into the
    # paste instead of submitting. A FOCUSED/visible pane reproduces this reliably
    # — its render loop runs full-speed so the paste + an immediate Enter arrive in
    # one read tick — which is why a focused pane pasted the message but never
    # started it. A short gap lands the Enter after the debounce, so it submits
    # whether the pane is focused or not.
    time.sleep(config.DISPATCH_ENTER_DELAY)
    _tmux("send-keys", "-t", pane, "Enter")
    return True


# --------------------------------------------------------------------------- #
# acks — short, natural, action-describing (never repeat the payload verbatim)  #
# --------------------------------------------------------------------------- #
_SENT_ACKS = (
    "Got it — sent that to {name}.",
    "On it, {name}'s got it.",
    "Sounds good, over to {name}.",
    "Done — passed that to {name}.",
)


def _ack(name: str) -> str:
    return _SENT_ACKS[int(time.time()) % len(_SENT_ACKS)].format(name=name)


# --------------------------------------------------------------------------- #
# grammar                                                                      #
# --------------------------------------------------------------------------- #
# Leading filler/wake words Whisper loves to prepend ("Hey, …", "Okay, so …",
# "actually …"). Stripped (with trailing punctuation) before any matching so the
# real command sits at the start where the anchored regexes expect it.
_RE_LEAD_FILLER = re.compile(
    r"^(?:(?:hey|hi|hello|ok|okay|so|well|um+|uh+|yeah|yep|yup|alright|"
    r"actually|now)\b[\s,.:;!?-]*)+",
    re.IGNORECASE)
# Leading bullet/dash/ellipsis/whitespace Whisper sometimes prepends ("- name this
# pane …"); stripped before the start-anchored grammars so it can't drop a real
# command to chat. (Also stripped at STT in stt._normalize; this covers the text
# CLI path + anything filler-stripping exposes.)
_LEAD_JUNK = re.compile(r"^[\s\-‐-―*•·.,:;!?]+")

# Naming REQUIRES a deictic ("this/it/here") or a pane noun so ordinary chat
# ("name three colors", "call my mom") isn't mistaken for naming a pane — this
# matters now that route detection shares the chat hotkey. "pain/panel" are
# included because Whisper routinely mishears "pane" as "pain" (and "panel").
# `rest` is everything after the noun; _name_after() pulls the actual name out of
# it, so trailing-clause phrasings work ("name the pane I'm focused on, X").
_PANE_NOUN = r"(?:panes?|pains?|panels?|tabs?|windows?|terminals?)\b"
_RE_NAME = re.compile(
    r"^(?:hey\s+)?(?:can you\s+|could you\s+|would you\s+|please\s+)?"
    r"(?:re)?(?:name|call|register|label|tag)\s+"
    r"(?:"
    r"(?:this|it|here)(?:\s+" + _PANE_NOUN + r")?"             # deictic (+ opt noun)
    r"|(?:(?:the|that|my|a|current(?:ly)?|focus(?:ed|ing)?|select(?:ed)?|active|open)\s+)*"
    + _PANE_NOUN +                                              # noun w/ opt modifiers
    r")"
    r"\s*(?P<rest>.*?)[.!?]*$",
    re.IGNORECASE,
)


def _name_after(rest: str) -> str:
    """Pull the spoken name out of whatever follows the pane noun. The name is
    always the TAIL of the utterance (the user names one pane and ends right
    after). So if there's a clause break — comma OR period (Whisper sentence
    split) — the name is the last segment ("…I'm focused on, X" / "…right now. X").
    Otherwise strip a leading "(that) I'm focused on/right now …" filler so a bare
    "name the pane I'm focused on X" still yields X."""
    s = rest.strip()
    parts = [p.strip() for p in re.split(r"[.,;]+", s) if p.strip()]
    if len(parts) > 1:
        s = parts[-1]                                 # name = last clause/sentence
    # no delimiter: drop a leading relative clause up to "on"/"now" ("I'm focused on")
    s = re.sub(r"^(?:that\s+)?i\b.*?\b(?:on|now)\b", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^\s*(?:as|to|is|it'?s)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(?:right\s+now|currently)\b", "", s, flags=re.IGNORECASE)
    return s.strip(" .,!?")
# "tell <pane> to/that/: <payload>" — also accepts a polite prefix ("can you
# tell …") and a SENTENCE BREAK as the separator ("tell Hermes Claude? we're
# testing …", "tell Hermes Claude. do X"), since you don't always say "to". The
# target is only dispatched if it matches a registered pane (guard in try_route),
# so loosening the separator can't hijack ordinary "tell me a joke".
_RE_DISPATCH = re.compile(
    r"^(?:hey\s+)?(?:can you\s+|could you\s+|would you\s+|please\s+)?"
    r"(?:tell|ask|send|have|route|message|paste)\s+(.+?)"
    r"(?:\s+(?:to|that)\s+|\s*[:,.?!]\s+)(.+?)[.!?]*$",
    re.IGNORECASE | re.DOTALL,
)
_RE_FORGET = re.compile(r"^(?:hey\s+)?forget\s+(.+?)[.!?]*$", re.IGNORECASE)
# "send it to <name>" / "pass that over to <name>" / "forward the message to
# <name>" — RE-dispatch the LAST message (stashed) to a pane. The object is a
# back-reference (it/that/this/the message), NOT a payload, so this is distinct
# from _RE_DISPATCH and MUST be checked first — otherwise _RE_DISPATCH parses
# "it" as the target pane. Used to recover after a mis-heard/wrong target.
_RE_SEND_IT = re.compile(
    r"^(?:hey\s+)?(?:no,?\s+|actually,?\s+|instead,?\s+)*"
    r"(?:send|pass|give|route|forward|paste|put|deliver)\s+"
    r"(?:it|that|this|the\s+(?:message|same\s+thing|last\s+(?:one|message)))\s+"
    r"(?:over\s+)?(?:to|into)\s+(.+?)[.!?]*$",
    re.IGNORECASE)
# Anchored to the start so listing only fires when it's the FIRST thing you said
# ("list my panes", "show me the panes", "what panes do I have") — not buried
# mid-sentence ("anyway show me what the panes are doing"). "what" counts only
# when the noun follows it directly, so "what should I name this" stays chat.
_RE_LIST = re.compile(
    r"^(?:hey\s+)?(?:"
    r"(?:list|show|which)\b.*\b(?:targets?|panes?|names?)\b"
    r"|what\b\s+(?:are\s+)?(?:my\s+|the\s+)?(?:targets?|panes?|names?)\b"
    r")",
    re.IGNORECASE)
# "what's the name of THIS pane?" — a query about the single focused pane (vs.
# _RE_LIST which lists them all). Checked before _RE_LIST in try_route because
# "which pane is this" would otherwise be caught by the list grammar. Answered
# deterministically from the registry — Hermes can't see tmux, so this must NOT
# fall through to chat (where "this pain" got read as a literal ache).
_RE_WHATIS = re.compile(
    r"^(?:hey\s+)?(?:"
    r"what(?:'?s| is| was)?\s+(?:the\s+)?name\s+of\s+(?:this|the)\s+(?:focused\s+|current\s+|active\s+)?" + _PANE_NOUN
    + r"|what(?:'?s| is)?\s+(?:this|the)\s+(?:focused\s+|current\s+|active\s+)?" + _PANE_NOUN + r"(?:'?s)?\s+(?:called|named|name)"
    + r"|wh(?:at|ich)\s+" + _PANE_NOUN + r"\s+(?:is\s+this|am\s+i\s+(?:in|on))"
    + r")",
    re.IGNORECASE)
# Reset ALL pane names. Matches "pane name reset", "reset pane names", "clear all
# pane names", "forget all panes", "wipe the panel names", etc. ("panel" too —
# STT often hears it that way.) Must be specific enough not to eat ordinary chat.
_RE_RESET = re.compile(
    r"^(?:hey\s+)?(?:"
    r"(?:reset|clear|wipe|forget|remove)\s+(?:all\s+)?(?:the\s+|my\s+)?"
    r"(?:pane|panel|tab|window|terminal)s?(?:\s*(?:name|names|title|titles))?"
    r"|(?:pane|panel)s?\s*(?:name|names|title|titles)?\s*reset"
    r")[.!?]*$",
    re.IGNORECASE,
)
# Clear ONE pane's name. Two shapes, both distinct from _RE_RESET (which wipes ALL
# and only matches when NO name trails it):
#   THIS = the focused pane: "clear this pane name", "clear the name of this pane",
#          "clear this pane's name", "forget this pane".
#   NAME = a registered pane: "clear pane name X", "remove the pane named X",
#          "clear name X".
_RE_CLEAR_THIS = re.compile(
    r"^(?:hey\s+)?(?:clear|remove|delete|unname|unregister|forget)\s+(?:the\s+)?"
    r"(?:name\s+of\s+)?(?:this|the\s+(?:focused|current|active)\s+)"
    + _PANE_NOUN + r"(?:'?s)?(?:\s+name)?[.!?]*$",
    re.IGNORECASE)
_RE_CLEAR_NAME = re.compile(
    r"^(?:hey\s+)?(?:clear|remove|delete|unname|unregister)\s+(?:the\s+)?"
    r"(?:(?:pane|panel)\s+name(?:d)?|name)\s+(.+?)[.!?]*$",
    re.IGNORECASE)


# "start a Claude/Codex session [in this pane]" — launch a fresh agent CLI in a
# tmux pane by voice. Accepts BOTH parameter orderings (the user's ask): the tool
# word can come BEFORE "session" ("start a claude session") or AFTER it ("start a
# session in this pane, codex"). Anchored to a start verb + the word "session" so
# ordinary chat can't trip it; the tool word + optional pane target are pulled out
# separately (below) rather than positionally, which is what lets either order work.
_RE_START_SESSION = re.compile(
    r"^(?:hey\s+)?(?:can you\s+|could you\s+|would you\s+|please\s+)?"
    r"(?:start|launch|open|spin\s*up|fire\s*up|boot(?:\s*up)?|create|make|begin|"
    r"run|kick\s*off|spawn)\s+"
    r"(?:a\s+|an\s+|another\s+|new\s+|fresh\s+)*"
    r".*\bsession\b.*$",
    re.IGNORECASE)
# Tool aliases, enumerated to survive Whisper. "claude" is already normalized from
# clawed/claud/… at STT (stt._normalize); the extra spellings cover the text CLI
# path (`do`/`ask`) that bypasses that normalization. "codex" is heard as codecs/
# code x/cortex/kotex etc. — `code` alone is deliberately EXCLUDED (too common).
_RE_TOOL_CLAUDE = re.compile(r"\b(?:claude|claud|clawed|clawde|clawd|clode)\b", re.IGNORECASE)
_RE_TOOL_CODEX = re.compile(r"\b(?:codex|codecs?|code\s*x|kodex|cortex|kotex|codeex)\b",
                            re.IGNORECASE)
# The pane target when named explicitly: "... in/into/on <phrase>". <phrase> is
# matched against the registry; "this pane"/"here"/generic words fall back to the
# focused pane. Anchored to the LAST "in"/"into"/"on" so "start a codex session in
# the build pane" grabs "the build pane", not an earlier word.
_RE_SESSION_TARGET = re.compile(r"\b(?:in|into|on|inside)\s+(.+?)[.!?]*$", re.IGNORECASE)


def run_in_pane(pane: str, command: str) -> bool:
    """Type a shell `command` into `pane` and run it (literal keys + Enter). Unlike
    send(), this is NOT a bracketed paste — it's a command typed at the pane's shell
    prompt, so the launched session inherits that pane's working directory. Returns
    False if the pane is gone."""
    if not pane_alive(pane):
        return False
    _tmux("send-keys", "-t", pane, "--", command)
    _tmux("send-keys", "-t", pane, "Enter")
    return True


def end_session_in_pane(pane: str) -> bool:
    """Gracefully end the agent session running in `pane` by sending the configured
    exit keys (default Ctrl-C twice — quits Claude Code / aborts Codex back to the
    shell). Returns False if the pane is gone. The pane and its name survive; only
    the session inside it ends."""
    if not pane_alive(pane):
        return False
    for k in config.SESSION_END_KEYS:
        _tmux("send-keys", "-t", pane, k)
        time.sleep(config.SESSION_END_KEY_DELAY)
    return True


# "end session <pane name>" — stop the agent CLI running in a pane. The pane name
# follows "session" directly ("end session Krita Claude"), or via "in"/"here"/no
# target (→ focused pane). Many verbs so it's forgiving; anchored to a verb + the
# word "session" so ordinary chat can't trip it.
_RE_END_SESSION = re.compile(
    r"^(?:hey\s+)?(?:can you\s+|could you\s+|please\s+)?"
    r"(?:end|quit|close|kill|stop|exit|terminate|finish|shut\s*down)\s+"
    r"(?:the\s+|this\s+|my\s+|a\s+)?"
    r"session\b\s*(?P<rest>.*?)[.!?]*$",
    re.IGNORECASE)


# Filler stripped from an "in <phrase>" target so a deictic ("this pane") reduces
# to nothing (→ focused pane) while a real name ("the build pane") survives.
_RE_TARGET_FILLER = re.compile(
    r"\b(?:this|the|that|my|current|currently|active|focused|focus|here|new|a|an|"
    r"session|pane|panes|window|windows|terminal|terminals|tab|tabs)\b",
    re.IGNORECASE)


def _session_target(text: str, targets: dict) -> tuple[Optional[str], Optional[str]]:
    """Resolve which pane to start the session in. Returns (pane_id, registered_name).

    Default is the FOCUSED pane ("start a Claude session in this pane" / "…here") →
    (active_pane(), None). If an explicit "in/into/on <phrase>" names a REGISTERED
    pane, that wins → (its pane, its name). Tool words and deictic/generic filler
    are stripped from the phrase first, so "in this pane, codex" and "in the build
    pane" both reduce correctly (the former to nothing → focused)."""
    m = _RE_SESSION_TARGET.search(text)
    if m:
        phrase = _RE_TOOL_CLAUDE.sub(" ", m.group(1))
        phrase = _RE_TOOL_CODEX.sub(" ", phrase)
        phrase = _norm(_RE_TARGET_FILLER.sub(" ", phrase))
        if phrase:
            key = _match_key(phrase, targets)
            if key is not None:
                return targets[key]["pane"], key
    return active_pane(), None


def try_route(text: str) -> Optional[Result]:
    """Detect a *confident* route intent and execute it; return None when the
    utterance isn't routing (so the caller falls through to chat). Shared by the
    chat hotkey — so it must NOT claim ambiguous chat as a route."""
    # Collapse newlines/whitespace first: Whisper splits a long utterance across
    # lines at sentence breaks, and an embedded newline makes the start-anchored
    # regexes (no re.DOTALL) fail to match — "name … right now.\nHermes Claude"
    # would silently fall through to chat. Voice has no meaningful newlines.
    text = re.sub(r"\s+", " ", (text or "")).strip()
    # Strip a leading bullet/dash/ellipsis Whisper sometimes prepends ("- name this
    # pane …") which would break the start-anchored grammars below. Defensive: the
    # STT path already strips it in stt._normalize, but the text CLI path doesn't.
    text = _LEAD_JUNK.sub("", text)
    # Strip leading filler/wake words AND their trailing punctuation. Whisper
    # almost always prepends "Hey," / "Okay," / "So," — and the trailing COMMA
    # (not just a space) was breaking the start-anchored verb regexes, sending
    # "Hey, name this pane X" to chat instead of routing. This only affects route
    # detection; chat still sees the caller's original text.
    text = _RE_LEAD_FILLER.sub("", text).strip()
    text = _LEAD_JUNK.sub("", text)   # again, in case filler removal exposed a dash
    if not text:
        return None

    targets = load()

    # --- start a fresh Claude/Codex session in a pane ----------------------
    # Highly specific (start-verb + "session" + a tool word), so it's safe up top
    # and can't be stolen by a later grammar (none of which begin with "start").
    if _RE_START_SESSION.match(text):
        pane, matched_name = _session_target(text, targets)
        has_claude = bool(_RE_TOOL_CLAUDE.search(text))
        has_codex = bool(_RE_TOOL_CODEX.search(text))
        # A tool word that's actually part of the target PANE name ("Krita Claude")
        # is not the tool being launched — drop it so "start a codex session in
        # Krita Claude" reads as codex, not an ambiguous both.
        if matched_name:
            nm = matched_name.lower()
            if _RE_TOOL_CLAUDE.search(nm):
                has_claude = False
            if _RE_TOOL_CODEX.search(nm):
                has_codex = False
        if not has_claude and not has_codex:
            return None                  # no tool named -> not ours; let chat handle it
        if has_claude and has_codex:     # genuinely ambiguous (both named)
            return Result(False, "Which one — a Claude session or a Codex session?")
        tool = "claude" if has_claude else "codex"
        if pane is None:
            return Result(False, "I can't tell which pane to start it in.")
        if not run_in_pane(pane, config.SESSION_LAUNCH.get(tool, tool)):
            return Result(False, "That pane is gone.")
        where = f"in {matched_name}" if matched_name else "here"
        return Result(True, f"Starting a {tool.capitalize()} session {where}.")

    # --- end the agent session running in a pane ---------------------------
    m = _RE_END_SESSION.match(text)
    if m:
        # <rest> is the target: a pane name follows "session" directly, optionally
        # after a preposition ("in Krita Claude"); a deictic/"here"/empty rest means
        # the focused pane.
        rest = re.sub(r"^\s*(?:in|into|on|for|inside|of)\b", " ", m.group("rest"),
                      flags=re.IGNORECASE)
        rest = _norm(_RE_TARGET_FILLER.sub(" ", rest))
        if rest:
            key = _match_key(rest, targets)
            if key is None:
                amb = ambiguous_matches(rest, targets)
                if amb:                           # "end session rice" with two rice panes
                    return Result(False, f"Which one — {or_join(amb)}?")
                return Result(False, f"I don't have a pane named {m.group('rest').strip()}.")
            pane, name = targets[key]["pane"], key
        else:                                     # "end this session" / "…here"
            pane = active_pane()
            name = next((n for n, d in targets.items() if d.get("pane") == pane), None)
        if pane is None:
            return Result(False, "I can't tell which pane's session to end.")
        if not end_session_in_pane(pane):
            return Result(False, f"{name}'s pane is gone." if name else "That pane is gone.")
        return Result(True, f"Ending the session in {name}." if name
                      else "Ending the session here.")

    # --- "what's the name of this pane?" -> report the focused pane's name --
    if _RE_WHATIS.match(text):
        pane = active_pane()
        if pane is None:
            return Result(False, "I can't tell which pane is focused right now.")
        here = next((n for n, d in targets.items() if d.get("pane") == pane), None)
        return Result(True, f"This pane is {here}." if here
                      else "This pane doesn't have a name yet.")

    # --- list ("list my panes") --------------------------------------------
    if _RE_LIST.search(text):
        if not targets:
            return Result(True, "No named panes yet.")
        return Result(True, f"Named panes: {', '.join(targets.keys())}.")

    # --- reset ALL pane names (checked before forget/name) ------------------
    if _RE_RESET.match(text):
        n = reset()
        return Result(True, "Pane names reset." if n else "No pane names to reset.")

    # --- clear THIS (focused) pane's name ----------------------------------
    if _RE_CLEAR_THIS.match(text):
        pane = active_pane()
        here = next((n for n, d in targets.items() if d.get("pane") == pane), None) if pane else None
        if here is None:
            return Result(True, "This pane doesn't have a name to clear.")
        forget(here)
        return Result(True, f"Cleared the name {here}.")

    # --- clear a NAMED pane ("clear pane name X") --------------------------
    m = _RE_CLEAR_NAME.match(text)
    if m:
        want = m.group(1).strip()
        key = _match_key(want, targets)
        if key is None:
            return Result(False, f"I don't have a pane named {want}.")
        forget(key)
        return Result(True, f"Cleared the name {key}.")

    # --- name this pane (deictic required by _RE_NAME) ----------------------
    m = _RE_NAME.match(text)
    if m:
        name = _name_after(m.group("rest"))
        if not name:
            return Result(False, "What should I call this pane?")
        pane = active_pane()
        if pane is None:
            return Result(False, "I don't see an attached terminal to name.")
        register(name, pane)
        return Result(True, f"Got it — this pane is now {name}.")

    # --- forget <name> — only if it names a registered target ---------------
    m = _RE_FORGET.match(text)
    if m and _match_key(m.group(1).strip(), targets) is not None:
        name = m.group(1).strip()
        forget(name)
        return Result(True, f"Forgot {name}.")

    # --- "send it to <name>" — re-dispatch the LAST message to another pane --
    # Checked before _RE_DISPATCH so "send it to Krita Claude" isn't parsed as a
    # dispatch to a pane literally named "it". The payload is whatever was last
    # dispatched/attempted (stashed), so a correction after a mis-heard name or a
    # wrong target works.
    m = _RE_SEND_IT.match(text)
    if m:
        tgt = m.group(1).strip()
        key = _match_key(tgt, targets)
        if key is None:
            amb = ambiguous_matches(tgt, targets)
            if amb:
                return Result(False, f"Which one — {or_join(amb)}?")
            if _resembles_pane(tgt, targets):
                return Result(False, f"I don't have a pane named {tgt}.")
            return None
        payload = _read_stash()
        if not payload:
            return Result(False, f"I don't have a recent message to send. Say, tell "
                                 f"{key} to, then your message.")
        return _do_send(key, payload, targets)

    # --- dispatch: tell <name> to <payload> — only if <name> is registered --
    m = _RE_DISPATCH.match(text)
    if m:
        tgt = m.group(1).strip()
        payload = m.group(2).strip()
        key = _match_key(tgt, targets)
        if key is not None:
            return _do_send(key, payload, targets)
        # Ambiguous target ("tell rice to …" with Rice Claude AND Rice Codex): ask
        # which, and stash the payload so a follow-up "send it to Rice Codex" works.
        amb = ambiguous_matches(tgt, targets)
        if amb:
            _stash_payload(payload)
            return Result(False, f"Which one — {or_join(amb)}?")
        # Unresolved, but clearly a pane (ends in a generic token like "claude"/
        # "pane", OR a close mis-hearing of a registered name): say so AND stash
        # the payload so a follow-up "send it to <correct name>" works. Don't fall
        # to chat — that's where Hermes ad-libbed / misfired into an OBS action.
        if _resembles_pane(tgt, targets):
            _stash_payload(payload)
            return Result(False, f"I don't have a pane named {tgt}.")
        return None  # "tell me about X" etc. -> let chat handle it

    # --- a registered name explicitly appears in the utterance --------------
    return _name_in_text(text, targets)


def handle(text: str) -> Result:
    """Explicit route/dispatch entry (CLI `route`/`dispatch`): always report,
    even when nothing matched."""
    res = try_route(text)
    if res is not None:
        return res
    if not (text or "").strip():
        return Result(False, "I didn't catch that.")
    if not load():
        return Result(False, "No named panes yet. Focus one and say, name this, then a name.")
    return Result(False, "I'm not sure which pane you meant.")


def _name_in_text(text: str, targets: dict) -> Optional[Result]:
    """Addressing style: a registered name is the FIRST thing you said, and the
    rest is the payload ("Krita Claude, add a double jump"). The name must lead
    the utterance — a name buried mid-sentence ("I love what Krita Claude did")
    is NOT a route, so it falls through to chat. Returns None when no registered
    name leads the text. (`tell <name> to …` is handled earlier by _RE_DISPATCH.)"""
    m = leading_pane_match(text, targets)
    if m and m[1]:
        return _do_send(m[0], m[1], targets)
    return None


def _mark_voice(pane: str) -> None:
    """Stamp this pane as having just received a VOICE-routed command, so the
    done-announcer (in voice-only mode) knows the resulting completion was
    voice-initiated and not something the user typed in directly."""
    try:
        config.DONE_VOICE_MARK_DIR.mkdir(parents=True, exist_ok=True)
        config.voice_mark_path(pane).write_text(str(time.time()), encoding="utf-8")
    except OSError:
        pass


def _do_send(key: str, payload: str, targets: dict) -> Result:
    pane = targets[key]["pane"]
    if not pane_alive(pane):
        return Result(False, f"{key}'s pane is gone.")
    _stash_payload(payload)          # remember it for a follow-up "send it to <name>"
    if not send(pane, payload):
        return Result(False, f"I couldn't reach {key}.")
    _mark_voice(pane)                # tell the done-announcer this task is voice-initiated
    return Result(True, _ack(key))


def send_named(name: str, payload: str) -> Optional[Result]:
    """Resolve a spoken pane name to a registered target and dispatch payload to
    it. Returns the Result, or None if no registered pane matches. Used by the
    Hermes `[[ask: <pane> | <msg>]]` forwarding so the chat brain can hand an
    unsupported request to a builder pane (e.g. 'Hermes Claude')."""
    targets = load()
    key = _match_key(name, targets)
    if key is None:
        return None
    return _do_send(key, payload, targets)
