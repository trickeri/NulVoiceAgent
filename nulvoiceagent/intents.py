"""Deterministic *parameterized* intents — spoken commands that carry ARGUMENTS
the allowlist classifier can't pass (it only picks a bare action name).

Matched before either brain (like dispatch.try_route for panes): pure regex, no
LLM cost. The safety guarantee is preserved because the command TEMPLATE is
hardcoded here and only validated integers/strings are interpolated — there is no
free-form command for the model (or speech) to inject.

Add a parser, append it to _PARSERS. Each returns an Intent(say, argv) or None.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from typing import Callable, Optional


class Intent:
    def __init__(self, say: str, argv: list[str]):
        self.say = say      # short spoken confirmation
        self.argv = argv    # exact argv to run (list form — never shell-joined)


# A parser inspects the utterance and returns an Intent or None.
Parser = Callable[[str], Optional[Intent]]


def _norm(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — the shared shape every
    "is the whole utterance just a command?" test works on."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).split())


# --- Krita: new document / resize with spoken dimensions -------------------- #
# Width then height, in any of the spoken shapes Trickery uses:
#   "...1920 wide 1080 high", "1920 width 1080 height", "1920 x 1080 y",
#   "1920 by 1080", "1920 px wide 1080 px tall".
_RE_WH = re.compile(
    r"(\d{2,5})\s*(?:px|pixels?)?\s*(?:wide|width|w|x|across)\b[^\d]*?"
    r"(\d{2,5})\s*(?:px|pixels?)?\s*(?:high|height|tall|h|y)\b"
    r"|(\d{2,5})\s*(?:px|pixels?)?\s*(?:by|x|times|×)\s*(\d{2,5})",
    re.IGNORECASE)
_RE_DPI = re.compile(r"(\d{2,4})\s*(?:dpi|ppi)\b", re.IGNORECASE)
_RE_NEW = re.compile(r"\b(?:new|create|make|start|open)\b", re.IGNORECASE)
_RE_RESIZE = re.compile(r"\b(?:resize|resample|rescale|scale|set|change|adjust)\b",
                        re.IGNORECASE)
_DOC_WORDS = ("document", "doc", "canvas", "file", "image")
_SIZE_WORDS = ("document", "doc", "canvas", "image", "size", "dimension",
               "dimensions", "resolution")


def try_krita_new(text: str) -> Optional[Intent]:
    """"create a new Krita document 1920 wide 1080 high" -> nulpaint new with
    those pixel dimensions. Returns None when it's not a Krita-new request or no
    dimensions were spoken (so the plain `krita.new` default-size action handles
    those)."""
    low = (text or "").lower()
    if "krita" not in low or not _RE_NEW.search(low):
        return None
    if not any(w in low for w in _DOC_WORDS):
        return None
    m = _RE_WH.search(text or "")
    if not m:
        return None
    w = int(m.group(1) or m.group(3))
    h = int(m.group(2) or m.group(4))
    if not (1 <= w <= 30000 and 1 <= h <= 30000):
        return None  # implausible — likely a misheard number, don't act
    argv = ["nulpaint", "new", "--width", str(w), "--height", str(h)]
    say = f"Creating a new {w} by {h} Krita document."
    d = _RE_DPI.search(text or "")
    if d and 1 <= int(d.group(1)) <= 2400:
        argv += ["--resolution", d.group(1)]
        say = f"Creating a new {w} by {h} Krita document at {d.group(1)} DPI."
    return Intent(say, argv)


def try_krita_resize(text: str) -> Optional[Intent]:
    """"set the current Krita document to 1920 by 1080" / "resize the canvas to
    800 wide 600 high" -> nulpaint resize. Default scales the image (resamples);
    say "canvas"/"crop"/"extend" to change bounds without resampling. Returns
    None unless it's clearly a resize of the EXISTING doc (has a resize verb)."""
    low = (text or "").lower()
    if "krita" not in low or not _RE_RESIZE.search(low):
        return None
    if "new" in low:                      # "...a NEW document..." is create, not resize
        return None
    if not any(w in low for w in _SIZE_WORDS):
        return None
    m = _RE_WH.search(text or "")
    if not m:
        return None
    w = int(m.group(1) or m.group(3))
    h = int(m.group(2) or m.group(4))
    if not (1 <= w <= 30000 and 1 <= h <= 30000):
        return None
    canvas = any(k in low for k in ("canvas", "crop", "extend", "trim"))
    argv = ["nulpaint", "resize", str(w), str(h)] + (["--canvas"] if canvas else [])
    verb = "Resizing the canvas to" if canvas else "Scaling the document to"
    return Intent(f"{verb} {w} by {h}.", argv)


# --- Kdenlive: cut/split all clips at the playhead -------------------------- #
# Fires the fork's `cut_timeline_all_clips` action via the scripting bridge
# (razors every track at the playhead). Deterministic so a bare "cut" is instant.
# Anchored + tightly enumerated so "cut to the chase"/"cut it out" stay chat.
_RE_KDEN_CUT = re.compile(
    r"^(?:hey\s+)?(?:"
    r"(?:make|do|add)\s+a\s+(?:cut|split)"
    r"|(?:cut|split|razor|slice|blade)"
    r"(?:\s+(?:it|this|here|now|them|the\s+clips?|all(?:\s+(?:the\s+)?clips?)?"
    r"|at\s+(?:the\s+)?playhead))?"
    r")[.!?]*$",
    re.IGNORECASE)
_KDEN_CUT_ARGV = ["qdbus6", "org.kde.kdenlive.scripting", "/kdenlive",
                  "triggerAction", "cut_timeline_all_clips"]


def try_kdenlive_cut(text: str) -> Optional[Intent]:
    """"cut" / "split" / "cut all clips" -> razor every track at the playhead in
    Kdenlive (fork action `cut_timeline_all_clips` via the scripting bridge)."""
    if not _RE_KDEN_CUT.match((text or "").strip()):
        return None
    return Intent("Cut.", list(_KDEN_CUT_ARGV))


# --- Kdenlive: new project at spoken dimensions/format ---------------------- #
# "new Kdenlive project 1920 by 1080", "new Kdenlive document vertical HD",
# "create a new Kdenlive project 4k at 60 fps" -> the fork's newProjectFormat,
# which reuses/creates the matching MLT profile itself.
_RE_FPS = re.compile(r"(\d{2,3})\s*(?:fps|frames?\s*per\s*second|hz)", re.IGNORECASE)
_PROJ_WORDS = ("project", "document", "doc")
# ASR consistently mishears "Kdenlive" as "Caden/Kaden Live", "CadenLive", etc.
# Match those so the deterministic intent fires (drives the RUNNING Kdenlive over its
# D-Bus scripting bridge) instead of falling to Hermes — which made a raw .kdenlive
# file on disk rather than using the open project. (Scoped to the kdenlive intents so
# it can't clash with the pane literally named "Kaden Live Claude".)
_KDENLIVE_RE = re.compile(r"\b[kc]a?den\s?live\b", re.IGNORECASE)


def _format_dims(low: str) -> Optional[tuple[int, int]]:
    """A format/resolution phrase -> a target (width, height) in landscape terms,
    flipped for vertical / equalized for square. None if no format cue is present."""
    if re.search(r"\b4\s*k\b|\buhd\b|\bultra\s*hd\b|\b2160\b", low):
        long_, short = 3840, 2160
    elif re.search(r"\b2\s*k\b|\bqhd\b|\b1440\b", low):
        long_, short = 2560, 1440
    elif re.search(r"\b720\b", low):
        long_, short = 1280, 720
    elif re.search(r"\b1080\b|\bfull\s*hd\b|\bfhd\b|\bhd\b", low):
        long_, short = 1920, 1080
    else:
        long_, short = 1920, 1080   # bare "vertical"/"square" default to 1080-class
        if not re.search(r"\b(vertical|portrait|square|horizontal|landscape|widescreen)\b", low):
            return None
    if re.search(r"\bsquare\b", low):
        return (short, short)
    if re.search(r"\b(vertical|portrait|vert)\b", low):
        return (short, long_)
    return (long_, short)


def try_kdenlive_new_project(text: str) -> Optional[Intent]:
    low = (text or "").lower()
    if not _KDENLIVE_RE.search(low) or not _RE_NEW.search(low):
        return None
    if not any(w in low for w in _PROJ_WORDS):
        return None
    w = h = None
    m = _RE_WH.search(text or "")
    if m:
        w = int(m.group(1) or m.group(3))
        h = int(m.group(2) or m.group(4))
    else:
        dims = _format_dims(low)
        if dims:
            w, h = dims
    if not w or not (1 <= w <= 30000 and 1 <= h <= 30000):
        return None  # no usable dimensions/format spoken
    fm = _RE_FPS.search(low)
    fps = int(fm.group(1)) if fm else 30   # newProjectFormat needs a concrete fps
    # newProjectFormat(width, height, fps) reuses or creates the matching MLT profile
    # ITSELF (handles NTSC fractional rates), so we pass raw dimensions — no fragile
    # "guess the installed profile path" step. Requires the RUNNING Kdenlive to be the
    # rebuilt binary (restart once after a fork rebuild to expose new bridge methods).
    return Intent(f"Creating a new {w} by {h} Kdenlive project at {fps} fps.",
                  ["qdbus6", "org.kde.kdenlive.scripting", "/kdenlive",
                   "newProjectFormat", str(w), str(h), str(fps)])


# --- Kdenlive: add a new sequence to the current project -------------------- #
# Sequences inherit the project profile, so this takes optional track counts only.
_RE_NEW_SEQ = re.compile(
    r"^(?:hey\s+)?(?:new|create|add|make|insert)\s+(?:a\s+)?(?:new\s+)?"
    r"(?:(?:kdenlive|[kc]a?den\s?live)\s+)?sequence\b",
    re.IGNORECASE)


def try_kdenlive_new_sequence(text: str) -> Optional[Intent]:
    low = (text or "").lower().strip()
    if not _RE_NEW_SEQ.match(low):
        return None
    mv = re.search(r"(\d+)\s*video\s*track", low)
    ma = re.search(r"(\d+)\s*audio\s*track", low)
    v = int(mv.group(1)) if mv else -1
    a = int(ma.group(1)) if ma else -1
    extra = ""
    if mv or ma:
        extra = f" with {v if v > 0 else 'default'} video and {a if a > 0 else 'default'} audio tracks"
    return Intent(f"Added a new sequence{extra}.",
                  ["qdbus6", "org.kde.kdenlive.scripting", "/kdenlive",
                   "newSequence", str(v), str(a)])


# --- OBS: scene switch + overlay toggles ------------------------------------ #
# Deterministic so "switch scene to cintiq" is near-instant AND actually fires
# (obs-state) — instead of going through the chat brain, which only TALKED about
# switching (and took ~30s). Mirrors apps/obs.conf, but here as a no-LLM intent.
#
# Firing rule (scenes): a scene alias appears AND, after removing the alias, the
# only words left are command filler ("switch (scene) to", "go to", "the … scene").
# So a bare/clear command fires ("cintiq", "go to main", "the intro scene") but
# the same word buried in chat does not ("I love the intro we made" keeps "i love
# we made"). That residual test is what keeps the generic word "main" safe.
_OBS_SCENES = [
    # (scene, spoken-name, alias regex). Specific first; generic "main" last.
    # ASR mishears "Cintiq" as Sintiq/Sintique/Centique/Syntique/Cintiqu, etc.
    # [cs] start, any front vowel, "nt", optional vowel, "q", optional "u"/"ue" tail.
    ("cintiq", "the Cintiq scene", r"[cs][iyae]nt[iyae]?q(?:ue|u)?"),
    ("fullscreencamera", "full screen camera",
     r"full\s*screen\s*cam(?:era)?|full\s*cam(?:era)?|face\s*cam(?:era)?"),
    ("intro", "the intro", r"intro"),
    ("outro", "the outro", r"outro"),
    ("main", "the main scene", r"main"),
]
# Words allowed to surround a scene name in a switch command (stripped before the
# "is anything left?" test). Anything outside this set means it's not a command.
_OBS_FILLER = {
    "hey", "ok", "okay", "so", "please", "can", "you", "could", "would", "the",
    "a", "an", "to", "my", "now", "current", "currently", "let", "lets", "us",
    "go", "going", "gonna", "switch", "switching", "change", "changing", "cut",
    "jump", "flip", "take", "bring", "put", "move", "back", "over", "into", "in",
    "on", "onto", "scene", "scenes", "just", "up", "view", "show",
    # scene-trigger verbs Trickery also uses ("roll the outro", "play the intro")
    "roll", "play", "playing", "open", "opening", "load", "pull", "swap",
    "queue", "cue", "fire", "throw", "head", "start", "begin", "run",
}


def try_obs_scene(text: str) -> Optional[Intent]:
    """"switch scene to cintiq" / "go to main" / "full screen camera" -> switch
    the OBS scene via obs-state (over obs-websocket). Returns None unless the
    utterance is essentially just a scene-switch command (see firing rule above)."""
    low = _norm(text)
    if not low:
        return None
    for scene, pretty, alias in _OBS_SCENES:
        m = re.search(rf"\b(?:{alias})\b", low)
        if not m:
            continue
        residual = (low[:m.start()] + " " + low[m.end():]).split()
        if all(tok in _OBS_FILLER for tok in residual):
            return Intent(f"Switching to {pretty}.", ["obs-state", scene])
    return None


# Overlay message toggles -> obs-toggle <source> <on|off>. Requires BOTH a clear
# on/off verb AND a known overlay source, so ordinary chat can't trip it.
_OBS_OVERLAYS = [
    ("brb", "be-right-back message", r"brb|be\s*right\s*back"),
    ("dinner", "dinner message", r"dinner(?:\s*break)?"),
    ("end", "end screen", r"end\s*screen|ending"),
    ("afk", "AFK message", r"afk|away\s*from\s*keyboard"),
]
_RE_OBS_OFF = re.compile(r"\b(?:off|hide|disable|remove|clear|dismiss|take\s*down)\b",
                         re.IGNORECASE)
_RE_OBS_ON = re.compile(r"\b(?:on|show|enable|display|put\s*up|bring\s*up)\b",
                        re.IGNORECASE)
# Commands are SHORT. A long utterance that happens to contain a toggle phrase is
# Trickery thinking out loud / talking to chat with the key held (or the passive
# listener overhearing him) — not a command. Real commands in the logs top out
# around 9 words ("Disable my, uh, or enable the camera background."); a 200-word
# ramble mentioning "camera background" used to fire with an arbitrary direction.
_MAX_CMD_WORDS = 12


def _too_long(low: str) -> bool:
    return len(low.split()) > _MAX_CMD_WORDS


def _direction_near(low: str, span: tuple[int, int]) -> Optional[str]:
    """on/off decided by whichever direction word sits CLOSEST to the thing being
    toggled — not by which regex is tried first. Fixes self-corrections like
    "disable my, uh, or enable the camera background" (means ON, used to give OFF)
    and stray hits ("why can I not click ON her ... show camera background")."""
    best: Optional[tuple[int, str]] = None
    for st, rx in (("off", _RE_OBS_OFF), ("on", _RE_OBS_ON)):
        for m in rx.finditer(low):
            if m.start() < span[1] and m.end() > span[0]:
                dist = 0                                  # overlapping the phrase
            else:
                dist = min(abs(m.start() - span[1]), abs(span[0] - m.end()))
            if best is None or dist < best[0]:
                best = (dist, st)
    return best[1] if best else None


def try_obs_overlay(text: str) -> Optional[Intent]:
    low = _norm(text)
    if not low or _too_long(low):
        return None
    for src, pretty, alias in _OBS_OVERLAYS:
        m = re.search(rf"\b(?:{alias})\b", low)
        if not m:
            continue
        st = _direction_near(low, m.span())
        if st is None:
            return None  # no clear toggle direction -> not an overlay command
        verb = "Showing" if st == "on" else "Hiding"
        return Intent(f"{verb} the {pretty}.", ["obs-toggle", src, st])
    return None


def try_obs_camera_bg(text: str) -> Optional[Intent]:
    """"show camera background" / "hide the camera background" -> swap the two
    camera sources via obs-camera-bg (CameraNoAlpha = full feed w/ background vs
    MeCamera = alpha cutout). Needs the phrase "camera background" AND a clear
    show/hide direction, so ordinary chat can't trip it."""
    low = _norm(text)
    if not low or _too_long(low):
        return None
    m = re.search(r"\b(?:camera|cam)\s+back\s*ground\b", low)
    if not m:
        return None
    st = _direction_near(low, m.span())
    if st == "off":
        return Intent("Hiding the camera background.", ["obs-camera-bg", "off"])
    if st == "on":
        return Intent("Showing the camera background.", ["obs-camera-bg", "on"])
    return None


# --- SystemFlow window management: focus / Z-layer / send-to-monitor --------- #
# All actions are existing KWin global shortcuts in the SystemFlow script, fired
# via `qdbus6 ... invokeShortcut <name>` (same path as the kdenlive-cut intent).
# Monitors 1-4 = Meta+Alt+1-4 (TrikWindowSendMonitor1..4): 1 main, 2 Cintiq,
# 3 lower-right, 4 upper-right.
#
# Safety: each parser requires its trigger token (a direction / "monitor N" /
# "layer") AND that the WHOLE utterance is window-command vocabulary — any other
# content word means it's chat, not a command. (Same philosophy as try_obs_scene.)
_SF_VOCAB = {
    "hey", "ok", "okay", "so", "well", "please", "can", "you", "could", "would",
    "lets", "let", "us", "the", "a", "an", "to", "on", "onto", "into", "over",
    "my", "this", "that", "it", "now", "just", "of", "and", "with", "here",
    # verbs
    "focus", "move", "send", "put", "push", "switch", "go", "navigate", "jump",
    "bring", "throw", "kick", "shove", "toss", "bump", "carry", "drag", "raise",
    "lower", "drop", "view", "look", "take",
    # nouns / targets
    "window", "windows", "screen", "monitor", "monitors", "display", "layer",
    "layers", "z", "level", "depth", "other", "all", "next", "one", "current",
    "active", "number", "cursor", "keyboard", "across",
    # directions
    "left", "right", "up", "down", "above", "below", "forward", "back", "behind",
    "front", "top", "bottom", "higher", "deeper", "under",
    # number words (monitor 1-4); bare digits handled separately
    "two", "three", "four",
}
_NUMWORD = {"one": "1", "two": "2", "three": "3", "four": "4",
            "1": "1", "2": "2", "3": "3", "4": "4"}
_DIR_WORDS = {"left", "right", "up", "down", "above", "below"}


def _kw(action: str) -> list[str]:
    return ["qdbus6", "org.kde.kglobalaccel", "/component/kwin", "invokeShortcut", action]


def _sf_only_vocab(low: str) -> bool:
    return bool(low) and all(t in _SF_VOCAB or t.isdigit() for t in low.split())


_RE_SF_MON = re.compile(r"\b(?:monitor|screen|display)\s*(?:number\s*)?(one|two|three|four|[1-4])\b",
                        re.IGNORECASE)
_RE_SF_MOVE = re.compile(r"\b(?:send|move|put|push|throw|shove|toss|bump|bring|drag|kick)\b",
                         re.IGNORECASE)


def try_sf_monitor(text: str) -> Optional[Intent]:
    """"send this window to monitor 2" / "move it to screen 3" -> TrikWindowSendMonitor<N>."""
    low = _norm(text)
    if not _sf_only_vocab(low):
        return None
    m = _RE_SF_MON.search(low)
    if not m or not _RE_SF_MOVE.search(low):
        return None
    n = _NUMWORD[m.group(1).lower()]
    return Intent(f"Sending the window to monitor {n}.", _kw(f"TrikWindowSendMonitor{n}"))


_RE_SF_LAYER = re.compile(r"\b(?:layers?|z\s*layers?|z\s*levels?|depth)\b", re.IGNORECASE)
_RE_SF_UP = re.compile(r"\b(?:up|raise|higher|forward|front|top|above)\b", re.IGNORECASE)
_RE_SF_DOWN = re.compile(r"\b(?:down|lower|deeper|behind|back|bottom|below|under)\b", re.IGNORECASE)
_RE_SF_NAV = re.compile(r"\b(?:go|focus|navigate|switch|jump|view|look)\b", re.IGNORECASE)
_RE_SF_PUSH = re.compile(r"\b(?:send|push|move|put|bump|carry|bring|raise|lower|drop|kick|throw|shove|toss)\b",
                         re.IGNORECASE)
# A bare layer command is NAVIGATION; the FOCUSED window only moves when the
# utterance explicitly names a window reference ("move THIS up a layer", "send the
# window down", "push it up"). Moving a window BY NAME is try_sf_layer_named below.
_RE_SF_SELF = re.compile(r"\b(?:this|it|current|active|focused|window|windows)\b", re.IGNORECASE)


def try_sf_layer(text: str) -> Optional[Intent]:
    """Bare Z-layer command. "go up a layer" / "move up one z layer" -> just NAVIGATE
    (TrikLayerFocus<Up|Down>). "move this up a layer" / "send the window down" / "push
    it up" -> move the FOCUSED window (TrikLayerPush<Up|Down>). Bare = navigate is the
    default — moving a window requires an explicit window reference (or a name)."""
    low = _norm(text)
    if not _sf_only_vocab(low) or not _RE_SF_LAYER.search(low):
        return None
    if _RE_SF_UP.search(low):
        d = "Up"
    elif _RE_SF_DOWN.search(low):
        d = "Down"
    else:
        return None
    if _RE_SF_PUSH.search(low) and _RE_SF_SELF.search(low):
        return Intent(f"Sending the window {d.lower()} a layer.", _kw(f"TrikLayerPush{d}"))
    return Intent(f"Going {d.lower()} a layer.", _kw(f"TrikLayerFocus{d}"))


# "move Krita up one z layer" / "send the StreamToons editor down two layers" -> move
# that NAMED window's Z-layer by an offset. Implemented as focus-by-name then the
# existing layer-push shortcut (sf-move-layer), so no new KWin/daemon code. Checked
# BEFORE try_sf_layer; a name that's pure command vocab (no real window) falls through
# to the bare navigation above.
_RE_SF_NAMED_LAYER = re.compile(
    r"^(?:hey\s+)?"
    r"(?:move|send|push|put|bump|bring|shift|raise|lower|drop|shove|toss|kick|carry|throw|take)\s+"
    r"(?P<name>.+?)\s+"
    r"(?P<dir>up|down|higher|lower|forward|back|above|below)\b\s*"
    r"(?:by\s+)?(?:a|an|the)?\s*"
    r"(?P<count>one|two|three|four|five|six|seven|eight|nine|ten|\d+)?\s*"
    r"(?:z\s*)?(?:layer|layers|level|levels|depth)\b",
    re.IGNORECASE)
_DIR_TO_UPDOWN = {"up": "up", "higher": "up", "above": "up", "forward": "up",
                  "down": "down", "lower": "down", "below": "down", "back": "down"}
_COUNTWORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
              "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def try_sf_layer_named(text: str) -> Optional[Intent]:
    low = _norm(text)
    if not _RE_SF_LAYER.search(low):
        return None
    m = _RE_SF_NAMED_LAYER.match(low)
    if not m:
        return None
    name_toks = [t for t in m.group("name").split() if t]
    # Need a real window name — if it's all command vocab ("move it up a layer"), let
    # the bare navigation/push path handle it instead.
    if not name_toks or all(t in _SF_VOCAB for t in name_toks):
        return None
    if name_toks[0] in _DIR_WORDS:
        return None
    direction = _DIR_TO_UPDOWN.get(m.group("dir").lower())
    if not direction:
        return None
    cw = m.group("count")
    count = (max(1, min(10, int(cw))) if (cw and cw.isdigit())
             else _COUNTWORD.get((cw or "").lower(), 1))
    needle = _focus_needle(name_toks)
    if not needle:
        return None
    plural = "s" if count > 1 else ""
    return Intent(f"Moving {needle} {direction} {count} layer{plural}.",
                  ["sf-move-layer", needle, direction, str(count)])


_RE_SF_FOCUS_DIR = re.compile(r"\b(left|right|up|above|down|below)\b", re.IGNORECASE)


def try_sf_focus(text: str) -> Optional[Intent]:
    """"focus left/right/up/down" (current monitor) / "focus left across monitors"
    -> TrikWindowFocus[Across]<Dir>. Requires the word "focus" + a direction."""
    low = _norm(text)
    if "focus" not in low.split() or not _sf_only_vocab(low):
        return None
    m = _RE_SF_FOCUS_DIR.search(low)
    if not m:
        return None
    word = m.group(1).lower()
    d = {"left": "Left", "right": "Right", "up": "Up", "above": "Up",
         "down": "Down", "below": "Down"}[word]
    across = bool(re.search(r"\bacross\b|\ball\s+monitors\b|\bother\s+monitor", low))
    action = f"TrikWindowFocus{'Across' if across else ''}{d}"
    return Intent(f"Focusing {word}{' across monitors' if across else ''}.", _kw(action))


# "focus Krita" / "focus on the StreamToons editor window" -> activate that app's
# window by name (sf-focus-app -> SystemFlow, which substring/token-matches the
# needle against window class+caption). Checked AFTER try_sf_focus, so a direction
# word is already handled as directional focus. A one-word name is always allowed;
# a MULTI-word name is only treated as a window-focus when they explicitly said
# "... window/app" (else it's likely chat, e.g. "focus on the task at hand").
_RE_SF_FOCUS_APP = re.compile(
    r"^(?:hey\s+)?focus(?:\s+on)?(?:\s+the)?\s+(.+?)"
    r"(?:\s+(?:window|windows|app|application))?$",
    re.IGNORECASE)
# Generic trailing descriptors that aren't part of an app's window class/caption.
# "editor" is intentionally KEPT — it IS part of names like "StreamToons-Editor",
# and the KWin matcher tokenizes, so it usefully disambiguates the target window.
_FOCUS_DROP = {"window", "windows", "app", "application", "the", "a", "an"}


def _focus_fix_tokens(toks: list[str]) -> list[str]:
    """Targeted ASR mishearing fix, only in window-focus context: the
    "StreamToons-Editor" window is regularly transcribed "stream tunes editor".
    Gated on "editor" so it never touches the real "StreamTunes" (music) portal."""
    if "editor" in toks:
        toks = ["streamtoons" if t in ("streamtunes", "streamtune") else t for t in toks]
    return toks


def _focus_needle(toks: list[str]) -> str:
    """Build the window-match needle from spoken tokens: apply the mishearing fix,
    drop generic descriptors. Shared by focus and named layer-move intents."""
    toks = _focus_fix_tokens(toks)
    core = [t for t in toks if t not in _FOCUS_DROP] or toks
    return " ".join(core)


def try_sf_focus_app(text: str) -> Optional[Intent]:
    low = _norm(text)
    m = _RE_SF_FOCUS_APP.match(low)
    if not m:
        return None
    toks = [t for t in m.group(1).split() if t]
    if not toks:
        return None
    if toks[0] in _DIR_WORDS:             # "focus up/left/…" is directional (handled earlier)
        return None
    if all(t in _SF_VOCAB for t in toks):  # "focus here"/"focus now" -> not an app
        return None
    explicit = bool(re.search(r"\b(?:window|windows|app|application)\b", low))
    if len(toks) > 1 and not explicit:    # multi-word without "window/app" = probably chat
        return None
    needle = _focus_needle(toks)
    return Intent(f"Focusing {needle}.", ["sf-focus-app", needle])


# "save workspace" / "save as workspace two" / "load workspace" / "restore workspace 3"
# -> sf-workspace save|load [name] (SystemFlow checkpoint of every open window: monitor,
# Z-layer, tile). No number = the "default" workspace. The WHOLE utterance must be the
# command (fillers allowed) — "load workspace" never fires from inside a sentence.
_RE_SF_WORKSPACE = re.compile(
    r"^(?:(?:hey|ok|okay|please|can you|could you|would you)\s+)*"
    r"(?P<verb>save|load|restore)\s+"
    r"(?:(?:as|the|my|this|our|current|a)\s+)*"
    r"(?P<default>default\s+)?"
    r"work\s?space"
    r"(?:\s+(?:as\s+)?(?:number\s+)?(?P<num>\w+))?$")
# ASR spells spoken numbers as homophones ("load workspace to"); "default" names it outright.
_WORKSPACE_NUM = {"one": "1", "won": "1", "two": "2", "to": "2", "too": "2", "three": "3",
                  "four": "4", "for": "4", "five": "5", "six": "6", "seven": "7",
                  "eight": "8", "nine": "9", "ten": "10", "default": "default"}


def try_sf_workspace(text: str) -> Optional[Intent]:
    m = _RE_SF_WORKSPACE.match(re.sub(r"\s+please$", "", _norm(text)))
    if not m:
        return None
    num = m.group("num")
    if num is None:
        name = "default"
    elif num.isdigit():
        name = str(int(num))
    else:
        name = _WORKSPACE_NUM.get(num)
        if name is None:              # "save workspace layout" etc. — not a numbered workspace
            return None
    op = "save" if m.group("verb") == "save" else "load"
    label = "the default workspace" if name == "default" else f"workspace {name}"
    say = f"Saving {label}." if op == "save" else f"Loading {label}."
    return Intent(say, ["sf-workspace", op, name])


# --- OBS clip: "clip that" -> save the replay buffer + speak confirmation ----- #
# The MONEY intent: a missed "clip" = a lost highlight. Kept deterministic (no
# LLM) so it's instant and never gets talked-about-instead-of-done by the chat
# brain (which otherwise hallucinates "the player isn't running"). Grammar is
# forgiving of ASR jitter: the utterance must START with a clip trigger, and
# after it only throwaway filler may remain — so "clip that's the same" and
# "clip that real quick dude" fire, but "clip the audio in Kdenlive" (real
# content words left over) falls through to chat.
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


def looks_like_clip(text: str) -> bool:
    """Pure check (no OBS call): does `text` open with a clip trigger? Lets the
    done-summarizer stand aside for "clip that title ion two summary"."""
    raw = (text or "").strip()
    tm = _RE_CLIP_TITLE.search(raw)
    body = raw[:tm.start()] if tm else raw
    return bool(_RE_CLIP_HEAD.match(_norm(body)))


def try_clip(text: str) -> Optional[Intent]:
    raw = (text or "").strip()
    # Peel off an optional inline title before checking the clip grammar.
    title = ""
    body = raw
    tm = _RE_CLIP_TITLE.search(raw)
    if tm:
        body = raw[:tm.start()]
        title = raw[tm.end():].strip(" \t,.;:!?-\"'").strip()
    norm = _norm(body)
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


# --- Stream chat: fire a chat-bot / avatar command by voice ------------------ #
# "stream command lurk" -> POST !lurk into Twitch chat, where Nightbot (or the
# stream-avatar command handler) sees it and responds. The prefix word avoids
# saying "nightbot" (Trickery calls it "Nully" / "stream command"). Everything
# after the trigger is the command + its optional args ("stream command shoutout
# someone" -> "!shoutout someone").
_HUB_SAY_URL = "http://localhost:3737/say"          # nul-chat-hub outbound API
_NIGHTBOT_PREFIX = "!"                               # matches hub commandPrefix
_RE_STREAM_CMD = re.compile(
    r"^(?:stream|nully|nightbot|chat)\s+commands?\b[\s:,.-]*(.+)$",
    re.IGNORECASE)


def try_stream_command(text: str) -> Optional[Intent]:
    """"stream command <word> [args]" -> post "!<word> [args]" to Twitch chat so
    Nightbot / the avatar commands pick it up. Deterministic: the trigger must lead
    the utterance and the message is built from a fixed template (only the spoken
    command/args are interpolated, sent as a JSON body via argv — no shell, nothing
    to inject)."""
    m = _RE_STREAM_CMD.match((text or "").strip())
    if not m:
        return None
    # Don't hijack a pane literally named "Stream Command(s)" (Trickery's
    # VtuberCommands pane): if the utterance leads with a registered pane name,
    # it's a route, not a chat command. (In chat mode try_route already caught
    # it; this also covers command mode, where try_route doesn't run.)
    try:
        from . import dispatch
        if dispatch.leading_pane_match(text):
            return None
    except Exception:  # noqa: BLE001 — a tmux-less box must not break the command
        pass
    remainder = m.group(1).strip().strip(".!?,").strip()
    if not remainder:
        return None  # "stream command" with nothing after it -> not actionable
    words = remainder.split()
    cmd = words[0].lower().lstrip("!").strip(".,!?")
    if not cmd:
        return None
    args = words[1:]
    message = _NIGHTBOT_PREFIX + cmd
    if args:
        message += " " + " ".join(args)
    body = json.dumps({"text": message, "platform": "twitch"})
    argv = ["curl", "-s", "-m", "5", "-X", "POST", _HUB_SAY_URL,
            "-H", "Content-Type: application/json", "-d", body]
    return Intent(f"Running the {cmd} command in chat.", argv)


# --- MikuVoice: switch the assistant's OWN voice (FX profile) by voice -------- #
# "switch to the miku voice", "use the aurinex voice", "change your voice to
# miku", "sound like aurinex" -> `mikuvoice-profile use <name>`. Runs in BOTH chat
# and command mode (intents.match is checked in each), so Trickery can either
# command OR just ask. Deterministic: the target is validated against the profiles
# that actually exist on disk — the model/speech never picks a free-form command.
#
# NOTE: Hermes' own ~/.hermes/config.yaml tts (edge/en-US-AriaNeural) is DEAD in
# this pipeline — we synthesize via Kokoro + the MikuVoice FX chain, and THIS is
# what actually changes the voice. Handling it here (before the brain) is also why
# asking Hermes no longer gets the wrong "I'm on AriaNeural" answer.
_MIKU_PROFILES_DIR = os.path.expanduser("~/.local/share/miku-voice/profiles")
_MIKU_ACTIVE_FILE = os.path.expanduser("~/.local/share/miku-voice/active")
# A switch VERB is required so ordinary chat ("I love the miku voice") doesn't
# flip it. "tell"/"ask" are deliberately absent so "tell Miku Claude ..." (a pane
# route) can't match. \bset\b is fine — paired below with a voice/name context.
_RE_VOICE_CUE = re.compile(
    r"\b(switch|change|swap|set|use|put on|go(?:ing)?\s+(?:back\s+)?to|back to|"
    r"become|turn into|talk|speak|sound|make)\b",
    re.IGNORECASE)
# ASR mishears — a few forgiving aliases mapping to canonical profile stems.
_VOICE_ALIASES = {
    "miku": "miku", "meeku": "miku", "mika": "miku", "mikku": "miku",
    "aurinex": "aurinex", "orinex": "aurinex", "aureenex": "aurinex",
    "aura nex": "aurinex", "aurinox": "aurinex", "aureenix": "aurinex",
    # "or next" / "ornex" are how ASR usually renders "Aurinex" (same variants
    # the persona file lists). Without them a real "switch to Aurinex voice" fell
    # through to the nameless-toggle branch and switched to MIKU instead — the
    # exact opposite of what was asked. Safe: try_voice_switch still requires a
    # switch cue + voice context + the short-command length cap.
    "or next": "aurinex", "ornex": "aurinex", "or nex": "aurinex",
    "aura next": "aurinex", "orenex": "aurinex",
}


def _miku_profiles() -> list[str]:
    try:
        return sorted(f[:-5] for f in os.listdir(_MIKU_PROFILES_DIR)
                      if f.endswith(".conf"))
    except OSError:
        return []


def _miku_active() -> Optional[str]:
    try:
        with open(_MIKU_ACTIVE_FILE) as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _wait_for_sink(name: str, timeout: float = 3.5) -> None:
    """After the profile switch restarts mikuvoice.service, give the MikuVoice
    sink a moment to re-register so the spoken confirmation isn't dropped."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            out = subprocess.run(["pactl", "list", "short", "sinks"],
                                 capture_output=True, text=True, timeout=3).stdout
        except Exception:  # noqa: BLE001
            return
        if any(line.split("\t")[1:2] == [name] for line in out.splitlines()):
            return
        time.sleep(0.15)


def _spoken_name(stem: str) -> str:
    return stem[:1].upper() + stem[1:]


def try_voice_switch(text: str) -> Optional[Intent]:
    low = (text or "").lower()
    if not low.strip() or not _RE_VOICE_CUE.search(low):
        return None
    # Length guard — a voice switch is a short command. Without this, a long
    # ramble could satisfy the pieces separately and silently flip his voice
    # mid-sentence: one real log line ("...my VOICE agent... for OTHER people...
    # MAKE stuff...") hit the cue + about_voice + the nameless-toggle branch and
    # would have switched Aurinex to Miku while he was talking to chat.
    if _too_long(_norm(low)):
        return None
    about_voice = bool(re.search(
        r"\bvoice\b|\b(?:sounds?|talk|speak)\s+like\b", low))
    profiles = _miku_profiles()
    if not profiles:
        return None
    # Which known profile did Trickery name? (canonical stems + ASR aliases)
    target = None
    for stem in profiles:
        if re.search(rf"\b{re.escape(stem)}\b", low):
            target = stem
            break
    if target is None:
        for alias, stem in _VOICE_ALIASES.items():
            if stem in profiles and re.search(rf"\b{re.escape(alias)}\b", low):
                target = stem
                break
    active = _miku_active()
    # Toggle: "switch your voice" / "use the other voice" with no name, when there
    # are exactly two profiles — flip to the one that isn't active.
    if target is None:
        if (about_voice and len(profiles) == 2 and active in profiles
                and re.search(r"\b(other|different|switch|swap|change|toggle|back)\b", low)):
            target = next(p for p in profiles if p != active)
        else:
            return None  # ambiguous — let the brain field it
    # A bare profile name needs the "voice" context so plain chat can't flip it;
    # a name + an explicit switch verb ("switch to miku") is enough on its own.
    if not about_voice and not re.search(
            r"\b(switch|change|swap|use|become|turn into|go(?:ing)?\s+to|back to)\b", low):
        return None
    name = _spoken_name(target)
    if active == target:
        return Intent(f"I'm already using the {name} voice.", [])
    try:
        r = subprocess.run(["mikuvoice-profile", "use", target],
                           capture_output=True, text=True, timeout=15)
        ok = r.returncode == 0
    except Exception:  # noqa: BLE001 — never crash the mic flow over the switch
        ok = False
    if not ok:
        return Intent(f"I couldn't switch to the {name} voice.", [])
    _wait_for_sink("MikuVoice")
    # Already executed (empty argv); the confirmation is spoken in the NEW voice.
    return Intent(f"Switched to the {name} voice.", [])


# --- narration: "narrate" reads the focused page / selection aloud ---------- #
# ASR renders "narrate" a few ways; "no rate" is the usual slip.
_NARRATE_WORD = r"(?:n[ae]rr?ate\w*|no\s+rate)"
_RE_NARRATE = re.compile(
    r"^(?:hey\s+|ok(?:ay)?\s+)?(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
    rf"(?:(?P<n>{_NARRATE_WORD})|(?P<r>read))\b",
    re.IGNORECASE)
# Bare "read" is Trickery's existing clipboard trigger, so the read-verb form only
# claims the utterance when it names something page-like to read.
# Deliberately does NOT accept a bare "this"/"it" — "read it" is far more likely
# to mean the clipboard than the page, so it stays with the clipboard trigger.
_RE_READ_TARGET = re.compile(
    r"\b(?:page|article|post|site|screen|tab|selection|highlight\w*|"
    r"out\s+loud|aloud)\b", re.IGNORECASE)
_RE_NARR_SUMMARY = re.compile(
    r"\b(?:summar\w+|sum\s+(?:it\s+)?up|gist|tl;?dr|brief\w*|condense|"
    r"overview|recap)\b", re.IGNORECASE)
_RE_NARR_SELECTION = re.compile(
    r"\b(?:selection|selected|highlight\w*)\b", re.IGNORECASE)
_RE_NARR_WHOLE = re.compile(
    r"\b(?:whole|entire|full|all\s+of)\b", re.IGNORECASE)
# "the section about X" / "the part on X" / "the X section"
_RE_NARR_SECTION = re.compile(
    r"\b(?:section|part|chapter|bit|paragraph|heading)\s+"
    r"(?:about|on|called|titled|named|regarding|for)\s+(?P<a>.+)$",
    re.IGNORECASE)
_RE_NARR_SECTION_2 = re.compile(
    r"\bthe\s+(?P<a>.+?)\s+(?:section|part|chapter|heading)\b", re.IGNORECASE)
_RE_NARR_ABOUT = re.compile(
    r"\b(?:about|on|regarding|covering)\s+(?P<a>.+)$", re.IGNORECASE)
_NARR_MAX_WORDS = 18
# Trailing politeness that isn't part of a section name.
_RE_NARR_TAIL = re.compile(
    r"\b(?:to|for)\s+me\b|\b(?:out\s+loud|aloud|please|now|for\s+me)\b",
    re.IGNORECASE)


def try_narrate(text: str) -> Optional[Intent]:
    """"narrate" -> read the focused page (or what's highlighted) aloud.

    Priority is the live in-page selection, then the article body; a named
    section slices the page by heading; "narrate a summary of this" routes
    through the brain instead of reading verbatim. Work happens inline (extract
    + spawn the detached narrator) and the returned .say is EMPTY on success —
    the narration is the reply, and a spoken ack would talk over its first chunk.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    m = _RE_NARRATE.match(raw)
    if not m:
        return None
    rest = raw[m.end():].strip()
    if m.group("r") and not _RE_READ_TARGET.search(rest):
        return None                    # bare "read …" stays the clipboard trigger
    # A narration command is short. Without this a long ramble that happens to
    # open with "read this…" could hijack the turn.
    if len(_norm(raw).split()) > _NARR_MAX_WORDS:
        return None
    rest = _RE_NARR_TAIL.sub(" ", rest)
    rest = re.sub(r"^[\s,\-–—:]+|[\s,\.\-–—:!?]+$", "", rest)
    summarize = bool(_RE_NARR_SUMMARY.search(rest))
    prefer = "auto"
    if _RE_NARR_SELECTION.search(rest):
        prefer = "selection"
    elif _RE_NARR_WHOLE.search(rest):
        prefer = "page"
    focus = ""
    for pat in (_RE_NARR_SECTION, _RE_NARR_SECTION_2, _RE_NARR_ABOUT):
        got = pat.search(rest)
        if got:
            focus = got.group("a").strip(" .,:;-")
            break
    # "narrate a summary of this page" — "this page" is the target, not a section.
    if focus and _norm(focus) in ("this", "that", "it", "this page", "the page",
                                 "this article", "the article", "this thing"):
        focus = ""
    from . import narrate                        # local: keeps import order simple
    say = narrate.handle(prefer=prefer, focus=focus, summarize=summarize)
    return Intent(say, [])


_RE_NARR_PAUSE = re.compile(r"^(?:pause|hold\s+(?:on|up)|wait)\b", re.IGNORECASE)
_RE_NARR_RESUME = re.compile(
    r"^(?:resume|continue|unpause|carry\s+on|go\s+on|go\s+ahead|"
    r"keep\s+(?:going|reading))\b", re.IGNORECASE)
_RE_NARR_STOP = re.compile(
    r"^(?:stop|cancel|quiet|shut\s+up|enough|that\s?s\s+enough|"
    r"never\s?mind|forget\s+it)\b", re.IGNORECASE)
_NARR_CTL_MAX_WORDS = 6


def try_narration_control(text: str) -> Optional[Intent]:
    """pause / resume / stop — but ONLY while a narration is running.

    Gated on narrate.is_active() so these very common words can never hijack
    normal chat when nothing is being read. Note the hotkey press itself already
    auto-paused the narration (so the mic doesn't hear her), which is what makes
    "resume" the meaningful half of this pair.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    from . import narrate
    if not narrate.is_active():
        return None
    low = _norm(raw)
    if not low or len(low.split()) > _NARR_CTL_MAX_WORDS:
        return None
    # "stop the stream" is a different machine entirely — never claim it here.
    if re.search(r"\b(?:stream|streaming|recording|music|obs)\b", low):
        return None
    if _RE_NARR_PAUSE.match(low):
        narrate.pause()
        return Intent("Paused.", [])
    if _RE_NARR_RESUME.match(low):
        narrate.resume()
        return Intent("", [])          # she just carries on reading
    if _RE_NARR_STOP.match(low):
        narrate.stop()
        return Intent("Stopped.", [])
    return None


# Registered parsers, tried in order. Order matters: resize (needs a resize verb +
# existing-doc words) is checked before new (needs a create verb) — they're
# otherwise mutually exclusive. try_clip is near the front — a unique bare word.
# Narration control leads: it is inert unless a narration is actually running,
# and while one is, "pause"/"stop" must beat every other grammar.
# The OBS/stream intents are cheap and specific (they only fire on a clear
# scene/overlay/chat command).
_PARSERS: list[Parser] = [
    try_narration_control,
    try_narrate,
    try_voice_switch,
    try_clip,
    try_stream_command,
    try_obs_camera_bg,
    try_obs_scene,
    try_obs_overlay,
    try_sf_workspace,
    try_sf_monitor,
    try_sf_layer_named,
    try_sf_layer,
    try_sf_focus,
    try_sf_focus_app,
    try_krita_resize,
    try_krita_new,
    try_kdenlive_cut,
    try_kdenlive_new_project,
    try_kdenlive_new_sequence,
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
    """Run an Intent's argv (detached). Returns True if it launched.

    Some intents (try_clip, try_voice_switch) already did their work inline and
    carry no argv — the caller still speaks intent.say, so there's nothing left
    to launch."""
    if not intent.argv:
        return False
    try:
        # argv form (no shell): the template is fixed above and only validated
        # values are interpolated, so there is nothing to inject.
        subprocess.Popen(intent.argv, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return True
    except OSError:
        return False
