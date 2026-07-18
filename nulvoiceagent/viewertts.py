"""Viewer-paid TTS — speak a chat viewer's message (a bits cheer or a channel-point
redeem) aloud through the Miku voice, one at a time, after moderation filtering.

Invoked once per paying event by the nul-chat-hub daemon:

    voiceagent viewer-say --user <name> --source bits|points [--bits N] "<raw text>"

This is deliberately a SEPARATE path from the chat brain: the viewer's text is
spoken verbatim and never reaches the agent's persistent memory. Design:

  - Filter  : strip URLs + cheermote tokens, collapse whitespace, hard length cap,
              drop the whole message if it hits the blocklist. Empty after
              cleaning -> silent no-op.
  - Serialize: an flock (config.VIEWER_TTS_LOCK) held across synth + playback, so
              simultaneous cheers queue instead of overlapping; and a bounded wait
              for any in-progress Trickery turn so viewer TTS never talks over him.
  - Speak    : tts.speak() with a spoken "From <user>:" attribution, optionally in
              a distinct viewer voice/pitch.
"""
from __future__ import annotations
import fcntl
import re
import time

from . import config, tts

# A standalone cheermote token: a name immediately followed by its bit amount,
# e.g. "Cheer100", "PogChamp50", "uni500". These litter cheer messages and read as
# gibberish, so drop any word that is <letters><digits> with nothing else.
_CHEERMOTE = re.compile(r"^[A-Za-z]+\d+$")
# URLs / bare domains — stripped so viewers can't have arbitrary links read aloud.
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_DOMAIN = re.compile(
    r"\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|io|gg|tv|xyz|link|live|me|co|app|dev)\b\S*",
    re.IGNORECASE,
)
# Attribution name: keep it human but harmless (no control chars / markup).
_NAME_OK = re.compile(r"[^0-9A-Za-z _\-]")


def _load_blocklist() -> list[str]:
    try:
        lines = config.VIEWER_TTS_BLOCKLIST.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            out.append(ln.lower())
    return out


def _blocked(text: str) -> str | None:
    """Return the blocklist phrase that matched (case-insensitive substring), or
    None if the text is clean."""
    low = text.lower()
    for phrase in _load_blocklist():
        if phrase in low:
            return phrase
    return None


def clean(raw: str) -> str:
    """Moderate + normalize a viewer message for speaking. Returns "" if there's
    nothing safe/speakable left."""
    text = (raw or "").strip()
    if not text:
        return ""
    text = _URL.sub(" ", text)
    text = _DOMAIN.sub(" ", text)
    # Drop standalone cheermote tokens, keep everything else.
    text = " ".join(w for w in text.split() if not _CHEERMOTE.match(w))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > config.VIEWER_TTS_MAX_CHARS:
        text = text[: config.VIEWER_TTS_MAX_CHARS].rstrip()
    return text


def _clean_name(user: str) -> str:
    name = _NAME_OK.sub("", (user or "").strip())
    return name[:40].strip() or "someone"


def _turn_active() -> bool:
    """True if Trickery has a live recording or reply in flight — mirrors the
    reuse-guarded pidfile checks in __main__ so viewer TTS can yield to him."""
    for pidfile, needpg in ((config.CACHE_DIR / "recording.pid", False),
                            (config.CACHE_DIR / "turn.pgid", True)):
        try:
            pid = int(pidfile.read_text())
        except (OSError, ValueError):
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read()
        except OSError:
            continue  # dead / reused -> not our turn
        if b"nulvoiceagent" in cmd:
            return True
    return False


def _log(line: str) -> None:
    from .__main__ import _log as main_log  # reuse the CLI's stderr+file logger
    main_log(line)


def say(user: str, text: str, source: str = "bits", bits: int | None = None) -> None:
    """Filter + speak one viewer message. Best-effort: any failure is logged, never
    raised (the hub spawns this detached and ignores exit status)."""
    if not config.VIEWER_TTS_ENABLE or config.VIEWER_TTS_OFF_FLAG.exists():
        _log("viewer-tts: disabled")
        return

    cleaned = clean(text)
    if not cleaned:
        _log(f"viewer-tts: empty after filter (user={user!r} source={source})")
        return
    hit = _blocked(cleaned)
    if hit:
        _log(f"viewer-tts: BLOCKED {user!r} on {hit!r}")
        return

    name = _clean_name(user)
    spoken = f"From {name}: {cleaned}"
    tag = f"{source}" + (f"/{bits}" if bits else "")
    _log(f"viewer-tts: {name} ({tag}) -> {cleaned!r}")

    # Serialize against other viewer requests: hold the lock across the whole
    # synth+playback so cheers never overlap. flock blocks until acquired.
    with open(config.VIEWER_TTS_LOCK, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Once we own the slot, wait (bounded) for any live Trickery turn to end so
        # we don't talk over him; then speak regardless so a stuck turn can't wedge
        # the queue forever.
        deadline = time.monotonic() + config.VIEWER_TTS_WAIT_TURN_SECS
        while _turn_active() and time.monotonic() < deadline:
            time.sleep(0.3)
        tts.speak(
            spoken,
            voice=config.VIEWER_TTS_VOICE or None,
            pitch=config.VIEWER_TTS_PITCH or None,
            speed=config.VIEWER_TTS_SPEED or None,
        )
