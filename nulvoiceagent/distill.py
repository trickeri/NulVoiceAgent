"""Offline memory distiller — keep Hermes' user profile current without paying
for it on the voice path.

WHY THIS EXISTS
Hermes has its own auto-memory, and it is enabled, but it can never fire here.
Its "consider writing memory" nudge counts user turns *inside one session*
(memory.nudge_interval, default 10 — agent/turn_context.py). Every voice turn is
a fresh `hermes -z` oneshot: the counter starts at 1, the process exits, the next
turn starts over at 1. It never reaches 10. Nine months of conversation produced
zero self-written notes.

Rather than make every spoken turn carry a memory-review tool call (latency on
the exact path we just sped up, and more quota burn), we distill in batch: read
what was actually said from Hermes' own state.db, ask a model what's durably true
about the user, and rewrite USER.md.

SAFETY
Rewriting a memory file is destructive, so every run: backs up the current file
first, requires the model to return the WHOLE file (preserving what it isn't
changing), validates the result before it lands, and writes a diff you can read.
Any validation failure aborts and leaves the existing file untouched.
"""
from __future__ import annotations
import difflib
import re
import sqlite3
import time
from pathlib import Path

from . import brain, config


# voiceagent appends an action-list hint to every user message before sending it
# to the brain; it's plumbing, not something the user said, so strip it.
_SYSTEM_HINT_RE = re.compile(r"\n*\[SYSTEM\].*\Z", re.DOTALL)
# Entries in a Hermes memory file are separated by this.
_ENTRY_DELIM = "\n§\n"


def _log(line: str) -> None:
    try:
        with open(config.CACHE_DIR / "nulvoiceagent.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')}  distill: {line}\n")
    except OSError:
        pass


def _read_watermark() -> int:
    try:
        return int(config.DISTILL_WATERMARK.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def _write_watermark(msg_id: int) -> None:
    try:
        config.DISTILL_WATERMARK.write_text(str(msg_id), encoding="utf-8")
    except OSError:
        pass


def new_messages(since_id: int) -> tuple[list[dict], int]:
    """Conversation rows newer than `since_id`, plus the highest id seen.

    Read-only, and opened read-only on purpose: Hermes may be mid-write in
    another process and this job must never be the thing that corrupts its DB.
    """
    db = config.HERMES_STATE_DB
    if not db.exists():
        return [], since_id
    uri = f"file:{db}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True, timeout=10)
    except sqlite3.Error:
        return [], since_id
    try:
        rows = con.execute(
            "SELECT id, role, content, timestamp FROM messages "
            "WHERE id > ? AND active = 1 AND content IS NOT NULL AND content != '' "
            "AND role IN ('user','assistant') ORDER BY id",
            (since_id,)).fetchall()
    except sqlite3.Error:
        return [], since_id
    finally:
        con.close()

    out, high = [], since_id
    for mid, role, content, ts in rows:
        high = max(high, mid)
        text = _SYSTEM_HINT_RE.sub("", content or "").strip()
        if text:
            out.append({"id": mid, "role": role, "text": text, "ts": ts})
    return out, high


def build_transcript(msgs: list[dict]) -> str:
    """Render messages oldest->newest, trimmed from the FRONT to the char cap so
    the most recent conversation always survives."""
    lines = []
    for m in msgs:
        who = "Trickery" if m["role"] == "user" else "Aurinex"
        day = time.strftime("%Y-%m-%d", time.localtime(m["ts"] or 0))
        lines.append(f"[{day}] {who}: {m['text']}")
    text = "\n".join(lines)
    cap = config.DISTILL_MAX_TRANSCRIPT_CHARS
    if len(text) > cap:
        text = "…(earlier conversation trimmed)…\n" + text[-cap:]
    return text


_SYS_PROMPT = (
    "You maintain the long-term user-profile file for a voice assistant. You are "
    "given the file's current contents and a transcript of recent conversation. "
    "You output ONLY the complete new contents of the file — no commentary, no "
    "code fences, no preamble."
)


def _instruction(current: str, transcript: str, budget: int) -> str:
    return (
        "Below are two blocks: the CURRENT contents of the assistant's user-profile "
        "memory file, and a TRANSCRIPT of recent conversation between the user "
        "(Trickery) and the assistant (Aurinex).\n\n"
        "Rewrite the file so it stays an accurate, durable profile of the user.\n\n"
        "Rules:\n"
        "- Output the ENTIRE file, not a diff. Anything you omit is deleted.\n"
        "- PRESERVE every existing fact unless the transcript clearly contradicts "
        "it (then correct it) or it is plainly obsolete.\n"
        "- ADD only durable things: stable preferences, how he wants the assistant "
        "to behave, recurring projects, workflow facts, corrections he made. "
        "Prefer facts he stated about himself or asked the assistant to remember.\n"
        "- Do NOT add one-off task details, transient status, anything you are "
        "unsure of, or anything the assistant merely inferred about him.\n"
        "- Keep the existing markdown structure and terse bullet style.\n"
        "- Never rename the user: he is Trickery, never Troy.\n"
        f"- HARD LIMIT: the whole file must be under {budget} characters. If adding "
        "something would exceed it, merge or tighten existing lines instead of "
        "dropping facts.\n"
        "- If nothing in the transcript is worth recording, output the current "
        "file back UNCHANGED, byte for byte.\n\n"
        f"<current_file>\n{current}\n</current_file>\n\n"
        f"<transcript>\n{transcript}\n</transcript>"
    )


def _ask_model(instruction: str) -> str:
    out = brain.complete(instruction, _SYS_PROMPT, config.DISTILL_MODEL)
    # Models like to wrap a whole-file answer in a fence even when told not to.
    m = re.match(r"^```(?:markdown|md)?\s*\n(.*)\n```\s*$", out, re.DOTALL)
    return (m.group(1) if m else out).strip()


class Rejected(RuntimeError):
    """The proposed file failed a safety check — abort, keep what we have."""


def validate(current: str, proposed: str, budget: int) -> None:
    """Refuse a proposal that would damage the profile. Deliberately strict: a
    silently-mangled memory file is far worse than a skipped nightly run."""
    if not proposed:
        raise Rejected("model returned nothing")
    if len(proposed) > budget:
        raise Rejected(f"over budget ({len(proposed)} > {budget} chars)")
    if "Trickery" not in proposed:
        raise Rejected("proposal dropped the user's name — almost certainly wrong")
    # Guard against a summarize-instead-of-rewrite failure: a real edit stays
    # roughly the same size. Shrinking by more than half means facts were lost.
    if current.strip() and len(proposed) < len(current.strip()) * 0.5:
        raise Rejected(f"proposal shrank the file by more than half "
                       f"({len(current.strip())} -> {len(proposed)} chars)")
    if re.search(r"^(here'?s|i've|sure[,!]|certainly)", proposed, re.IGNORECASE):
        raise Rejected("proposal looks like chat commentary, not file contents")


def _backup(path: Path, text: str) -> Path | None:
    try:
        config.DISTILL_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        dest = config.DISTILL_BACKUP_DIR / f"{path.name}.{time.strftime('%Y%m%d-%H%M%S')}"
        dest.write_text(text, encoding="utf-8")
        return dest
    except OSError:
        return None


def _write_diff(current: str, proposed: str) -> None:
    try:
        d = "\n".join(difflib.unified_diff(
            current.splitlines(), proposed.splitlines(),
            fromfile="USER.md (before)", tofile="USER.md (after)", lineterm=""))
        config.DISTILL_DIFF_FILE.write_text(
            d or "(no textual change)\n", encoding="utf-8")
    except OSError:
        pass


def run(dry_run: bool = False, verbose: bool = False) -> int:
    """One distillation pass. Returns a shell exit code."""
    def say(msg: str) -> None:
        if verbose:
            print(msg)

    target = config.HERMES_USER_FILE
    if not target.exists():
        _log(f"no user profile at {target} — nothing to do")
        say(f"no user profile at {target}")
        return 1

    since = _read_watermark()
    msgs, high = new_messages(since)
    say(f"{len(msgs)} new messages since id {since}")
    if len(msgs) < config.DISTILL_MIN_MESSAGES:
        _log(f"skipped — only {len(msgs)} new messages "
             f"(need {config.DISTILL_MIN_MESSAGES})")
        say("below threshold, skipping")
        return 0

    current = target.read_text(encoding="utf-8")
    transcript = build_transcript(msgs)
    say(f"transcript {len(transcript)} chars; asking {config.DISTILL_MODEL}…")

    try:
        proposed = _ask_model(_instruction(current, transcript,
                                           config.DISTILL_BUDGET_CHARS))
        validate(current, proposed, config.DISTILL_BUDGET_CHARS)
    except (Rejected, RuntimeError) as e:
        _log(f"REJECTED: {e} — profile left unchanged")
        say(f"rejected: {e}")
        return 1

    _write_diff(current, proposed)
    if proposed.strip() == current.strip():
        _write_watermark(high)          # nothing to learn, but don't re-read it
        _log(f"no change ({len(msgs)} messages reviewed)")
        say("no change")
        return 0

    if dry_run:
        say("--- proposed diff ---")
        say(config.DISTILL_DIFF_FILE.read_text(encoding="utf-8"))
        say("(dry run — nothing written)")
        return 0

    bak = _backup(target, current)
    try:
        target.write_text(proposed, encoding="utf-8")
    except OSError as e:
        _log(f"write failed: {e}")
        return 1
    _write_watermark(high)
    _log(f"updated USER.md ({len(current)} -> {len(proposed)} chars) from "
         f"{len(msgs)} messages; backup {bak.name if bak else 'none'}")
    say(f"updated {target} ({len(current)} -> {len(proposed)} chars)")
    say(f"diff: {config.DISTILL_DIFF_FILE}")
    return 0
