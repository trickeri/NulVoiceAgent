"""Agent completion announcer — driven by Claude Code or Codex **Stop hooks**.

Flow (per finished agent turn):
  1. Speak a short, varied confirmation in the configured voice
     ("Codex finished the bug fix.", "Claude's plan is ready.", ...).
  2. Open a brief mic window. If the very next thing the user says contains
     "summarize", speak a 2-3 sentence summary of what the agent said — focused on
     anything it's waiting on from him and any gotchas.

Claude Code's Stop hook payload (JSON on stdin) carries `transcript_path`; Codex
Stop hook payloads carry lifecycle JSON directly. We recover the last assistant
message + tool signals, classify the task type for phrasing, and feed the
assistant text to brain.summarize() on request.

Runs detached (the hook forks + returns 0 immediately) so Claude Code's prompt
isn't blocked for the speak/listen/summarize round-trip. A flock makes it
single-flight so several sessions stopping at once don't talk over each other,
and we bail if a nulvoiceagent push-to-talk turn is live so we never barge in on it.
"""
from __future__ import annotations
import fcntl
import json
import os
import random
import re
import sys
import time

from . import brain, config, dispatch, stt, tts

_LOG = config.CACHE_DIR / "nulvoiceagent.log"
_LOCK_FH = None  # kept alive for the lock's lifetime (released on process exit)

# Varied confirmations per detected task type. random.choice gives the variation
# the user asked for; the type comes from _classify(). "{who}" is filled with the
# pane's spoken name ("Hermes Claude") so he knows WHICH Claude finished — or just
# "Claude" when the pane isn't named.
_PHRASES = {
    "task": [
        "{who} finished the task.",
        "{who}'s done with the task.",
        "{who} wrapped up the task.",
        "{who} completed the task.",
    ],
    "plan": [
        "{who}'s plan is ready.",
        "{who} finished planning.",
        "{who}'s done laying out the plan.",
        "{who} has a plan ready for you.",
    ],
    "bugfix": [
        "{who} finished the bug fix.",
        "{who} squashed the bug.",
        "{who}'s done with the fix.",
        "{who} finished the fix.",
    ],
    "investigation": [
        "{who} finished investigating.",
        "{who} wrapped up the investigation.",
        # _pick_phrase() flips "digging into it" <-> "looking into it" 50/50.
        "{who} finished digging into it.",
    ],
    # the user asked a question and the agent answered it (no edits, not asking back).
    "answer": [
        "{who} answered your question.",
        "{who} has your answer.",
        "{who} answered that for you.",
        "{who} got your answer ready.",
    ],
    "question": [
        "{who} stopped for a clarifying question.",
        "{who} has a question for you.",
        "{who} paused to ask you something.",
        "{who} needs your input before it continues.",
    ],
}


def _log(line: str) -> None:
    try:
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')}  agent-done: {line}\n")
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# runtime mute (taskbar agent-state context menu / `nulvoiceagent done-mute`)     #
# --------------------------------------------------------------------------- #
def is_muted() -> bool:
    """True if agent-completion announcements are currently muted."""
    try:
        return config.DONE_MUTE_FLAG.read_text(encoding="utf-8").strip() == "1"
    except OSError:
        return False


def set_muted(muted: bool) -> bool:
    """Mute (write the flag) or unmute (remove it). Returns the resulting state."""
    try:
        if muted:
            config.DONE_MUTE_FLAG.write_text("1", encoding="utf-8")
        else:
            config.DONE_MUTE_FLAG.unlink(missing_ok=True)
    except OSError:
        pass
    return is_muted()


# --------------------------------------------------------------------------- #
# voice-only mode: announce ONLY for tasks started by voice (not typed)       #
# --------------------------------------------------------------------------- #
def is_voice_only() -> bool:
    """True if completion announcements are restricted to voice-initiated tasks."""
    try:
        return config.DONE_VOICE_ONLY_FLAG.read_text(encoding="utf-8").strip() == "1"
    except OSError:
        return False


def set_voice_only(on: bool) -> bool:
    """Enable/disable voice-only announcements (write/remove the flag). Returns the
    resulting state."""
    try:
        if on:
            config.DONE_VOICE_ONLY_FLAG.write_text("1", encoding="utf-8")
        else:
            config.DONE_VOICE_ONLY_FLAG.unlink(missing_ok=True)
    except OSError:
        pass
    return is_voice_only()


def _consume_voice_mark(pane: str | None) -> bool:
    """Read + delete this pane's voice-command marker (dispatch stamps it when it
    routes a spoken command into the pane). True if a FRESH mark was present — i.e.
    the just-finished task was started by voice rather than typed. Consuming it
    means one voice command authorizes exactly one completion announcement; a typed
    follow-up afterward (which leaves no mark) won't be announced in voice-only
    mode."""
    if not pane:
        return False
    p = config.voice_mark_path(pane)
    try:
        ts = float(p.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass
    return (time.time() - ts) <= config.DONE_VOICE_MARK_TTL


def peek_voice_mark(pane: str | None) -> bool:
    """Like _consume_voice_mark but NON-destructive: True if a fresh voice mark is
    present without removing it. Used by mid-task announcers (e.g. the AskUserQuestion
    nudge) that must respect voice-only mode but mustn't steal the mark the
    end-of-task done-announcer still needs to consume."""
    if not pane:
        return False
    try:
        ts = float(config.voice_mark_path(pane).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return (time.time() - ts) <= config.DONE_VOICE_MARK_TTL


# --------------------------------------------------------------------------- #
# transcript parsing                                                          #
# --------------------------------------------------------------------------- #
def _content_items(msg: dict) -> list:
    c = msg.get("content")
    if isinstance(c, str):
        return [{"type": "text", "text": c}]
    return c if isinstance(c, list) else []


def _user_prompt_text(entry: dict) -> str:
    """Genuine user prompt text for a transcript `user` entry, or '' if the entry
    is just a tool_result carrier (which is NOT a turn boundary)."""
    parts = []
    for it in _content_items(entry.get("message", {})):
        if it.get("type") == "text" and it.get("text"):
            parts.append(it["text"])
    return "\n".join(parts).strip()


def _extract_claude_transcript(path: str | None) -> tuple[str, str, set[str]]:
    """Return (last user prompt, last assistant text, tool signals) for the most
    recent turn in the transcript. Tool signals are lowercased tool names plus
    `skill:<name>` for Skill invocations."""
    if not path:
        return "", "", set()
    try:
        with open(path, encoding="utf-8") as f:
            entries = [json.loads(ln) for ln in f if ln.strip()]
    except (OSError, json.JSONDecodeError):
        # tolerate a partially-written line at EOF
        entries = []
        try:
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        entries.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
        except OSError:
            return "", "", set()

    boundary = None
    user_text = ""
    for i in range(len(entries) - 1, -1, -1):
        if entries[i].get("type") == "user":
            t = _user_prompt_text(entries[i])
            if t:
                boundary, user_text = i, t
                break

    start = boundary + 1 if boundary is not None else 0
    asst_parts: list[str] = []
    tools: set[str] = set()
    for e in entries[start:]:
        if e.get("type") != "assistant":
            continue
        for it in _content_items(e.get("message", {})):
            if it.get("type") == "text" and it.get("text"):
                asst_parts.append(it["text"])
            elif it.get("type") == "tool_use":
                name = str(it.get("name", ""))
                tools.add(name.lower())
                if name == "Skill":
                    inp = it.get("input") or {}
                    s = inp.get("skill") or inp.get("command")
                    if s:
                        tools.add("skill:" + str(s).lower())
    return user_text.strip(), "\n".join(asst_parts).strip(), tools


def _text_from_content(content) -> str:
    """Best-effort text extraction across Claude and Codex transcript shapes."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for it in content:
            if isinstance(it, str):
                parts.append(it)
            elif isinstance(it, dict):
                for key in ("text", "input_text", "output_text", "message", "content"):
                    val = it.get(key)
                    if isinstance(val, str) and val:
                        parts.append(val)
                        break
        return "\n".join(parts)
    return ""


def _collect_codex_events(obj, out: list[tuple[str, str]], tools: set[str]) -> None:
    """Walk Codex hook JSON and collect user/assistant text plus tool signals.

    Codex hook payload details are intentionally tolerated rather than tightly
    coupled: current session records and hook payloads both expose message-like
    dicts with `role`/`content`, while tool activity appears under names such as
    `tool_name`, `recipient`, `call_id`, or `type`.
    """
    if isinstance(obj, list):
        for it in obj:
            _collect_codex_events(it, out, tools)
        return
    if not isinstance(obj, dict):
        return

    payload = obj.get("payload")
    if isinstance(payload, dict):
        _collect_codex_events(payload, out, tools)

    role = obj.get("role")
    if role in ("user", "assistant"):
        text = _text_from_content(obj.get("content") or obj.get("message"))
        if text.strip():
            out.append((role, text.strip()))

    typ = str(obj.get("type", "")).lower()
    if typ in ("function_call", "tool_call", "tool_use"):
        name = obj.get("name") or obj.get("tool_name") or obj.get("recipient")
        if name:
            tools.add(str(name).lower())
    for key in ("tool_name", "tool", "recipient_name", "recipient", "call_id"):
        val = obj.get(key)
        if isinstance(val, str) and val:
            tools.add(val.lower())

    for key, val in obj.items():
        if key == "payload":
            continue
        if isinstance(val, (dict, list)):
            _collect_codex_events(val, out, tools)


def _extract_codex_payload(payload: dict) -> tuple[str, str, set[str]]:
    events: list[tuple[str, str]] = []
    tools: set[str] = set()
    _collect_codex_events(payload, events, tools)
    boundary = None
    user_text = ""
    for i in range(len(events) - 1, -1, -1):
        role, text = events[i]
        if role == "user" and text:
            boundary, user_text = i, text
            break
    start = boundary + 1 if boundary is not None else 0
    asst_text = "\n".join(text for role, text in events[start:] if role == "assistant")
    if not asst_text:
        for role, text in reversed(events):
            if role == "assistant":
                asst_text = text
                break
    return user_text.strip(), asst_text.strip(), tools


def _extract(payload: dict) -> tuple[str, str, set[str]]:
    if payload.get("transcript_path"):
        return _extract_claude_transcript(payload.get("transcript_path"))
    return _extract_codex_payload(payload)


def _hook_event_name(payload: dict) -> str:
    """Lifecycle event name from Claude/Codex hook JSON, if one is present."""
    for key in ("hook_event_name", "hookEventName", "event", "name"):
        val = payload.get(key)
        if isinstance(val, str):
            return val.strip()
    return ""


def _is_subagent_completion(payload: dict) -> bool:
    """True for subagent lifecycle completions, which should not be announced.

    Codex exposes a distinct SubagentStop hook event. Claude-compatible hook
    payloads also commonly carry `hook_event_name`; keep this check structural
    and top-level so normal main-agent turns that merely mention subagents do not
    get suppressed.
    """
    event = _hook_event_name(payload).lower().replace("_", "").replace("-", "")
    if event in {"subagentstop", "subagentcomplete", "subagentcompleted"}:
        return True
    if event and event != "stop":
        return False
    for key in ("subagent_type", "subagentType", "subagent_name", "subagentName"):
        if isinstance(payload.get(key), str) and payload[key].strip():
            return True
    return False


def _ends_with_question(text: str) -> bool:
    """True if the assistant's final message ends on a question (a clarifying
    question it's waiting on the user to answer), ignoring trailing whitespace and
    markdown emphasis / closing punctuation."""
    t = re.sub(r"[\s*`\"')\]}]+$", "", text or "")
    return t.endswith("?")


# Interrogative openers — used to spot when *the user* asked a question, so a
# no-edit completion is phrased "answered your question" rather than "looked into
# it". Whisper rarely transcribes a spoken "?", so we also match these openers.
_QUESTION_OPENER = re.compile(
    r"^(what|whats|why|how|hows|when|where|who|which|whose|whom|"
    r"is|are|am|do|does|did|can|could|would|should|will|shall|may|might|"
    r"has|have|had|was|were)\b",
    re.IGNORECASE)


def _is_user_question(user_text: str) -> bool:
    """True if the user's prompt reads as a question — ends with '?' or opens with
    an interrogative word."""
    t = (user_text or "").strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    return bool(_QUESTION_OPENER.match(t))


def _pick_phrase(kind: str) -> str:
    """Choose a confirmation for `kind`. For investigation, the 'digging into it'
    phrasing is a 50/50 coin flip with the synonym 'looking into it'."""
    phrase = random.choice(_PHRASES[kind])
    if "digging into it" in phrase and random.random() < 0.5:
        phrase = phrase.replace("digging into it", "looking into it")
    return phrase


def _classify(user_text: str, asst_text: str, tools: set[str]) -> str:
    edited = bool(tools & {"edit", "write", "multiedit", "notebookedit",
                           "apply_patch", "functions.apply_patch"})
    edited = edited or any(t.endswith(".apply_patch") for t in tools)
    blob = f"{user_text} {asst_text}".lower()
    ulow = user_text.lower()
    if "exitplanmode" in tools or "skill:plan" in tools:
        return "plan"
    if not edited and re.search(r"\bplan\b", ulow):
        return "plan"
    # A clarifying-question stop takes precedence over the work type: what matters
    # is that it's waiting on him. (Plan-mode is handled above with its own line.)
    if "askuserquestion" in tools or _ends_with_question(asst_text):
        return "question"
    if "skill:fix" in tools or (
        edited and re.search(r"\b(bug|crash|broken|regression|fix|fixed|error)\b", blob)
    ):
        return "bugfix"
    # No edits + the user's prompt was a question -> Claude answered it.
    if not edited and _is_user_question(user_text):
        return "answer"
    if not edited:
        return "investigation"
    return "task"


def _speaker(pane: str | None = None, default: str = "Claude") -> str:
    """The spoken name of THIS agent's tmux pane (e.g. 'Hermes Codex'), from the
    given pane id (defaulting to the hook's inherited $TMUX_PANE), or `default` if
    the pane isn't named / not tmux."""
    if pane is None:
        pane = os.environ.get("TMUX_PANE")
    try:
        name = dispatch.name_for_pane(pane)
    except Exception:  # noqa: BLE001 — never break the announce flow over a name
        name = None
    return name or default


# Latest announced completion, stashed so a follow-up "summarize" — said ANY time
# (the auto-listen window OR, more importantly, the normal chat hotkey) — knows
# what to summarize. Single latest entry: "summarize" means the one just reported.
_COMPLETION = config.CACHE_DIR / "last-completion.json"
_COMPLETION_TTL = 300.0  # seconds — don't summarize something reported ages ago
_SUMMARY_KW = re.compile(r"\bsummar(?:ize|ise|y)\b", re.IGNORECASE)


def _stash_completion(who: str, kind: str, text: str) -> None:
    try:
        _COMPLETION.write_text(
            json.dumps({"who": who, "kind": kind, "text": text, "ts": time.time()}),
            encoding="utf-8")
    except OSError:
        pass


def _read_completion() -> dict | None:
    try:
        d = json.loads(_COMPLETION.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if time.time() - float(d.get("ts", 0)) > _COMPLETION_TTL:
        return None
    return d


def wants_summary_request(text: str) -> bool:
    """True if `text` is a bare 'summarize' command (e.g. 'summarize', 'summarize
    that', 'summarize what Krita Claude just did') — short + contains the keyword.
    The word cap keeps a real chat request like 'summarize this 5-page doc …' out;
    even so, make_summary() only fires when there's a FRESH completion to summarize,
    otherwise the caller falls through to normal chat."""
    t = (text or "").strip()
    return bool(_SUMMARY_KW.search(t)) and len(t.split()) <= 8


def make_summary() -> str | None:
    """Summarize the most recently announced completion. Returns the spoken summary,
    or None when there's no fresh completion (caller should fall through to chat)."""
    d = _read_completion()
    if d is None:
        return None
    who = d.get("who", "the agent")
    text = (d.get("text") or "").strip()
    if not text:
        return f"{who} didn't leave any detail I can summarize."
    try:
        s = brain.summarize(text[: config.DONE_MAX_SUMMARY_CHARS])
    except Exception as e:  # noqa: BLE001 — never crash the mic flow
        _log(f"summarize error: {e}")
        return "I couldn't put together a summary, sorry."
    _log(f"summary of {who!r}: {s!r}")
    return s or f"{who} didn't leave any detail I can summarize."


# --------------------------------------------------------------------------- #
# don't talk over a live nulvoiceagent turn                                      #
# --------------------------------------------------------------------------- #
def _busy() -> bool:
    """True if a nulvoiceagent push-to-talk recording or reply is in flight (its
    pidfiles live), so we skip rather than barge in on the user."""
    rec = config.CACHE_DIR / "recording.pid"
    try:
        if rec.exists():
            os.kill(int(rec.read_text().strip()), 0)
            return True
    except (OSError, ValueError):
        pass
    turn = config.CACHE_DIR / "turn.pgid"
    try:
        if turn.exists():
            os.killpg(int(turn.read_text().strip()), 0)
            return True
    except (OSError, ValueError):
        pass
    return False


def _acquire_lock():
    global _LOCK_FH
    fh = open(config.CACHE_DIR / "claude-done.lock", "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    _LOCK_FH = fh
    return fh


# --------------------------------------------------------------------------- #
# entry points                                                                #
# --------------------------------------------------------------------------- #
def run(payload: dict, default_who: str = "Claude") -> None:
    """Announce + offer the summary. Synchronous; call via run_detached()."""
    if _is_subagent_completion(payload):
        _log("skip: subagent completion")
        return
    if not config.DONE_ENABLE or is_muted():
        return
    if _acquire_lock() is None:
        return                       # another announcement is already speaking
    if _busy():
        return                       # a nulvoiceagent turn is live — don't barge in

    pane = os.environ.get("TMUX_PANE")
    # One voice-routed command authorizes one announcement: read + consume this
    # pane's voice mark. In voice-only mode a task the user TYPED in (no mark)
    # finishes silently; the default (flag off) announces regardless.
    voice_initiated = _consume_voice_mark(pane)
    if is_voice_only() and not voice_initiated:
        _log(f"skip: voice-only mode, task wasn't voice-initiated (pane={pane})")
        return

    user_text, asst_text, tools = _extract(payload)
    kind = _classify(user_text, asst_text, tools)
    who = _speaker(pane, default=default_who)
    # Stash BEFORE announcing so "summarize" works whether it arrives in the brief
    # auto-listen window below or later via the normal chat hotkey (see __main__).
    _stash_completion(who, kind, asst_text)
    phrase = _pick_phrase(kind).format(who=who)
    _log(f"who={who!r} type={kind} -> {phrase!r}")
    tts.speak(phrase)

    # Convenience no-hotkey path: listen briefly for an immediate "summarize".
    if _busy():
        return
    heard = stt.listen(onset_timeout=config.DONE_LISTEN_SECS)
    if not heard:
        return
    _log(f"heard: {heard!r}")
    if _SUMMARY_KW.search(heard):
        tts.speak(make_summary() or "There wasn't really anything to summarize.")


def run_detached(payload: dict, default_who: str = "Claude") -> None:
    """Fork a detached child to do the speak/listen/summarize work and return at
    once, so the Stop hook doesn't block Claude Code's prompt."""
    if _is_subagent_completion(payload):
        _log("skip detach: subagent completion")
        return
    if not config.DONE_ENABLE or is_muted():
        return                       # don't even fork when off / muted
    try:
        pid = os.fork()
    except OSError:
        run(payload, default_who=default_who)  # no fork available — just run inline
        return
    if pid > 0:
        return                       # parent returns to the hook immediately
    # child: detach into its own session and silence std streams
    try:
        os.setsid()
    except OSError:
        pass
    try:
        devnull = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            try:
                os.dup2(devnull, fd)
            except OSError:
                pass
    except OSError:
        pass
    try:
        run(payload, default_who=default_who)
    finally:
        os._exit(0)


def from_stdin(default_who: str = "Claude", codex_noop: bool = False) -> None:
    """CLI seam: read the Stop-hook JSON from stdin and kick off run_detached()."""
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        payload = {}
    run_detached(payload, default_who=default_who)
    if codex_noop:
        print("{}")
