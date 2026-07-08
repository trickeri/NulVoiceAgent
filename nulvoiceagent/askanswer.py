"""Answer a Claude Code AskUserQuestion prompt **by voice**.

Two halves:

1. ANNOUNCE (Claude Code `PreToolUse` hook, matcher `AskUserQuestion`).
   When any Claude pane calls AskUserQuestion the hook fires `nulvoiceagent
   ask-pending`, which reads the hook JSON on stdin (it carries the full
   structured `tool_input.questions` + inherits `$TMUX_PANE`), records a small
   pending file for that pane, and speaks a short configured-voice nudge — so the user
   hears "Build Claude needs you to pick options" instead of silently missing it.
   A `PostToolUse` hook (`ask-clear`) drops the record once it's been answered.

2. ANSWER (voice command, shared on the chat hotkey via try_answer()).
   "answer Build Claude" / "pick options for the build pane" / "let me answer the
   questions" opens an interactive loop: for each question it reads the options
   aloud, listens for a spoken number (one/two/three…, ordinals, or the option's
   word), and DRIVES THE LIVE SELECTOR by sending keystrokes into the pane. After
   the last question it reads the choices back and waits for "submit".

The selector keystroke contract (verified against Claude Code v2.1.195):
  * single-select question: digit N selects option N AND auto-advances to the next
    question tab;
  * multi-select question:  digit N toggles option N (no advance); send Tab to
    advance once the toggles are done;
  * the final Submit tab is a 1=Submit / 2=Cancel list — Enter (or "1") submits;
  * Esc cancels the whole prompt.

Driving is CLOSED-LOOP: every step re-reads the pane with `tmux capture-pane` and
parses what's actually on screen (current question, its options, multi-select?,
or "we're at Submit"), so the loop tracks the real UI rather than trusting an open
-loop key count — and a wrong keystroke can't silently cascade. This also lets it
answer a prompt that was already on screen before the hook recorded anything.
"""
from __future__ import annotations

import fcntl
import json
import os
import random
import re
import subprocess
import sys
import time
from typing import Optional

from . import config, dispatch, done, stt, tts

_LOG = config.CACHE_DIR / "nulvoiceagent.log"
_LOCK_FH = None  # held for the announce lock's lifetime


def _log(line: str) -> None:
    try:
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')}  ask: {line}\n")
    except OSError:
        pass


class Result:
    """Mirror dispatch.Result so __main__ handles both the same way."""
    def __init__(self, ok: bool, say: str):
        self.ok = ok
        self.say = say


# --------------------------------------------------------------------------- #
# pending records                                                             #
# --------------------------------------------------------------------------- #
def _record_pending(pane: Optional[str], who: str, questions: list) -> None:
    if not pane:
        return
    try:
        config.ASK_PENDING_DIR.mkdir(parents=True, exist_ok=True)
        config.ask_pending_path(pane).write_text(
            json.dumps({"pane": pane, "who": who, "questions": questions,
                        "ts": time.time()}),
            encoding="utf-8")
    except OSError:
        pass


def clear_pending(pane: Optional[str]) -> None:
    if not pane:
        return
    try:
        config.ask_pending_path(pane).unlink(missing_ok=True)
    except OSError:
        pass


def _read_pending(pane: str) -> Optional[dict]:
    try:
        d = json.loads(config.ask_pending_path(pane).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if time.time() - float(d.get("ts", 0)) > config.ASK_PENDING_TTL:
        clear_pending(pane)
        return None
    return d


def _all_pending() -> list[dict]:
    """Every live (non-stale) pending record, newest first."""
    out = []
    try:
        files = sorted(config.ASK_PENDING_DIR.glob("*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return out
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if time.time() - float(d.get("ts", 0)) > config.ASK_PENDING_TTL:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# tmux plumbing                                                               #
# --------------------------------------------------------------------------- #
def _tmux(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(("tmux", *args), capture_output=True, text=True)


def _capture(pane: str) -> str:
    """The visible text of `pane` (the selector lives at the bottom, on screen)."""
    cp = _tmux("capture-pane", "-p", "-t", pane)
    return cp.stdout if cp.returncode == 0 else ""


def _send_digit(pane: str, n: int) -> None:
    _tmux("send-keys", "-t", pane, "-l", str(n))   # -l = literal, not a key name


def _send_key(pane: str, key: str) -> None:
    _tmux("send-keys", "-t", pane, key)            # "Enter" / "Tab" / "Escape"


# --------------------------------------------------------------------------- #
# screen parsing — what's on the selector right now                           #
# --------------------------------------------------------------------------- #
_OPT_RE = re.compile(r"^[\s❯>●○•]*?(\d+)\.\s+(.*\S)\s*$")
_RULE_RE = re.compile(r"^[\s─━—\-]+$")
_SKIP_LABELS = {"type something", "type something.", "chat about this",
                "submit answers", "cancel"}


def _strip_checkbox(label: str) -> str:
    return re.sub(r"^\[.?\]\s*", "", label).strip()


def read_screen(pane: str) -> Optional[dict]:
    """Parse what the selector is showing in `pane` right now. Returns one of:
      {"kind": "submit", "review": "<choices text>"}  — at the Submit/review tab,
      {"kind": "question", "question": str, "options": [labels], "multi": bool},
      {"kind": "gone"}                                  — no selector visible,
    or None if the pane can't be read."""
    text = _capture(pane)
    if not text:
        return None
    low = text.lower()
    # Submit/review screen first — it's a confirm list with its OWN footer, so it
    # mustn't be gated behind the question footer below.
    if "submit answers" in low or "ready to submit" in low or "review your answers" in low:
        review = []
        for ln in text.splitlines():
            s = ln.strip()
            if s.startswith("→") or s.startswith("->"):
                review.append(re.sub(r"^(→|->)\s*", "", s))
        return {"kind": "submit", "review": ", ".join(review)}

    lines = text.splitlines()
    opt_idx = [i for i, ln in enumerate(lines) if _OPT_RE.match(ln)]
    # A live question selector has numbered option lines AND the nav/cancel footer;
    # without both, there's no AskUserQuestion prompt on screen.
    footer = any(k in low for k in ("to cancel", "to navigate", "to select"))
    if not opt_idx or not footer:
        return {"kind": "gone"}
    first = opt_idx[0]
    multi = False
    options: list[str] = []
    for i in opt_idx:
        m = _OPT_RE.match(lines[i])
        raw = m.group(2)
        if "[" in raw[:4] and "]" in raw[:6]:
            multi = True
        label = _strip_checkbox(raw)
        if label.strip().lower() in _SKIP_LABELS:
            continue
        options.append(label)
    # Question text = the nearest non-empty, non-rule, non-tab line above the first
    # option (tab line carries the ☐/☒/Submit headers; skip it).
    question = ""
    for j in range(first - 1, -1, -1):
        s = lines[j].strip()
        if not s or _RULE_RE.match(s):
            continue
        if "✔ submit" in s.lower() or s.startswith("←") or "☐" in s or "☒" in s:
            continue
        question = s
        break
    if not options:
        return {"kind": "gone"}
    return {"kind": "question", "question": question, "options": options, "multi": multi}


def _wait_screen(pane: str, want, tries: int = 6, delay: Optional[float] = None):
    """Re-read the pane until `want(screen)` is true (or tries run out). Returns the
    matching screen, or the last screen read."""
    delay = config.ASK_RENDER_DELAY if delay is None else delay
    scr = None
    for _ in range(tries):
        scr = read_screen(pane)
        if scr is not None and want(scr):
            return scr
        time.sleep(delay)
    return scr


# --------------------------------------------------------------------------- #
# driving — apply picks, submit, cancel                                       #
# --------------------------------------------------------------------------- #
def apply_picks(pane: str, screen: dict, picks: list[int]) -> None:
    """Send the chosen option(s) for the CURRENT question. `picks` are 1-based
    option numbers. Single-select: one digit (auto-advances). Multi-select: a digit
    per pick (toggles), then Tab to advance."""
    if screen.get("multi"):
        for n in picks:
            _send_digit(pane, n)
            time.sleep(0.12)
        _send_key(pane, "Tab")
    elif picks:
        _send_digit(pane, picks[0])


def submit(pane: str) -> bool:
    scr = _wait_screen(pane, lambda s: s.get("kind") in ("submit", "gone"))
    if scr and scr.get("kind") == "submit":
        _send_key(pane, "Enter")
        time.sleep(config.ASK_RENDER_DELAY)
        return True
    return False


def cancel(pane: str) -> None:
    _send_key(pane, "Escape")


# --------------------------------------------------------------------------- #
# spoken-number parsing                                                       #
# --------------------------------------------------------------------------- #
_NUM_WORDS = {
    "one": 1, "won": 1, "juan": 1, "first": 1,
    "two": 2, "to": 2, "too": 2, "tu": 2, "second": 2,
    "three": 3, "tree": 3, "third": 3,
    "four": 4, "for": 4, "fore": 4, "fourth": 4,
    "five": 5, "fifth": 5,
    "six": 6, "sixth": 6,
    "seven": 7, "seventh": 7,
    "eight": 8, "ate": 8, "eighth": 8,
    "nine": 9, "ninth": 9,
    "ten": 10, "tenth": 10,
}
_RE_CANCEL = re.compile(r"\b(cancel|never\s*mind|nevermind|forget it|abort|"
                        r"quit|escape|exit|stop)\b", re.IGNORECASE)
_RE_REPEAT = re.compile(r"\b(repeat|again|one more time|read (them|it|that)|"
                        r"what (were|are) (the|my)|say (that|them) again)\b",
                        re.IGNORECASE)
_RE_SUBMIT = re.compile(r"\b(submit|send it|send|confirm|do it|go ahead|"
                        r"that'?s? (it|all|everything)|yes|yep|yeah|"
                        r"looks good)\b", re.IGNORECASE)
_RE_DONE = re.compile(r"\b(done|next|finish(ed)?|that'?s (it|all)|go on|"
                      r"continue|nothing else|move on)\b", re.IGNORECASE)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())


def parse_numbers(text: str, options: list[str]) -> list[int]:
    """Spoken text -> 1-based option numbers, in spoken order, de-duplicated and
    restricted to 1..len(options). Accepts digits, number words/ordinals, and the
    option's own label word ('blue' -> the Blue option)."""
    n = len(options)
    out: list[int] = []
    toks = _norm(text).split()
    for t in toks:
        v = None
        if t.isdigit():
            v = int(t)
        elif t in _NUM_WORDS:
            v = _NUM_WORDS[t]
        if v is not None and 1 <= v <= n and v not in out:
            out.append(v)
    # Label match: a distinctive option word spoken instead of a number.
    low = _norm(text)
    for i, label in enumerate(options, start=1):
        for w in _norm(label).split():
            if len(w) >= 3 and re.search(rf"\b{re.escape(w)}\b", low) and i not in out:
                out.append(i)
                break
    return out


# --------------------------------------------------------------------------- #
# the interactive answer loop                                                 #
# --------------------------------------------------------------------------- #
def _say_question(idx: int, total: int, screen: dict) -> None:
    opts = screen["options"]
    parts = []
    if total > 1:
        parts.append(f"Question {idx + 1}.")
    q = screen.get("question") or "Pick an option"
    parts.append(q.rstrip(".") + ".")
    if screen.get("multi"):
        parts.append("Pick any that apply — say the numbers, then say done.")
    ordinals = ["one", "two", "three", "four", "five", "six", "seven", "eight",
                "nine", "ten"]
    opt_say = " ".join(
        f"{ordinals[i] if i < len(ordinals) else i + 1}, {lbl}."
        for i, lbl in enumerate(opts))
    parts.append("Your options: " + opt_say)
    tts.speak(" ".join(parts))


def _listen_pick(screen: dict) -> tuple[str, list[int]]:
    """Listen for one answer to the current question. Returns a (verb, numbers)
    pair: verb in {'pick','done','cancel','repeat','none'}."""
    heard = stt.listen(onset_timeout=config.ASK_LISTEN_SECS)
    _log(f"heard: {heard!r}")
    if not heard:
        return ("none", [])
    if _RE_CANCEL.search(heard):
        return ("cancel", [])
    if _RE_REPEAT.search(heard):
        return ("repeat", [])
    nums = parse_numbers(heard, screen["options"])
    if nums:
        return ("pick", nums)
    if screen.get("multi") and (_RE_DONE.search(heard) or _RE_SUBMIT.search(heard)):
        return ("done", [])
    return ("none", [])


def _answer_question(idx: int, total: int, screen: dict) -> tuple[str, list[int]]:
    """Read the question aloud + collect the answer. Returns ('picks', [1-based]) /
    ('cancel', []) / ('none', []). Single-select takes the first number from one
    utterance; multi-select accumulates numbers across utterances until 'done'."""
    _say_question(idx, total, screen)
    if not screen.get("multi"):
        for attempt in range(3):
            verb, nums = _listen_pick(screen)
            if verb == "repeat":
                _say_question(idx, total, screen)
            elif verb == "cancel":
                return ("cancel", [])
            elif verb == "pick":
                return ("picks", nums[:1])
            elif attempt < 2:
                tts.speak("I didn't catch a number — say it again, or say cancel.")
        return ("none", [])

    # multi-select: keep toggling until he says "done"
    picked: list[int] = []
    misses = 0
    while True:
        verb, nums = _listen_pick(screen)
        if verb == "cancel":
            return ("cancel", [])
        if verb == "repeat":
            _say_question(idx, total, screen)
            continue
        if verb == "done":
            if picked:
                return ("picks", picked)
            tts.speak("Which ones? Say the numbers, or say cancel.")
            continue
        if verb == "pick":
            misses = 0
            for n in nums:
                if n not in picked:
                    picked.append(n)
            lbls = ", ".join(screen["options"][n - 1] for n in picked)
            tts.speak(f"Got {lbls}. Any more, or say done.")
            continue
        # nothing parsed
        misses += 1
        if misses >= 3:
            return ("picks", picked) if picked else ("none", [])
        tts.speak("Say more numbers, or say done." if picked
                  else "Say a number, or say cancel.")


def _answer_loop(pane: str, who: str) -> Result:
    """Walk the live selector question-by-question, then submit on 'submit'."""
    scr = read_screen(pane)
    if scr is None:
        return Result(False, f"I can't read {who}'s pane right now.")
    if scr.get("kind") == "gone":
        clear_pending(pane)
        return Result(False, f"{who} isn't asking anything right now.")

    chosen: list[str] = []
    idx = 0
    # We don't know the total up front from the screen (other tabs' options aren't
    # visible), so count via the tab headers when present; fall back to "more".
    total = _count_questions(pane)
    end_kind = "gone"
    while True:
        scr = read_screen(pane)
        if scr is None:
            return Result(False, "I lost the prompt — you'll have to finish it.")
        if scr.get("kind") in ("submit", "gone"):
            end_kind = scr.get("kind")
            break
        prev_q = scr.get("question")
        verb, picks = _answer_question(idx, total, scr)
        if verb == "cancel":
            cancel(pane)
            clear_pending(pane)
            return Result(True, f"Okay, I left {who}'s questions for you.")
        if verb == "none":
            cancel(pane)
            clear_pending(pane)
            return Result(False, f"I didn't get an answer, so I left {who}'s prompt alone.")
        labels = [scr["options"][p - 1] for p in picks]
        chosen.append(", ".join(labels))
        apply_picks(pane, scr, picks)
        idx += 1
        # Wait for the UI to move on (next question, the Submit review, or — for a
        # lone single-select question — straight to submitted/'gone').
        _wait_screen(pane, lambda s, pq=prev_q: s.get("kind") != "question"
                     or s.get("question") != pq)

    summary = "; ".join(chosen) if chosen else "your selections"
    # Lone single-select prompts auto-submit on the pick (Claude Code shows no
    # review tab), so the screen is already 'gone' — there's nothing left to
    # confirm. Otherwise we're on the Submit review: read back + wait for "submit".
    if end_kind == "gone":
        if chosen:
            clear_pending(pane)
            dispatch._mark_voice(pane)
            return Result(True, f"Done — sent {summary} to {who}.")
        clear_pending(pane)
        return Result(False, f"{who} isn't asking anything right now.")

    tts.speak(f"That's {summary}. Say submit to send it, or cancel.")
    for attempt in range(3):
        heard = stt.listen(onset_timeout=config.ASK_LISTEN_SECS)
        _log(f"confirm heard: {heard!r}")
        if heard and _RE_CANCEL.search(heard):
            cancel(pane)
            clear_pending(pane)
            return Result(True, f"Cancelled {who}'s prompt.")
        if heard and _RE_SUBMIT.search(heard):
            ok = submit(pane)
            clear_pending(pane)
            dispatch._mark_voice(pane)   # voice-initiated -> done-announcer counts it
            return Result(ok, f"Submitted to {who}." if ok
                          else f"I picked everything for {who} but couldn't hit submit.")
        if attempt < 2:
            tts.speak("Say submit to send, or cancel to back out.")
    cancel(pane)
    clear_pending(pane)
    return Result(False, f"No confirmation, so I backed out of {who}'s prompt.")


def _count_questions(pane: str) -> int:
    """How many question tabs the selector shows (☐/☒ headers, excluding Submit).
    Best-effort, for phrasing only ('Question 1' vs not)."""
    text = _capture(pane)
    for ln in text.splitlines():
        if "✔ submit" in ln.lower() or "☐" in ln or "☒" in ln:
            return ln.count("☐") + ln.count("☒")
    return 1


# --------------------------------------------------------------------------- #
# command grammar — "answer <pane>" on the chat hotkey                         #
# --------------------------------------------------------------------------- #
# Verb that opens the answerer. "answer"/"pick"/"select"/"choose"/"respond to".
# Requires either an answer-noun (question/option/prompt) OR a resolvable pending
# pane, so ordinary chat ("answer me this", "pick a number") isn't hijacked.
_RE_ANSWER = re.compile(
    r"^(?:hey\s+)?(?:can you\s+|could you\s+|please\s+|let me\s+|i'?ll\s+|i want to\s+)?"
    r"(?:answer|respond to|pick|choose|select)\s+(?P<rest>.+?)[.!?]*$",
    re.IGNORECASE)
_RE_ANSWER_NOUN = re.compile(r"\b(question|questions|option|options|prompt|prompts|"
                             r"the ask|aska?s)\b", re.IGNORECASE)
# A bare deictic object ("answer this", "pick that", "answer it") — only treated as
# an answer command when exactly one prompt is actually waiting.
_RE_DEICTIC = re.compile(r"^(?:the\s+)?(this|that|it|these|those|them)\b", re.IGNORECASE)


def _resolve_pane(rest: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """From the words after the verb, resolve which pane to answer. Returns
    (pane_id, who, error_say). Strategy: an explicitly named pane wins; else the
    single live pending; else a single pane currently showing a live selector."""
    targets = dispatch.load()
    # strip answer-nouns / fillers so "Build Claude's questions" -> "Build Claude"
    name = re.sub(r"\b(the|these|those|this|that|my|its?|'s)\b", " ", rest, flags=re.IGNORECASE)
    name = _RE_ANSWER_NOUN.sub(" ", name)
    name = re.sub(r"\b(for|on|in|from|to|please|now|right now)\b", " ", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip()
    if name:
        key = dispatch._match_key(name, targets)
        if key is not None:
            pane = targets[key]["pane"]
            return pane, key, None

    pend = _all_pending()
    if len(pend) == 1:
        d = pend[0]
        return d.get("pane"), d.get("who") or "that pane", None
    if len(pend) > 1:
        names = ", ".join(d.get("who") or "a pane" for d in pend)
        return None, None, f"A few panes are waiting: {names}. Which one?"

    # No pending record — scan live panes for an open selector (covers prompts that
    # were already on screen before the hook recorded anything).
    live = _live_selector_panes()
    if len(live) == 1:
        pane = live[0]
        return pane, dispatch.name_for_pane(pane) or "that pane", None
    if len(live) > 1:
        return None, None, "More than one pane is asking — say which one, like answer the Build Claude."
    return None, None, "Nothing's waiting on an answer right now."


def _live_selector_panes() -> list[str]:
    cp = _tmux("list-panes", "-a", "-F", "#{pane_id}")
    out = []
    for pane in cp.stdout.split():
        scr = read_screen(pane)
        if scr is not None and scr.get("kind") in ("question", "submit"):
            out.append(pane)
    return out


def try_answer(text: str) -> Optional[Result]:
    """Detect + run an 'answer the questions' command. Returns a Result (the loop
    speaks the per-question prompts itself; .say is the closing line for the caller
    to speak), or None when this isn't an answer command (caller falls to chat)."""
    if not config.ASK_ENABLE:
        return None
    t = re.sub(r"\s+", " ", (text or "")).strip()
    m = _RE_ANSWER.match(t)
    if not m:
        return None
    rest = m.group("rest").strip()
    targets = dispatch.load()
    has_noun = bool(_RE_ANSWER_NOUN.search(rest))
    names_pane = dispatch._match_key(
        _RE_ANSWER_NOUN.sub(" ", rest).strip(), targets) is not None
    # A bare deictic ("answer this") only counts when exactly one prompt is waiting,
    # so it can't grab a random live pane.
    deictic = bool(_RE_DEICTIC.match(rest)) and (
        len(_all_pending()) == 1 or len(_live_selector_panes()) == 1)
    # Claim ONLY when the utterance is clearly about answering a prompt — it names
    # questions/options/a prompt, names a known pane, or is a deictic with exactly
    # one prompt pending. The mere existence of a pending/live prompt is NOT enough
    # (so "answer me this riddle" / "pick a number" stay ordinary chat).
    if not (has_noun or names_pane or deictic):
        return None
    pane, who, err = _resolve_pane(rest)
    if err is not None:
        return Result(False, err)
    if not pane or not dispatch.pane_alive(pane):
        return Result(False, f"{who or 'That pane'} is gone.")
    _log(f"answering pane={pane} who={who!r}")
    return _answer_loop(pane, who or "that pane")


# --------------------------------------------------------------------------- #
# announce (PreToolUse hook) + clear (PostToolUse hook)                        #
# --------------------------------------------------------------------------- #
_PHRASES = [
    "{who} needs you to pick {opts}.",
    "{who} is asking you {q}.",
    "{who} has {q} for you.",
    "{who} paused to ask you something.",
    "{who} wants you to choose {opts}.",
]


def _busy() -> bool:
    """A nulvoiceagent push-to-talk turn is live — don't barge in (mirrors done._busy)."""
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


def _muted() -> bool:
    """Reuse the done-announcer mute flag: muting the 'Claude is done' voice from
    the taskbar also silences these question nudges (still recorded so he can
    answer)."""
    try:
        return config.DONE_MUTE_FLAG.read_text(encoding="utf-8").strip() == "1"
    except OSError:
        return False


def _announce(who: str, n_questions: int) -> None:
    if _muted() or _busy():
        return
    global _LOCK_FH
    fh = open(config.CACHE_DIR / "ask-announce.lock", "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return
    _LOCK_FH = fh
    q = "a question" if n_questions <= 1 else f"{_num_word(n_questions)} questions"
    opts = "an option" if n_questions <= 1 else "some options"
    phrase = random.choice(_PHRASES).format(who=who, q=q, opts=opts)
    _log(f"announce who={who!r} n={n_questions} -> {phrase!r}")
    tts.speak(phrase)


def _num_word(n: int) -> str:
    return {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(n, str(n))


def announce_payload(payload: dict) -> None:
    """PreToolUse entrypoint: record the pending question for the calling pane and
    speak a short nudge. Runs detached (the hook forks + returns 0) so Claude Code
    isn't blocked while the cue speaks."""
    if not config.ASK_ENABLE:
        return
    pane = os.environ.get("TMUX_PANE")
    questions = (payload.get("tool_input") or {}).get("questions") or []
    if not isinstance(questions, list) or not questions:
        return
    try:
        who = dispatch.name_for_pane(pane) or "Claude"
    except Exception:  # noqa: BLE001 — never break the announce over a name lookup
        who = "Claude"
    _record_pending(pane, who, questions)   # synchronous: always recorded
    # Voice-only mode: only NUDGE for prompts inside a voice-initiated task. A task
    # the user typed in gets no spoken nudge (the pending record still stands, so he
    # can choose to answer it by voice). Peek — don't consume — the mark; the
    # end-of-task done-announcer still needs to consume it. Mirrors done.run().
    if done.is_voice_only() and not done.peek_voice_mark(pane):
        _log(f"announce skip: voice-only mode, task wasn't voice-initiated (pane={pane})")
        return
    # fork a detached child for the speak so the hook returns immediately
    try:
        pid = os.fork()
    except OSError:
        _announce(who, len(questions))
        return
    if pid > 0:
        return
    try:
        os.setsid()
        devnull = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            try:
                os.dup2(devnull, fd)
            except OSError:
                pass
    except OSError:
        pass
    try:
        _announce(who, len(questions))
    finally:
        os._exit(0)


def from_stdin_pending() -> None:
    """CLI seam for the PreToolUse hook: read JSON on stdin, announce + record."""
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        payload = {}
    announce_payload(payload)


def from_stdin_clear() -> None:
    """CLI seam for the PostToolUse hook: the question was answered (by voice OR by
    typing) — drop the pending record for the calling pane."""
    sys.stdin.read()
    clear_pending(os.environ.get("TMUX_PANE"))
