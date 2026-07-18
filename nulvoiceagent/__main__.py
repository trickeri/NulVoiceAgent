"""nulvoiceagent CLI — the central spoken-control agent.

Modes (the triggers map to `chat`, `command` and `route`):
  nulvoiceagent chat          mic -> STT -> brain (conversational, remembers) -> speak
  nulvoiceagent command       mic -> STT -> classify -> run allowlisted action -> speak
  nulvoiceagent route         mic -> STT -> dispatch to a named tmux pane (toggle)

Mic-free testing / scripting:
  nulvoiceagent ask  "<text>"   text -> brain chat -> speak
  nulvoiceagent do   "<text>"   text -> classify -> run action -> speak
  nulvoiceagent dispatch "<t>"  text -> route to a named pane (name/tell/list/forget)
  nulvoiceagent targets         list named panes
  nulvoiceagent panes-reset     clear ALL pane names (registry + every pane title)
  nulvoiceagent panes-prune     drop dead panes from the registry (tmux pane-close hook)
  nulvoiceagent say  "<text>"   just speak text (TTS check)
  nulvoiceagent viewer-say --user <name> --source bits|points [--bits N] "<text>"  paying viewer's chat message, read aloud (nul-chat-hub)
  nulvoiceagent claude-done     Claude Code Stop hook: announce done + offer summary (reads hook JSON on stdin)
  nulvoiceagent codex-done      Codex Stop hook: announce done + offer summary (reads hook JSON on stdin)
  nulvoiceagent done-mute [x]   mute/unmute the "agent is done" voice; x = toggle(default)|on|off|status
  nulvoiceagent done-voice-only [x]  announce only VOICE-started tasks (typed tasks stay silent); x = toggle(default)|on|off|status
  nulvoiceagent newtopic        forget the chat conversation context

Global: --brain codex|claude overrides NULVOICEAGENT_BRAIN for this run.
"""
from __future__ import annotations
import os
import signal
import sys
import time

from . import actions, askanswer, brain, config, dispatch, done, intents, state, stt, tts, viewertts

_LOG = config.CACHE_DIR / "nulvoiceagent.log"
# PID of an in-progress manual (toggle) recording. Any hotkey press while this is
# live ends + processes that recording instead of starting a new one.
_REC_PID = config.CACHE_DIR / "recording.pid"
# PGID of an in-progress *turn* (past recording: thinking or speaking). A voice
# hotkey pressed while this is live kills the whole group — including the pw-play
# that's speaking — so the agent shuts up at once and we barge in with a fresh
# recording.
_TURN_PGID = config.CACHE_DIR / "turn.pgid"

# Seconds the last STT transcribe took, set by the recorders and read by the
# brain-path `timing:` log line. 0.0 for text-in (`ask`/`do`) runs.
_STT_SECS = 0.0


def _is_nulvoiceagent(pid: int) -> bool:
    """True if `pid` is alive AND is actually one of our processes. Guards against
    PID/PGID reuse: a crashed recorder can leave a stale pidfile whose number gets
    recycled by an unrelated long-lived process — without this check every hotkey
    press would see that innocent PID as "a recording in progress", fire a signal
    at it, and silently do nothing (the toggle looking "completely broken")."""
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read()
    except OSError:
        return False
    return b"nulvoiceagent" in cmd


def _recording_pid() -> int | None:
    try:
        pid = int(_REC_PID.read_text())
    except (OSError, ValueError):
        return None
    if _is_nulvoiceagent(pid):     # alive AND really our recorder
        return pid
    try:                        # stale (dead, or PID reused by another process)
        _REC_PID.unlink()
    except OSError:
        pass
    return None


def _stop_active_recording() -> bool:
    """If a manual recording is in progress, signal it to stop+process. Returns
    True if one was stopped (so the caller should do nothing else)."""
    pid = _recording_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGUSR1)
        except OSError:
            return False
        return True
    return False


def _mark_turn() -> None:
    """Record this process group as the active turn so a later hotkey can
    interrupt our reasoning/speech (barge-in)."""
    try:
        _TURN_PGID.write_text(str(os.getpgrp()))
    except OSError:
        pass


def _clear_turn() -> None:
    try:
        _TURN_PGID.unlink()
    except OSError:
        pass


def _interrupt_active_turn() -> bool:
    """If a previous turn is still thinking/speaking, kill its whole process
    group (Python + pw-play + any synth helpers) so it stops talking at once.
    Returns True if one was interrupted. The taskbar face reverts on its own via
    the `talking` until-backstop, so no cleanup is needed here."""
    try:
        pgid = int(_TURN_PGID.read_text())
    except (OSError, ValueError):
        return False
    if pgid == os.getpgrp():
        return False                 # never signal our own group
    # The group leader's PID equals the PGID; verify it's really our turn (not a
    # recycled PGID) before SIGTERMing the whole group — same reuse guard as the
    # recording pidfile, so we never kill an innocent group.
    if not _is_nulvoiceagent(pgid):
        _clear_turn()                # stale or reused pgidfile
        return False
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return False
    _clear_turn()
    return True


def _chat_toggle() -> None:
    """Press to start; press again (or any voice hotkey) to stop + process.
    Pressing while a previous reply is still thinking/speaking interrupts it
    (kills the speech) and starts a fresh recording — barge-in."""
    if _stop_active_recording():
        return
    _interrupt_active_turn()         # shut up an in-progress reply, then listen
    _REC_PID.write_text(str(os.getpid()))
    tts.cue("start")
    state.listening()
    try:
        # Republish the listening state with live mic level so the taskbar voice
        # indicator's bars dance the same as they do for dictation.
        wav = stt.record_manual(level_cb=lambda v: state.listening(level=v))
    finally:
        try:
            _REC_PID.unlink()
        except OSError:
            pass
    tts.cue("stop")
    _mark_turn()
    try:
        _handle_chat(_transcribe(wav))
    finally:
        _clear_turn()


def _transcribe(wav) -> str:
    """STT a recording, recording how long it took for the per-turn timing line."""
    global _STT_SECS
    if not wav:
        _STT_SECS = 0.0
        return ""
    s = time.monotonic()
    text = stt.transcribe(wav)
    _STT_SECS = time.monotonic() - s
    return text


def _route_toggle() -> None:
    """Like chat: press to start a (long-form) recording, press again (or any
    voice hotkey) to stop + route it to a named pane. No silence cutoff."""
    if _stop_active_recording():
        return
    _interrupt_active_turn()
    _REC_PID.write_text(str(os.getpid()))
    tts.cue("start")
    state.listening()
    try:
        wav = stt.record_manual(config.ROUTE_MAX_SECS,
                                level_cb=lambda v: state.listening(level=v))
    finally:
        try:
            _REC_PID.unlink()
        except OSError:
            pass
    tts.cue("stop")
    _mark_turn()
    try:
        _handle_route(_transcribe(wav))
    finally:
        _clear_turn()


def _log(line: str) -> None:
    """Echo to stderr AND append to the log, so hotkey-triggered runs (whose
    stderr is invisible) are observable: `tail -f ~/.cache/nulvoiceagent/nulvoiceagent.log`."""
    print(line, file=sys.stderr)
    try:
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')}  {line}\n")
    except OSError:
        pass


def _handle_chat(text: str, reset: bool = False) -> None:
    text = (text or "").strip()
    if not text:
        _log("(nothing heard)")
        state.idle()
        return
    _log(f"you: {text}")
    # "Summarize" right after a done announcement -> summarize that
    # completion (decisions/gotchas/architecture in 2-3 sentences) instead of
    # going to Hermes, which has no idea what to summarize. Falls through to chat
    # if there's no fresh completion (so "summarize this article" still works).
    if done.wants_summary_request(text):
        summary = done.make_summary()
        if summary:
            _log(f"agent: {summary}")
            tts.speak(summary)
            return
    # "answer <pane>" / "pick the options" -> walk a live AskUserQuestion prompt by
    # voice (reads each question's options, listens for numbers, drives the
    # selector, submits on "submit"). The loop speaks the per-question prompts
    # itself; the returned .say is the closing line. Checked before route/brain so
    # it isn't mistaken for a dispatch or sent to chat.
    answered = askanswer.try_answer(text)
    if answered is not None:
        _log(f"answer: ok={answered.ok}  say: {answered.say!r}")
        tts.speak(answered.say)
        return
    #   1. a confident "name this …" / "tell <named pane> …" routes to a pane;
    #   2. otherwise Hermes chats AND may trigger one allowlisted fork/OS action
    #      via a [[do: <name>]] directive (gated by actions.run);
    #   3. plain conversation just speaks.
    # (Meta+Escape stays the deterministic, Hermes-free command path.)
    routed = dispatch.try_route(text)
    if routed is not None:
        _log(f"route: ok={routed.ok}  say: {routed.say!r}")
        tts.speak(routed.say)
        return
    # Parameterized intents (deterministic, carry args the classifier can't) —
    # e.g. "new Krita document 1920 wide 1080 high". Checked before the brain.
    intent = intents.match(text)
    if intent is not None:
        intents.run(intent)
        _log(f"intent: {' '.join(intent.argv)}")
        tts.speak(intent.say)
        return
    state.thinking()
    t0 = time.monotonic()
    try:
        reply, action, ask = brain.chat_act(text, actions.names(), reset=reset)
    except Exception as e:  # noqa: BLE001 — speak failures, don't crash the mic flow
        _log(f"brain error: {e}")
        tts.speak("Sorry, my brain isn't reachable right now.")
        return
    t_brain = time.monotonic() - t0
    if action:
        ran = actions.run(action)
        _log(f"action: {action}  ran={ran}")
        if not ran:
            reply = reply or f"I couldn't run {action}."
    if ask:
        pane, msg = ask
        res = dispatch.send_named(pane, msg)
        _log(f"ask: {pane!r} <- {msg!r}  -> {None if res is None else (res.ok, res.say)}")
        if res is None:
            reply = (reply or "") + f" (But I couldn't find a pane named {pane}.)"
    _log(f"agent: {reply}")
    tts.speak(reply or "Done.")
    _log(f"timing: stt={_STT_SECS:.2f}s brain={t_brain:.2f}s "
         f"tts_synth={tts.last_synth_secs:.2f}s")


def _handle_command(text: str) -> None:
    text = (text or "").strip()
    if not text:
        _log("(nothing heard)")
        state.idle()
        return
    _log(f"you: {text}")
    # Parameterized intents first (args the name-only classifier can't carry).
    intent = intents.match(text)
    if intent is not None:
        intents.run(intent)
        _log(f"intent: {' '.join(intent.argv)}")
        tts.speak(intent.say)
        return
    names = actions.names()
    state.thinking()
    t0 = time.monotonic()
    try:
        result = brain.classify(text, names)
    except Exception as e:  # noqa: BLE001
        _log(f"brain error: {e}")
        tts.speak("Sorry, I couldn't reach my brain to run that.")
        return
    t_brain = time.monotonic() - t0
    action, say = result.get("action", "none"), result.get("say", "")
    _log(f"action: {action}  say: {say!r}")
    if action != "none":
        if not actions.run(action):
            say = say or f"I don't have an action called {action}."
    tts.speak(say or "Done.")
    _log(f"timing: stt={_STT_SECS:.2f}s brain={t_brain:.2f}s "
         f"tts_synth={tts.last_synth_secs:.2f}s")


def _handle_route(text: str) -> None:
    text = (text or "").strip()
    if not text:
        _log("(nothing heard)")
        state.idle()
        return
    _log(f"you: {text}")
    state.thinking()
    res = dispatch.handle(text)
    _log(f"route: ok={res.ok}  say: {res.say!r}")
    tts.speak(res.say)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 0

    # optional --brain override (sets the chat brain; also the classify brain
    # when it's a backend that supports classification)
    if "--brain" in argv:
        i = argv.index("--brain")
        try:
            b = argv[i + 1].strip().lower()
            config.CHAT_BRAIN = b
            if b in ("codex", "claude"):
                config.CMD_BRAIN = b
            del argv[i:i + 2]
        except IndexError:
            print("--brain needs a value (hermes|codex|claude)", file=sys.stderr)
            return 2

    cmd, rest = argv[0], argv[1:]
    text = " ".join(rest)

    # Voice modes run as their own process group so an interrupting hotkey can
    # take the whole group (Python + pw-play) down in one killpg — barge-in.
    if cmd in ("chat", "command", "route"):
        try:
            os.setpgrp()
        except OSError:
            pass

    if cmd == "chat":
        _chat_toggle()
    elif cmd == "command":
        # any hotkey press during a live chat recording ends it instead
        if not _stop_active_recording():
            _interrupt_active_turn()       # barge in on a still-speaking reply
            state.listening()
            heard = stt.listen()
            _mark_turn()
            try:
                _handle_command(heard)
            finally:
                _clear_turn()
    elif cmd == "route":
        _route_toggle()
    elif cmd == "ask":
        _handle_chat(text)
    elif cmd == "do":
        _handle_command(text)
    elif cmd == "dispatch":
        _handle_route(text)
    elif cmd == "targets":
        if rest and rest[0] == "forget":
            name = " ".join(rest[1:])
            print(f"forgot {name}" if dispatch.forget(name) else f"no target named {name}")
        else:
            tgts = dispatch.load()
            if not tgts:
                print("(no named panes)")
            for n, d in tgts.items():
                print(f"{n:24} -> {d.get('pane')}")
    elif cmd == "panes-reset":
        n = dispatch.reset()
        print(f"cleared {n} pane name(s)")
    elif cmd == "panes-prune":
        # Called by the tmux pane-exited hook (silent, no TTS). Reconciles the
        # registry against live panes and drops any that are gone.
        dropped = dispatch.prune()
        if dropped:
            print(f"pruned: {', '.join(dropped)}")
    elif cmd == "say":
        tts.speak(text)
    elif cmd == "viewer-say":
        # A paying viewer's chat message, read aloud (bits cheer / channel-point
        # redeem). Spawned by the nul-chat-hub daemon, one per event:
        #   viewer-say --user <name> --source bits|points [--bits N] "<raw text>"
        user, source, bits = "someone", "bits", None
        words = list(rest)
        parsed: list[str] = []
        i = 0
        while i < len(words):
            w = words[i]
            if w == "--user" and i + 1 < len(words):
                user = words[i + 1]; i += 2
            elif w == "--source" and i + 1 < len(words):
                source = words[i + 1]; i += 2
            elif w == "--bits" and i + 1 < len(words):
                try:
                    bits = int(words[i + 1])
                except ValueError:
                    bits = None
                i += 2
            else:
                parsed.append(w); i += 1
        viewertts.say(user, " ".join(parsed), source=source, bits=bits)
    elif cmd == "claude-done":
        # Driven by a Claude Code Stop hook: reads the hook JSON on stdin, then
        # forks a detached child that speaks the completion + offers a summary.
        done.from_stdin()
    elif cmd == "codex-done":
        # Driven by a Codex Stop hook: reads the hook JSON on stdin, forks a
        # detached child for speech, then emits {} so Codex treats the hook as
        # advisory and never blocks the completed turn.
        done.from_stdin(default_who="Codex", codex_noop=True)
    elif cmd == "ask-pending":
        # Claude Code PreToolUse hook (matcher AskUserQuestion): reads the hook JSON
        # on stdin, records the pending question for the pane + speaks a nudge.
        askanswer.from_stdin_pending()
    elif cmd == "ask-clear":
        # Claude Code PostToolUse hook (matcher AskUserQuestion): the prompt was
        # answered — drop the pending record for the calling pane.
        askanswer.from_stdin_clear()
    elif cmd == "answer":
        # Text seam for the answer grammar (mic-free testing of detection/resolve).
        res = askanswer.try_answer(text)
        if res is None:
            print("(not an answer command)")
        else:
            print(f"ok={res.ok} say={res.say!r}")
    elif cmd == "ask-test":
        # Hidden driver check: `ask-test <pane> <picks-for-q1> <picks-for-q2> …`
        # where each picks arg is comma-separated 1-based option numbers. Drives the
        # live selector with no mic/TTS, then submits. For verifying the keystroke
        # contract against a real prompt.
        if not rest:
            print("usage: ask-test <pane> <q1picks> [q2picks ...]")
            return 2
        pane = rest[0]
        steps = [[int(x) for x in a.split(",") if x.strip()] for a in rest[1:]]
        for picks in steps:
            scr = askanswer.read_screen(pane)
            print(f"screen: {scr}")
            if not scr or scr.get("kind") != "question":
                print("not on a question; stopping")
                break
            askanswer.apply_picks(pane, scr, picks)
            askanswer._wait_screen(pane, lambda s: s.get("kind") != "question"
                                   or s.get("question") != scr.get("question"))
        print(f"final screen: {askanswer.read_screen(pane)}")
        print("submit:", askanswer.submit(pane))
    elif cmd == "done-mute":
        # Toggle/set the runtime mute for completion announcements. Called by the
        # taskbar agent-state context menu; prints the resulting state.
        arg = (rest[0].strip().lower() if rest else "toggle")
        if arg in ("status", "state"):
            new = done.is_muted()
        elif arg in ("on", "mute", "1", "true"):
            new = done.set_muted(True)
        elif arg in ("off", "unmute", "0", "false"):
            new = done.set_muted(False)
        else:  # toggle
            new = done.set_muted(not done.is_muted())
        print("muted" if new else "unmuted")
    elif cmd == "done-voice-only":
        # Toggle/set "announce only voice-initiated tasks". Called by the taskbar
        # agent-state context menu; prints the resulting state.
        arg = (rest[0].strip().lower() if rest else "toggle")
        if arg in ("status", "state"):
            new = done.is_voice_only()
        elif arg in ("on", "1", "true"):
            new = done.set_voice_only(True)
        elif arg in ("off", "0", "false"):
            new = done.set_voice_only(False)
        else:  # toggle
            new = done.set_voice_only(not done.is_voice_only())
        print("voice-only" if new else "all-tasks")
    elif cmd == "newtopic":
        brain.reset_session()
        tts.speak("Okay, fresh start. What's on your mind?")
    else:
        print(f"unknown command: {cmd}\n{__doc__}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
