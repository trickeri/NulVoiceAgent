"""Passive listening — Hermes occasionally eavesdrops on normal mic speech.

`nulvoiceagent passive` runs a capture loop (meant to live in
nulvoiceagent-passive.service, toggled from the avatar's right-click menu).
It overhears Trickery talking on the mic OUTSIDE any push-to-talk turn —
addressing Twitch chat, thinking out loud — and occasionally interjects a
short in-character remark. Strictly conversational: nothing here touches
intents/actions/dispatch, so overheard speech can never fire a command.

Per overheard utterance, cheapest gates first:
  1. discard if a hotkey turn was live at ANY point from just before the capture
     started to now — a push-to-talk turn (recording.pid / turn.pgid), a
     non-idle avatar state (command-mode listening records without a pidfile;
     also her own TTS on the speakers), or a voicechat DICTATION turn. See
     _BusyWatch: a point-in-time check isn't enough, because a hotkey pressed
     mid-capture means the audio we just recorded is his hotkey speech;
  2. discard blips shorter than PASSIVE_MIN_SECS;
  3. skip while the PASSIVE_COOLDOWN_SECS since the last brain call runs —
     the hard cost cap;
  4. roll PASSIVE_PROB — most utterances are never even considered;
  5. STT; skip if under PASSIVE_MIN_WORDS;
  6. brain.eavesdrop() -> either PASS (she holds her tongue) or a remark,
     spoken through the normal Miku/Aurinex TTS path.

The think+speak phase runs in a forked child with its own process group,
registered in turn.pgid — so a voice hotkey barges in on an interjection
exactly like it does on a chat reply (kills the child, daemon survives). The
dictation key isn't a voice hotkey and can't signal us, so the child watches
for it itself and barges in on its own speech (_speak_guard).
"""
from __future__ import annotations

import fcntl
import json
import os
import random
import signal
import sys
import threading
import time

from . import brain, config, state, stt, tts

_LOG = config.CACHE_DIR / "nulvoiceagent.log"
_LOCK_FH = None  # module-held so the flock lives as long as the daemon


def _log(line: str) -> None:
    print(line, file=sys.stderr)
    try:
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')}  {line}\n")
    except OSError:
        pass


def _single_instance() -> bool:
    global _LOCK_FH
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _LOCK_FH = open(config.PASSIVE_LOCK_FILE, "w")
    try:
        fcntl.flock(_LOCK_FH.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    _LOCK_FH.write(str(os.getpid()))
    _LOCK_FH.flush()
    return True


def _voice_turn_live() -> bool:
    """A push-to-talk recording or a spoken reply is in flight (same pidfile
    checks as done._busy, minus the import cycle)."""
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


def _avatar_state() -> str:
    """Current published state (idle/listening/thinking/talking), with the same
    60s stale-guard the taskbar face uses — a crashed writer mustn't wedge
    passive listening off forever."""
    try:
        o = json.loads((config.CACHE_DIR / "state.json").read_text())
    except (OSError, ValueError):
        return "idle"
    s = str(o.get("state", "idle"))
    if s != "idle" and time.time() - float(o.get("ts", 0)) > 60:
        return "idle"
    return s


def _dictation_live() -> bool:
    """A voicechat DICTATION turn is in flight (Meta+CapsLock). voicechat is a
    separate daemon with its own capture — it publishes
    ~/.cache/voicechat/status.json as 'listening' the moment the key is pressed,
    then 'processing'. Nothing about that turn is ours to overhear: the words go
    into whatever window he's typing into, and anything she says over them gets
    transcribed into his dictation."""
    try:
        o = json.loads(config.VOICECHAT_STATUS_FILE.read_text())
    except (OSError, ValueError):
        return False
    if str(o.get("state", "idle")) not in config.VOICECHAT_BUSY_STATES:
        return False
    # Stale-guard a wedged 'listening' (crashed daemon) so passive listening
    # can't be switched off permanently by a stale file.
    try:
        age = time.time() - float(o.get("ts", 0))
    except (TypeError, ValueError):
        return False
    return age <= config.PASSIVE_DICTATION_STALE_SECS


def _busy_now() -> bool:
    """Any hotkey turn — ours or voicechat's dictation — is live right now."""
    return _voice_turn_live() or _avatar_state() != "idle" or _dictation_live()


class _BusyWatch:
    """Samples the hotkey/dictation state in the background so we can ask not
    just "is a turn live?" but "was one live at any point since T?".

    That distinction is the whole fix for dictation bleed. The capture call
    blocks for the length of an utterance, so a point-in-time check before and
    after it misses the case that actually bites: passive opens the mic, THEN
    Trickery presses dictation and talks, and the dictation finishes inside our
    capture window. Both end-point checks see 'idle' and his dictated sentence
    gets treated as overheard musing — the interjection then talks over his next
    dictation. Sampling throughout closes that window."""

    def __init__(self) -> None:
        self._last_busy = 0.0      # monotonic; 0 = not seen busy yet
        self._busy = False
        self._lock = threading.Lock()

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            try:
                busy = _busy_now()
            except Exception:      # a transient stat/parse hiccup isn't fatal
                busy = False
            with self._lock:
                self._busy = busy
                if busy:
                    self._last_busy = time.monotonic()
            time.sleep(config.PASSIVE_BUSY_POLL_SECS)

    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def busy_since(self, mark: float) -> bool:
        """Was a hotkey turn live at any sampled moment since `mark`
        (monotonic), including the PASSIVE_BUSY_GRACE_SECS run-up to it?"""
        with self._lock:
            busy, last = self._busy, self._last_busy
        return busy or (last > 0.0 and last >= mark - config.PASSIVE_BUSY_GRACE_SECS)


def _cooldown_left() -> float:
    try:
        last = float(config.PASSIVE_LAST_FILE.read_text().strip())
    except (OSError, ValueError):
        return 0.0
    return max(0.0, config.PASSIVE_COOLDOWN_SECS - (time.time() - last))


def _stamp_cooldown() -> None:
    try:
        config.PASSIVE_LAST_FILE.write_text(str(time.time()))
    except OSError:
        pass


def _speak_guard(turn) -> None:
    """Abort our own interjection the moment a dictation turn starts.

    A voice hotkey barges in by killing turn.pgid, but the dictation key belongs
    to voicechat and can't signal us — so we watch for it and do to ourselves
    exactly what the hotkey would: kill this child's process group, taking down
    python + pw-play + any synth helper at once. Without this, an interjection
    that started a moment before Meta+CapsLock keeps playing into his dictation.
    Runs in the forked child (its own session/pgroup), so the daemon is safe."""
    while True:
        time.sleep(config.PASSIVE_BUSY_POLL_SECS)
        try:
            if not _dictation_live():
                continue
        except Exception:
            continue
        _log("passive: aborting interjection (dictation started)")
        state.idle()
        try:
            turn.unlink()
        except OSError:
            pass
        os.killpg(os.getpgrp(), signal.SIGTERM)
        return


def _interject(text: str) -> None:
    """Think + (maybe) speak in a forked child with its own pgroup, registered
    as the active turn so a hotkey press can barge in on it (killing the child,
    not the daemon). The parent waits, so capture never runs over her speech."""
    pid = os.fork()
    if pid:
        try:
            os.waitpid(pid, 0)
        except OSError:
            pass
        return
    # child
    os.setsid()
    turn = config.CACHE_DIR / "turn.pgid"
    try:
        turn.write_text(str(os.getpgrp()))
    except OSError:
        pass
    try:
        state.thinking()
        t0 = time.monotonic()
        reply = brain.eavesdrop(text)
        secs = time.monotonic() - t0
        if brain.is_pass(reply):
            _log(f"passive: PASS  (brain={secs:.2f}s)")
            state.idle()
        elif _dictation_live():
            # He started dictating during the (multi-second) brain call. Drop the
            # remark rather than queue it: speaking now lands her voice inside
            # his transcript, and by the time he's done the remark is stale.
            _log(f"passive: dropped (dictation started during brain={secs:.2f}s): "
                 f"{reply!r}")
            state.idle()
        else:
            _log(f"passive: say: {reply!r}  (brain={secs:.2f}s)")
            threading.Thread(target=_speak_guard, args=(turn,),
                             daemon=True).start()
            tts.speak(reply)          # publishes talking/idle states itself
    except Exception as e:            # never let one bad turn kill the child oddly
        _log(f"passive: error: {e}")
        state.idle()
    finally:
        try:
            turn.unlink()
        except OSError:
            pass
    os._exit(0)


def run() -> int:
    if not _single_instance():
        print("passive: already running", file=sys.stderr)
        return 1
    _log(f"passive: up  (prob={config.PASSIVE_PROB:g} "
         f"cooldown={config.PASSIVE_COOLDOWN_SECS:g}s "
         f"min={config.PASSIVE_MIN_SECS:g}s words>={config.PASSIVE_MIN_WORDS})")
    watch = _BusyWatch()
    watch.start()
    while True:
        # Don't even open the mic while a hotkey turn / her own voice is live.
        if watch.busy():
            time.sleep(1.0)
            continue
        mark = time.monotonic()
        wav = stt.record(
            max_secs=config.PASSIVE_ONSET_WAIT + config.PASSIVE_MAX_SECS,
            onset_timeout=config.PASSIVE_ONSET_WAIT,
            trail_silence=config.PASSIVE_TRAIL)
        if not wav:
            continue                  # onset window elapsed silently; recycle
        dur = state.wav_duration(wav)
        # Discard if a hotkey turn overlapped the capture AT ALL — a push-to-talk
        # press, command-mode listening, her own reply, or a voicechat dictation.
        # Whatever we just recorded was said ON a key, so it isn't ours to hear.
        if watch.busy_since(mark):
            _log(f"passive: heard {dur:.1f}s -> skip (hotkey turn)")
            continue
        if dur < config.PASSIVE_MIN_SECS:
            continue                  # a cough, a "yep" — not worth logging
        left = _cooldown_left()
        if left > 0:
            _log(f"passive: heard {dur:.1f}s -> skip (cooldown {left:.0f}s left)")
            continue
        if random.random() >= config.PASSIVE_PROB:
            _log(f"passive: heard {dur:.1f}s -> skip (dice)")
            continue
        text = stt.transcribe(wav).strip()
        if len(text.split()) < config.PASSIVE_MIN_WORDS:
            _log(f"passive: heard {dur:.1f}s -> skip (short: {text!r})")
            continue
        _stamp_cooldown()             # every brain call pays the cooldown
        _log(f"passive: overheard: {text}")
        _interject(text)
    return 0
