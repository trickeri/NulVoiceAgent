"""nulvoiceagent configuration — all env-overridable, sane local defaults.

The pieces this orchestrator wires together (all run locally on this box):
  - NulSpeech2Text STT  :48450  (POST /inference, multipart wav)
  - kokoromodel    TTS  :48490  (POST /speak -> audio/wav)
  - the "brain": Hermes (default chat), codex or claude CLIs, all headless.

Every value below can be overridden with a NULVOICEAGENT_* environment variable.
The things that make the agent *yours* — its voice and its personality — live in
user config under ~/.config/nulvoiceagent/, never in code, so you can shape it
without touching this repo (see the README).
"""
from __future__ import annotations
import os
import re
from pathlib import Path

# --- paths -------------------------------------------------------------------
REPO_DIR = Path(__file__).resolve().parent.parent          # codex cwd (AGENTS.md persona lives here)
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "nulvoiceagent"
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "nulvoiceagent"
CHAT_SESSION_FILE = CACHE_DIR / "chat-session"             # persisted brain session id
ACTIONS_FILE = CONFIG_DIR / "actions.conf"                 # the command-mode allowlist (data, not code)
APPS_DIR = CONFIG_DIR / "apps"                             # per-app command allowlists: apps/<app>.conf
TARGETS_FILE = CONFIG_DIR / "targets.json"                 # route-mode named tmux panes (data)
# Optional persona override. If present, its text is used as the spoken-assistant
# system prompt; otherwise the neutral template in the repo's AGENTS.md is used.
PERSONA_FILE = CONFIG_DIR / "persona.md"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- service endpoints -------------------------------------------------------
STT_URL = os.environ.get("NULVOICEAGENT_STT_URL", "http://127.0.0.1:48450/inference")
KOKORO_URL = os.environ.get("NULVOICEAGENT_KOKORO_URL", "http://127.0.0.1:48490/speak")

# --- brain selection ---------------------------------------------------------
# Two independent brains:
#   CHAT_BRAIN — conversation / "explore ideas". Default "hermes": the Hermes
#     agent (running on a provider of your choice) so you get its persistent
#     memory + skills, including handing tasks off to named tmux panes.
#     Alternatives: "codex" (thin codex exec) | "claude".
#   CMD_BRAIN  — command-mode classifier. Default "codex": a strict schema-
#     constrained classifier that only ever picks an allowlisted action name.
#     ("hermes" isn't used here — command mode must stay deterministic.)
CHAT_BRAIN = os.environ.get("NULVOICEAGENT_CHAT_BRAIN", "hermes").strip().lower()
CMD_BRAIN = os.environ.get("NULVOICEAGENT_CMD_BRAIN", "codex").strip().lower()
# Fallback brain when the primary is out of tokens / over its usage limit for the
# session (or otherwise can't answer). Default "claude" — headless `claude -p`, a
# DIFFERENT provider, so when the primary session runs dry we retry the SAME turn
# on the fallback instead of just saying "my brain isn't reachable". Empty = no
# fallback (surface the error). Applies to chat + command.
CHAT_FALLBACK = os.environ.get("NULVOICEAGENT_CHAT_FALLBACK", "claude").strip().lower()
CMD_FALLBACK = os.environ.get("NULVOICEAGENT_CMD_FALLBACK", "claude").strip().lower()
# Optional per-mode model overrides. Empty = let the CLI use its own default.
CHAT_MODEL = os.environ.get("NULVOICEAGENT_CHAT_MODEL", "").strip()
CMD_MODEL = os.environ.get("NULVOICEAGENT_CMD_MODEL", "").strip()
# Reasoning effort for the codex-backed paths (command-mode classify + the codex
# chat backend). "low" is a good default for short spoken replies / a pick-one
# classify. Passed as `-c model_reasoning_effort=<x>`. Empty = leave CLI default.
CMD_REASONING = os.environ.get("NULVOICEAGENT_CMD_REASONING", "low").strip()
CHAT_REASONING = os.environ.get("NULVOICEAGENT_CHAT_REASONING", "low").strip()

# --- mic capture -------------------------------------------------------------
# PipeWire/PulseAudio source name (parec --device=). Empty (default) = the system
# default input. Set NULVOICEAGENT_SOURCE to pin a specific mic (find names with
# `pactl list sources short`).
MIC_SOURCE = os.environ.get("NULVOICEAGENT_SOURCE", "").strip()
SAMPLE_RATE = 16000  # STT wants 16 kHz mono

# Noise-gate endpointer (dBFS) for the auto-endpointed `command` mode. Hysteresis:
# must SUSTAIN above OPEN for GATE_ATTACK to start; then end the turn once it holds
# below CLOSE for GATE_TRAIL. The trailing counter is leaky (GATE_LEAK) so a stray
# spike doesn't hold the gate open forever. Tune toward -18/-24 if a noisy room
# false-opens, or toward -30 for a quiet one.
GATE_OPEN_DB = float(os.environ.get("NULVOICEAGENT_GATE_OPEN_DB", "-20"))
GATE_CLOSE_DB = float(os.environ.get("NULVOICEAGENT_GATE_CLOSE_DB", "-28"))
GATE_ATTACK = float(os.environ.get("NULVOICEAGENT_GATE_ATTACK", "0.06"))
GATE_TRAIL = float(os.environ.get("NULVOICEAGENT_GATE_TRAIL", "1.0"))
GATE_LEAK = float(os.environ.get("NULVOICEAGENT_GATE_LEAK", "2.0"))
GATE_ONSET_TIMEOUT = float(os.environ.get("NULVOICEAGENT_GATE_ONSET_TIMEOUT", "6.0"))
GATE_MAX_SECS = float(os.environ.get("NULVOICEAGENT_GATE_MAX_SECS", "15.0"))
# Manual (toggle) chat recording: no silence endpoint, this is just a safety cap
# so a forgotten session can't record forever.
MANUAL_MAX_SECS = float(os.environ.get("NULVOICEAGENT_MANUAL_MAX_SECS", "120.0"))

# --- voice / profile ---------------------------------------------------------
# The spoken voice + delivery come from the ACTIVE profile at
# ~/.config/nulvoiceagent/profiles/<active>.conf — a simple KEY=value file. Keep
# as many profiles as you like and switch by writing the profile name into the
# sibling `active` file. Every key is also overridable with an env var. Ships
# neutral: a stock Kokoro voice, no effects.
def _active_profile() -> dict:
    base = CONFIG_DIR / "profiles"
    out: dict[str, str] = {}
    try:
        name = (CONFIG_DIR / "active").read_text(encoding="utf-8").strip() or "default"
        for line in (base / f"{name}.conf").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


_PROFILE = _active_profile()

# Optional start/stop cue sounds, played to the default sink when a manual
# recording begins/ends. Empty (default) = silent. Set SOUND_START / SOUND_STOP in
# the active profile (or NULVOICEAGENT_SOUND_START / _STOP) to any short wav/mp3.
CUES = os.environ.get("NULVOICEAGENT_CUES", _PROFILE.get("CUES", "1")).strip() not in ("0", "false", "")
CUE_START = os.environ.get("NULVOICEAGENT_SOUND_START", _PROFILE.get("SOUND_START", "")).strip()
CUE_STOP = os.environ.get("NULVOICEAGENT_SOUND_STOP", _PROFILE.get("SOUND_STOP", "")).strip()

# A stock Kokoro voice by default (see the kokoromodel voice list). Override per
# profile (VOICE=...) or with NULVOICEAGENT_VOICE.
VOICE = os.environ.get("NULVOICEAGENT_VOICE", _PROFILE.get("VOICE", "af_heart")).strip()
# Synthesis speed multiplier (kokoromodel `speed`). Blank = server default.
SPEED = os.environ.get("NULVOICEAGENT_SPEED", _PROFILE.get("SPEED", "")).strip()
# Optional semitone pitch shift applied by kokoromodel. Blank = none.
PITCH = os.environ.get("NULVOICEAGENT_PITCH", _PROFILE.get("PITCH", "")).strip()
# Optional: play the agent's voice INTO a named PipeWire sink — e.g. your own
# effects chain. Empty (default) = the system default sink, dry. Falls back to the
# default sink if the named sink isn't present.
TTS_SINK = os.environ.get("NULVOICEAGENT_TTS_SINK", _PROFILE.get("TTS_SINK", "")).strip()

# Optional harmony LAYER: mix in a second copy of the voice pitched LAYER_PITCH
# semitones (e.g. -7 = a perfect fifth below) at LAYER_GAIN, adding weight under
# the lead. Needs `rubberband` (pitch) + `ffmpeg` (EQ + mix); degrades to an
# unfiltered python mix without ffmpeg, and to no layer without rubberband. Blank
# LAYER_PITCH (default) = no layer. Requires a profile that opts in (LAYER_PITCH=…).
LAYER_PITCH = os.environ.get("NULVOICEAGENT_LAYER_PITCH", _PROFILE.get("LAYER_PITCH", "")).strip()
LAYER_GAIN = os.environ.get("NULVOICEAGENT_LAYER_GAIN", _PROFILE.get("LAYER_GAIN", "0.7")).strip()
# Complementary EQ split between the two layers (Hz; blank = no filter):
#   LAYER_LEAD_HPF — highpass the main (lead) so the low copy owns the lows
#   LAYER_DOWN_LPF — lowpass the pitched-down copy so the lead owns the highs
LAYER_LEAD_HPF = os.environ.get("NULVOICEAGENT_LAYER_LEAD_HPF", _PROFILE.get("LAYER_LEAD_HPF", "400")).strip()
LAYER_DOWN_LPF = os.environ.get("NULVOICEAGENT_LAYER_DOWN_LPF", _PROFILE.get("LAYER_DOWN_LPF", "4000")).strip()

# --- "agent is done" announcer (Claude/Codex Stop hooks) ---------------------
# Speak a short confirmation when a Claude Code or Codex turn finishes, then
# listen briefly for "summarize" and, if heard, speak a 2-3 sentence summary of
# what the agent said.
DONE_ENABLE = os.environ.get("NULVOICEAGENT_DONE_ENABLE", "1").strip() not in ("0", "false", "")
# How long to wait for you to start saying "summarize" after the confirmation
# (the noise-gate onset timeout for that follow-up listen).
DONE_LISTEN_SECS = float(os.environ.get("NULVOICEAGENT_DONE_LISTEN_SECS", "5.0"))
# Model for the on-demand summary (headless `claude -p`). Fast/cheap by default.
DONE_SUMMARY_MODEL = os.environ.get("NULVOICEAGENT_DONE_SUMMARY_MODEL", "claude-haiku-4-5").strip()
# Cap the assistant text fed to the summarizer (keeps the prompt arg sane).
DONE_MAX_SUMMARY_CHARS = int(os.environ.get("NULVOICEAGENT_DONE_MAX_SUMMARY_CHARS", "12000"))
# Runtime mute for completion announcements. A FILE flag, not an env var, on
# purpose: each Stop hook spawns a fresh process, so a file checked at announce
# time gives live mute/unmute. Contains "1" while muted; absent = on.
DONE_MUTE_FLAG = CACHE_DIR / "done-muted"
# A SECOND, independent toggle: when set, completion announcements are restricted
# to tasks whose prompt was sent VIA VOICE (routed through dispatch into the pane).
# Tasks you TYPED directly into a pane finish silently. Absent (default) = announce
# every completion regardless of how the task was started.
DONE_VOICE_ONLY_FLAG = CACHE_DIR / "done-voice-only"
# Per-pane "last command arrived by voice" markers (one file per tmux pane id):
# dispatch stamps one when it routes a spoken command into a pane; done.py reads +
# consumes it to tell a voice-initiated completion from a typed one.
DONE_VOICE_MARK_DIR = CACHE_DIR / "voice-cmd"
# A voice marker older than this (seconds) is treated as orphaned — its task never
# produced a Stop — so a stale mark can't authorize a later typed task's announce.
DONE_VOICE_MARK_TTL = float(os.environ.get("NULVOICEAGENT_DONE_VOICE_MARK_TTL", "21600"))  # 6h


def voice_mark_path(pane: str) -> Path:
    """File recording the most recent voice-routed command into tmux pane id `pane`
    (e.g. '%5'); the id is sanitized so it's filesystem-safe."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", pane or "none")
    return DONE_VOICE_MARK_DIR / safe


# --- AskUserQuestion voice answering (Claude Code PreToolUse/PostToolUse hooks) -
# When a Claude Code pane calls the interactive AskUserQuestion tool, a PreToolUse
# hook announces "<who> needs you to pick options" + records the question payload;
# you then say "answer <pane>" to be read the options and pick them by voice
# (one/two/three… then "submit"). The answerer drives the live selector by sending
# keystrokes into the pane (digit selects, Tab advances multi-select, Enter submits
# — see askanswer.py). A PostToolUse hook clears the record once answered.
ASK_ENABLE = os.environ.get("NULVOICEAGENT_ASK_ENABLE", "1").strip() not in ("0", "false", "")
# Per-pane pending-question records (one JSON file per tmux pane id): written by the
# PreToolUse announce, consumed by "answer <pane>", cleared by PostToolUse / on answer.
ASK_PENDING_DIR = CACHE_DIR / "ask-pending"
# A pending record older than this (seconds) is treated as stale (the prompt was
# probably already dismissed) — liveness is still re-checked against the live pane.
ASK_PENDING_TTL = float(os.environ.get("NULVOICEAGENT_ASK_PENDING_TTL", "3600"))  # 1h
# Onset timeout (s) for each "say the number" listen inside the answer loop —
# more generous than a normal turn since you're reading/choosing between options.
ASK_LISTEN_SECS = float(os.environ.get("NULVOICEAGENT_ASK_LISTEN_SECS", "9.0"))
# Seconds to wait after sending a keystroke into the selector before re-reading the
# pane (lets the Claude Code TUI re-render). Bump if reads race the render.
ASK_RENDER_DELAY = float(os.environ.get("NULVOICEAGENT_ASK_RENDER_DELAY", "0.4"))


def ask_pending_path(pane: str) -> Path:
    """Pending-question record for tmux pane id `pane` (e.g. '%5'); id sanitized
    filesystem-safe, same scheme as voice_mark_path()."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", pane or "none")
    return ASK_PENDING_DIR / f"{safe}.json"


# --- viewer-paid TTS (bits / channel points -> spoken aloud) -----------------
# A paying viewer's chat message, read aloud through the Miku voice. Requests are
# spawned per-event by the nul-chat-hub daemon (`voiceagent viewer-say ...`), one
# at a time (flock), after moderation filtering. This is a SEPARATE path from the
# chat brain: the text is spoken verbatim, never sent to the agent's memory.
VIEWER_TTS_ENABLE = os.environ.get("NULVOICEAGENT_VIEWER_TTS", "1").strip() not in ("0", "false", "")
# Runtime kill switch (a FILE flag, like DONE_MUTE_FLAG): present = viewer TTS is
# off, so it can be silenced mid-stream without restarting anything. Each request
# is a fresh process that checks this at speak time. Absent (default) = enabled.
VIEWER_TTS_OFF_FLAG = CACHE_DIR / "viewer-tts-off"
# Hard cap on spoken characters (post-filter, pre-attribution). Twitch cheer/redeem
# messages can be long; keep the read short.
VIEWER_TTS_MAX_CHARS = int(os.environ.get("NULVOICEAGENT_VIEWER_TTS_MAX_CHARS", "200"))
# Blocklist: one word/phrase per line (`#` comments). A case-insensitive substring
# match on ANY line drops the whole message (we never half-speak a censored line).
VIEWER_TTS_BLOCKLIST = Path(os.environ.get(
    "NULVOICEAGENT_VIEWER_TTS_BLOCKLIST", str(CONFIG_DIR / "tts-blocklist.txt")))
# Serialize concurrent requests: each viewer-say holds this flock across synth +
# playback, so simultaneous cheers queue instead of overlapping.
VIEWER_TTS_LOCK = CACHE_DIR / "viewer-tts.lock"
# Yield to an in-progress Trickery turn (his chat/command recording or reply) for
# up to this long before speaking anyway, so viewer TTS doesn't talk over him.
VIEWER_TTS_WAIT_TURN_SECS = float(os.environ.get("NULVOICEAGENT_VIEWER_TTS_WAIT_TURN_SECS", "30"))
# Optional distinct voice/pitch/speed for viewers so they don't sound like they ARE
# the agent. Blank = use the active MikuVoice profile (same as chat). A small pitch
# shift is a cheap way to set viewers apart while keeping the autotune chain.
VIEWER_TTS_VOICE = os.environ.get("NULVOICEAGENT_VIEWER_TTS_VOICE", "").strip()
VIEWER_TTS_PITCH = os.environ.get("NULVOICEAGENT_VIEWER_TTS_PITCH", "").strip()
VIEWER_TTS_SPEED = os.environ.get("NULVOICEAGENT_VIEWER_TTS_SPEED", "").strip()


# --- route mode --------------------------------------------------------------
# "tell <named pane> to <payload>": a long-form manual recording (like chat, no
# silence cutoff), so it shares the manual safety cap.
ROUTE_MAX_SECS = float(os.environ.get("NULVOICEAGENT_ROUTE_MAX_SECS", str(MANUAL_MAX_SECS)))
# Gap (seconds) between bracketed-pasting a routed message into a pane and sending
# Enter. The target TUI (Claude Code) debounces input right after a paste, so an
# Enter sent too soon is swallowed into the paste and the task never starts —
# worst on a FOCUSED/visible pane. Big enough to clear that window; bump it if a
# slow/large paste still doesn't submit.
DISPATCH_ENTER_DELAY = float(os.environ.get("NULVOICEAGENT_DISPATCH_ENTER_DELAY", "0.4"))
# When a routed pane is mid-run (Claude Code's footer shows "esc to interrupt"),
# a dispatch sends Escape to interrupt the current turn before pasting the new
# message. This is the settle gap AFTER the Escape, so Claude Code finishes
# tearing down the running turn and returns to an empty prompt before we paste —
# too short and the paste lands while it's still aborting and gets eaten.
DISPATCH_INTERRUPT_DELAY = float(os.environ.get("NULVOICEAGENT_DISPATCH_INTERRUPT_DELAY", "0.6"))

# Shell commands typed into a pane to start a fresh agent session by voice
# ("start a Claude session in this pane"). Just the launcher name so the session
# starts in that pane's current working directory; override to add flags
# (e.g. "claude --resume", a model flag). See dispatch.try_start_session.
SESSION_LAUNCH = {
    "claude": os.environ.get("NULVOICEAGENT_CLAUDE_CMD", "claude"),
    "codex": os.environ.get("NULVOICEAGENT_CODEX_CMD", "codex"),
}
# Keys sent (in order) to gracefully end an agent session in a pane ("end session
# Krita Claude"). Ctrl-C twice is the cross-tool "get me out": Claude Code exits on
# the second Ctrl-C (the first shows "press again to exit"), and it aborts/quits
# Codex too. tmux key names; override the env with a comma-separated list.
SESSION_END_KEYS = [k for k in os.environ.get(
    "NULVOICEAGENT_SESSION_END_KEYS", "C-c,C-c").split(",") if k.strip()]
# Gap (seconds) between those keypresses so the target registers each one (Claude
# Code needs the first Ctrl-C to land before the second confirms the exit).
SESSION_END_KEY_DELAY = float(os.environ.get("NULVOICEAGENT_SESSION_END_KEY_DELAY", "0.3"))
