"""Narration — read the focused page (or what's highlighted) aloud, with
pause/resume.

Say "narrate" and Aurinex reads what's in front of you. Priority order:

  1. a LIVE in-page selection (what you have highlighted right now),
  2. the focused tab's main article text,
  3. outside the browser, the primary/clipboard selection.

The focused tab comes from CDP `document.hasFocus()`, which is true only for the
active tab of the focused window — so it answers "is the browser even the app
he's looking at?" and "which tab?" in one probe, with no window-manager query.

WHY A SEPARATE PROCESS: speech normally runs inside the turn, and barge-in
(pressing any voice hotkey) kills the turn's whole process group. That is exactly
wrong for narration — pressing the key to *say* "pause" would kill the reading
instead of pausing it. So the narrator is spawned detached (its own session), and
is driven by signals: SIGUSR1 pause, SIGUSR2 resume, SIGTERM stop.

Pressing a voice hotkey auto-pauses the narration (see __main__), both so the mic
doesn't hear Aurinex reading and so "pause"/"resume" are meaningful commands.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from difflib import SequenceMatcher
from typing import Optional

from . import cdp, config, state, tts


# --- page extraction ------------------------------------------------------- #

# Pull the live selection + a Readability-lite block list from the page. Blocks
# keep their tag so headings survive — that's what makes "narrate the section
# about X" possible without an LLM.
_PAGE_JS = r"""
(() => {
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  const sel = clean(window.getSelection ? String(window.getSelection()) : '');
  const SKIP = 'nav,header,footer,aside,form,figure,figcaption,button,select,' +
    'option,textarea,script,style,noscript,iframe,svg,[aria-hidden="true"],' +
    '[role="navigation"],[role="banner"],[role="contentinfo"],[role="complementary"]';
  const HEAD = {H1: 1, H2: 1, H3: 1, H4: 1, H5: 1, H6: 1};

  // Walk TEXT NODES and attribute each to its nearest block-level ancestor,
  // rather than selecting p/li/h*. Hand-rolled and CSS-grid sites put prose
  // straight in <div>s, which a tag selector silently drops; going through the
  // text lets any container hold text while inline spans still merge into their
  // parent block. Insertion order into the Map is document order.
  const styles = new Map();
  function inline(el) {
    let d = styles.get(el);
    if (d === undefined) { d = getComputedStyle(el).display; styles.set(el, d); }
    return d === 'inline' || d === 'inline-block' || d === 'contents' ||
           d === 'inline-flex';
  }
  function collect(root) {
    const groups = new Map();
    if (!root) return [];
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n = w.nextNode(); n; n = w.nextNode()) {
      if (!n.nodeValue || !n.nodeValue.trim()) continue;
      let el = n.parentElement;
      if (!el || el.closest(SKIP)) continue;
      while (el && el !== root && el.parentElement && inline(el)) el = el.parentElement;
      if (!el) continue;
      groups.set(el, (groups.get(el) || '') + ' ' + n.nodeValue);
    }
    const out = [];
    let prev = '';
    for (const [el, raw] of groups) {
      const t = clean(raw);
      if (!t || t === prev) continue;            // drop empties + repeats
      const tag = el.tagName.toLowerCase();
      // Semantic blocks are always kept; a bare div/td needs enough text to be
      // prose rather than a badge, label or breadcrumb.
      if (!HEAD[el.tagName] && !/^(p|li|blockquote|pre|dd|dt)$/.test(tag) &&
          t.length < 12) continue;
      if (!el.getClientRects().length) continue; // hidden subtree / collapsed
      out.push({tag: tag, text: t});
      prev = t;
    }
    return out;
  }
  const chars = b => b.reduce((n, x) => n + x.text.length, 0);
  let best = null;
  for (const s of ['article', 'main', '[role="main"]', '#content', '.post',
                   '.entry-content', '.article-body']) {
    const el = document.querySelector(s);
    if (!el) continue;
    const b = collect(el);
    if (chars(b) >= 400) { best = b; break; }   // a real article container
  }
  if (!best || !chars(best)) best = collect(document.body);
  if (!chars(best)) {
    const t = clean(document.body ? document.body.innerText : '');
    best = t ? [{tag: 'p', text: t}] : [];
  }
  return {sel: sel, title: document.title, url: location.href, blocks: best};
})()
"""


class Source:
    """Extracted text plus where it came from (used for the spoken preamble)."""

    def __init__(self, kind: str, title: str, blocks: list[dict]):
        self.kind = kind            # selection | page | clipboard
        self.title = title or ""
        self.blocks = blocks        # [{tag, text}]

    @property
    def text(self) -> str:
        return "\n".join(b["text"] for b in self.blocks)

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def minutes(self) -> float:
        return self.words / max(1.0, config.NARRATE_WPM)


def focused_app() -> str:
    """Best-effort name of the focused app, from the cache the taskbar keeps
    fresh for voicechat's desktop-paste guard. '' when nothing is focused."""
    try:
        raw = (config.VOICECHAT_STATUS_FILE.parent / "active-window").read_text().strip()
    except OSError:
        return ""
    return re.sub(r"\.desktop$", "", raw)


def _selection_fallback() -> str:
    """Text highlighted outside the browser, via the Wayland PRIMARY selection
    (falling back to the clipboard).

    NOTE: primary selection is *sticky* — it holds the last thing highlighted
    anywhere, possibly hours ago — so this is only ever a fallback for apps we
    can't inspect, never the browser (where CDP gives us a truly live selection).
    """
    for args in (["wl-paste", "--primary", "--no-newline"],
                 ["wl-paste", "--no-newline"]):
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=3)
        except (OSError, subprocess.SubprocessError):
            continue
        text = (r.stdout or "").strip()
        # A bare URL is almost always a stale address-bar highlight, not prose.
        if len(text) >= config.NARRATE_MIN_SELECTION and not re.fullmatch(
                r"https?://\S+", text):
            return text
    return ""


def extract(prefer: str = "auto", focus: str = "") -> tuple[Optional[Source], str]:
    """Return (Source, error). `prefer`: auto | selection | page.

    auto      — a live selection long enough to be deliberate wins, else the page.
    selection — only what's highlighted ("narrate the highlighted text").
    page      — ignore the selection ("narrate the whole page").
    """
    page = None
    try:
        page = cdp.focused_page()
    except Exception:  # noqa: BLE001 — browser trouble must never break the turn
        page = None

    if page is not None:
        try:
            data = cdp.evaluate(page, _PAGE_JS) or {}
        except Exception as e:  # noqa: BLE001
            return None, f"I couldn't read that tab. {e}"
        sel = (data.get("sel") or "").strip()
        title = data.get("title") or ""
        if prefer != "page" and len(sel) >= config.NARRATE_MIN_SELECTION:
            return Source("selection", title, [{"tag": "p", "text": sel}]), ""
        if prefer == "selection":
            return None, ("I don't see anything highlighted on that page, Trickery.")
        blocks = [b for b in (data.get("blocks") or []) if b.get("text")]
        if not blocks:
            return None, "There's no readable text on that page."
        if focus:
            blocks, err = _section(blocks, focus)
            if err:
                return None, err
        return Source("page", title, _cap(blocks)), ""

    # Not the browser — fall back to whatever is highlighted system-wide.
    text = _selection_fallback()
    if text:
        return Source("clipboard", "", [{"tag": "p", "text": text}]), ""
    app = focused_app()
    if app:
        return None, (f"I can't read {app} yet, Trickery — highlight some text "
                      "and I'll narrate that.")
    return None, ("I can't tell what you're looking at. Highlight some text or "
                  "focus a page and try again.")


def _cap(blocks: list[dict]) -> list[dict]:
    """Trim to NARRATE_MAX_CHARS so a pathological page can't queue hours of speech."""
    out, total = [], 0
    for b in blocks:
        total += len(b["text"])
        if total > config.NARRATE_MAX_CHARS:
            break
        out.append(b)
    return out or blocks[:1]


_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def _norm_words(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split())


def _section(blocks: list[dict], focus: str) -> tuple[list[dict], str]:
    """Slice out the section under the heading that best matches `focus` — from
    that heading up to the next heading of the same or higher rank."""
    want = _norm_words(focus)
    if not want:
        return blocks, ""
    best_i, best_score = -1, 0.0
    for i, b in enumerate(blocks):
        if b["tag"] not in _HEADINGS:
            continue
        head = _norm_words(b["text"])
        if not head:
            continue
        # Containment is the strong signal ("indexing" in "B-Tree Indexing");
        # fuzzy ratio catches ASR drift on the section name.
        score = 1.0 if want in head else SequenceMatcher(None, want, head).ratio()
        if score > best_score:
            best_i, best_score = i, score
    if best_i < 0 or best_score < 0.6:
        return [], f"I couldn't find a section about {focus} on that page."
    level = int(blocks[best_i]["tag"][1])
    out = [blocks[best_i]]
    for b in blocks[best_i + 1:]:
        if b["tag"] in _HEADINGS and int(b["tag"][1]) <= level:
            break
        out.append(b)
    return out, ""


# --- chunking -------------------------------------------------------------- #

_RE_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_RE_CITE = re.compile(r"\[\d+(?:,\s*\d+)*\]")       # "[1]", "[2, 3]" wiki refs
_RE_URL = re.compile(r"https?://\S+")


def _speakable(text: str) -> str:
    """Light cleanup so the TTS doesn't read plumbing out loud."""
    text = _RE_CITE.sub("", text)
    text = _RE_URL.sub("a link", text)
    return " ".join(text.split())


def chunks_for(source: Source) -> list[str]:
    """Split into spoken chunks. Chunks are the pause/resume granularity and the
    replay unit after a pause, so they stay around a sentence or two."""
    out: list[str] = []
    for b in source.blocks:
        text = _speakable(b["text"])
        if not text:
            continue
        if b["tag"] in _HEADINGS:
            out.append(text if text.endswith((".", "!", "?", ":")) else text + ".")
            continue
        buf = ""
        for sent in _RE_SENT.split(text):
            sent = sent.strip()
            if not sent:
                continue
            if buf and len(buf) + 1 + len(sent) > config.NARRATE_CHUNK_CHARS:
                out.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            out.append(buf)
    return out


# --- control (caller side) ------------------------------------------------- #

def _pid() -> Optional[int]:
    try:
        pid = int(config.NARRATE_PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None
    # Guard against PID reuse the same way the turn/recording pidfiles do.
    try:
        cmdline = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "replace")
    except OSError:
        _clear()
        return None
    if "nulvoiceagent" not in cmdline:
        _clear()
        return None
    return pid


def _clear() -> None:
    for p in (config.NARRATE_PID_FILE, config.NARRATE_STATE_FILE):
        try:
            p.unlink()
        except OSError:
            pass


def status() -> dict:
    """{state: playing|paused|idle, i, n, title}. state=idle when nothing runs."""
    if _pid() is None:
        return {"state": "idle", "i": 0, "n": 0, "title": ""}
    try:
        d = json.loads(config.NARRATE_STATE_FILE.read_text())
    except (OSError, ValueError):
        return {"state": "idle", "i": 0, "n": 0, "title": ""}
    if d.get("state") not in ("playing", "paused"):
        return {"state": "idle", "i": 0, "n": 0, "title": ""}
    return d


def is_active() -> bool:
    """True while a narration is playing OR paused — the gate that lets 'pause',
    'resume' and 'stop' claim an utterance only during a narration."""
    return status()["state"] in ("playing", "paused")


def is_paused() -> bool:
    return status()["state"] == "paused"


def _signal(sig: int) -> bool:
    pid = _pid()
    if pid is None:
        return False
    try:
        os.kill(pid, sig)
        return True
    except OSError:
        _clear()
        return False


def pause() -> bool:
    return _signal(signal.SIGUSR1)


def resume() -> bool:
    return _signal(signal.SIGUSR2)


def stop() -> bool:
    pid = _pid()
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        _clear()
        return False
    for _ in range(40):                          # let it clean up its own files
        time.sleep(0.05)
        if _pid() is None:
            return True
    # Still alive after 2s — it's most likely blocked in a synth HTTP call. Escalate
    # rather than just deleting the pidfile, which would orphan a process that goes
    # on to speak with nothing left able to signal it.
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    _clear()
    return True


def pause_for_input() -> bool:
    """Pause a running narration because a voice hotkey was pressed.

    Called before recording starts: the mic must not hear Aurinex reading, and
    it makes 'pause'/'resume' real commands rather than races. Returns True if a
    playing narration was paused.
    """
    if status()["state"] != "playing":
        return False
    return pause()


def start(source: Source, chunks: list[str]) -> bool:
    """Spawn the detached narrator. Replaces any narration already running."""
    if not chunks:
        return False
    stop()
    try:
        config.NARRATE_TEXT_FILE.write_text(json.dumps(
            {"title": source.title, "kind": source.kind, "chunks": chunks}))
    except OSError:
        return False
    env = dict(os.environ)
    env["PYTHONPATH"] = str(config.REPO_DIR) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        subprocess.Popen(
            [sys.executable, "-m", "nulvoiceagent", "narrate-player"],
            cwd=str(config.REPO_DIR), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            # Its OWN session, so barge-in's killpg on the turn's process group
            # can't take the narration down — pausing must not mean killing.
            start_new_session=True)
    except OSError:
        return False
    for _ in range(40):                          # wait for it to claim the pidfile
        if _pid() is not None:
            return True
        time.sleep(0.05)
    return True


# --- the narrator process -------------------------------------------------- #

_stop_evt = threading.Event()
_pause_evt = threading.Event()


def _write_state(st: str, i: int, n: int, title: str) -> None:
    try:
        tmp = config.NARRATE_STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"state": st, "i": i, "n": n, "title": title,
                                   "pid": os.getpid(), "ts": time.time()}))
        tmp.replace(config.NARRATE_STATE_FILE)
    except OSError:
        pass


def _play_interruptible(wav: bytes, sink: Optional[str]) -> bool:
    """Play one chunk. Returns True if it finished, False if pause/stop cut it
    short (the chunk is then replayed from its start on resume — chunks are short
    enough that re-hearing one sentence reads as natural)."""
    # Synth can take a moment, so a pause/stop may have landed while we were in it
    # — check before making any sound, or "stop" emits a blip of the next chunk.
    if _stop_evt.is_set() or _pause_evt.is_set():
        return False
    dur = state.wav_duration(wav)
    until = time.time() + max(0.0, dur) + 0.5
    env, hop = tts._mouth_envelope(wav)
    state.write("talking", until=until, mouth=(env[0] if env else None))
    mouth_stop = threading.Event()
    driver = None
    if env:
        driver = threading.Thread(target=tts._drive_mouth,
                                  args=(env, hop, dur, until, mouth_stop, None),
                                  daemon=True)
        driver.start()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        f.write(wav)
        f.flush()
        if tts._has("pw-play"):
            cmd = ["pw-play"] + (["--target", sink] if sink else []) + [f.name]
        else:
            cmd = ["paplay", f.name]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        except OSError:
            mouth_stop.set()
            return True                          # no player: don't spin forever
        try:
            while proc.poll() is None:
                if _stop_evt.is_set() or _pause_evt.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return False
                time.sleep(0.05)
        finally:
            mouth_stop.set()
            if driver is not None:
                driver.join(timeout=0.3)
    return True


def player_main() -> int:
    """The detached narrator: synth + play chunk by chunk, honouring signals."""
    try:
        payload = json.loads(config.NARRATE_TEXT_FILE.read_text())
    except (OSError, ValueError):
        return 1
    chunks = payload.get("chunks") or []
    title = payload.get("title") or ""
    if not chunks:
        return 1

    signal.signal(signal.SIGUSR1, lambda *_: _pause_evt.set())
    signal.signal(signal.SIGUSR2, lambda *_: _pause_evt.clear())
    signal.signal(signal.SIGTERM, lambda *_: _stop_evt.set())
    signal.signal(signal.SIGINT, lambda *_: _stop_evt.set())

    try:
        config.NARRATE_PID_FILE.write_text(str(os.getpid()))
    except OSError:
        return 1

    # Resolve the FX sink once — _sink_present shells out to pw-cli, which is far
    # too expensive to repeat per chunk.
    sink = (config.TTS_SINK
            if config.TTS_SINK and tts._sink_present(config.TTS_SINK) else None)
    n = len(chunks)
    try:
        i = 0
        while i < n:
            if _stop_evt.is_set():
                break
            if _pause_evt.is_set():
                _write_state("paused", i, n, title)
                state.idle()
                while _pause_evt.is_set() and not _stop_evt.is_set():
                    time.sleep(0.1)
                continue                          # re-check, then replay chunk i
            _write_state("playing", i, n, title)
            try:
                wav = tts.synth(chunks[i])
            except Exception:  # noqa: BLE001 — skip a chunk TTS chokes on
                i += 1
                continue
            if _stop_evt.is_set():
                break
            if _play_interruptible(wav, sink):
                i += 1
            # else: paused/stopped mid-chunk — loop replays this chunk on resume
    finally:
        _clear()
        state.idle()
    return 0


# --- entry point used by the intent ---------------------------------------- #

def _log(line: str) -> None:
    try:
        with open(config.CACHE_DIR / "nulvoiceagent.log", "a",
                  encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')}  narrate: {line}\n")
    except OSError:
        pass


def handle(prefer: str = "auto", focus: str = "",
           summarize: bool = False) -> str:
    """Extract, then start narrating. Returns what to SAY: empty on success,
    because the narration itself is the reply — anything spoken here would talk
    over the first chunk."""
    src, err = extract(prefer=prefer, focus=focus)
    if err or src is None:
        _log(f"no source (prefer={prefer} focus={focus!r}): {err}")
        return err or "I couldn't find anything to read."
    _log(f"{src.kind} {src.words}w ~{src.minutes:.1f}min "
         f"prefer={prefer} focus={focus!r} summarize={summarize} "
         f"title={src.title[:60]!r}")
    if summarize:
        from . import brain                      # local: keeps the fast path light
        state.thinking()
        digest = brain.condense(src.text, focus=focus)
        if not digest:
            state.idle()
            _log("condense returned nothing")
            return "I couldn't get a summary of that."
        src = Source("summary", src.title, [{"tag": "p", "text": digest}])
        _log(f"condensed to {src.words}w")
    chunks = chunks_for(src)
    if not chunks:
        return "There's nothing to read there."
    if not start(src, chunks):
        _log("failed to start the narrator")
        return "I couldn't start narrating."
    _log(f"started: {len(chunks)} chunks")
    return ""
