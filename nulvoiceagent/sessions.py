"""Remote control of agent (Claude Code / Codex) sessions in tmux panes.

The text/JSON twin of the voice route mode ("tell Krita Claude to …", "answer
Build Claude"): same registry (`@va_name` via dispatch.load), same paste path
(dispatch.send, auto-interrupt when busy), same closed-loop selector driver
(askanswer.read_screen / apply_picks / submit) — just with no mic and no TTS, so
Hermes (and anything behind the tailnet API: the laptop Claude, the mobile app)
can drive sessions deterministically:

    nulvoiceagent claude list [--all]
    nulvoiceagent claude read <target> [lines]
    nulvoiceagent claude send <target> <text…>
    nulvoiceagent claude wait <target> [timeout_s]
    nulvoiceagent claude ask  <target> <text…> [--timeout N]   # send + wait
    nulvoiceagent claude question <target>
    nulvoiceagent claude answer <target> <pick> [<pick> …]     # "2" / "1,3"
    nulvoiceagent claude interrupt <target>
    nulvoiceagent claude new <name> <dir> [claude|codex]
    nulvoiceagent claude end <target>
    nulvoiceagent claude name <target> <name>

<target> = a registered name ("Krita Claude", fuzzy like voice), a guessed label
shown by `list` ("JobHunter"), or a raw pane id ("%4"). Every verb prints ONE JSON
object; "ok": false + "error" on failure (exit status 1).
"""
from __future__ import annotations

import json
import re
import sys
import time
from typing import Optional

from . import askanswer, config, dispatch

# tmux session that holds sessions started remotely (`new`). Detached, so it isn't
# one of the per-Ghostty-window `w<pid>` sessions that die with their window, and
# destroy-unattached is off, so it survives with no client. `tmux attach -t remote`.
REMOTE_SESSION = "remote"

_AGENT_CMDS = {"claude": "claude", "codex": "codex"}
_FIELDS = "#{pane_id}\t#{@va_name}\t#{pane_current_command}\t#{pane_current_path}\t#{session_name}:#{window_index}"


def _panes() -> list[dict]:
    cp = dispatch._tmux("list-panes", "-a", "-F", _FIELDS)
    rows = []
    for ln in cp.stdout.splitlines():
        parts = ln.split("\t")
        if len(parts) != 5:
            continue
        pane, name, cmd, cwd, where = parts
        rows.append({"pane": pane, "name": name.strip() or None, "command": cmd,
                     "cwd": cwd, "where": where,
                     "agent": _AGENT_CMDS.get(cmd.lower())})
    return rows


def _state(pane: str) -> str:
    scr = askanswer.read_screen(pane)
    if scr and scr.get("kind") in ("question", "submit"):
        return "question"
    return "busy" if dispatch.pane_busy(pane) else "idle"


def _label(row: dict) -> Optional[str]:
    if row["name"]:
        return row["name"]
    kw = dispatch.guess_pane_label(row["pane"])
    return f"{kw} {(row['agent'] or 'pane').title()}" if kw else None


def list_sessions(include_all: bool = False) -> dict:
    out = []
    for row in _panes():
        if not include_all and not row["agent"] and not row["name"]:
            continue
        row = dict(row, label=_label(row))
        row["state"] = _state(row["pane"]) if row["agent"] else "shell"
        out.append(row)
    return {"ok": True, "sessions": out}


def resolve(target: str) -> Optional[str]:
    """Name / guessed label / %id -> pane id."""
    t = target.strip()
    if re.fullmatch(r"%\d+", t):
        return t if dispatch.pane_alive(t) else None
    targets = dispatch.load()
    key = dispatch._match_key(t, targets)
    if key is not None:
        return targets[key]["pane"]
    # Unnamed agent panes, by the label `list` shows.
    labels = {}
    for row in _panes():
        if row["agent"] and not row["name"]:
            lab = _label(row)
            if lab:
                labels[lab] = {"pane": row["pane"]}
    key = dispatch._match_key(t, labels)
    return labels[key]["pane"] if key is not None else None


def _who(pane: str) -> str:
    return dispatch.speaker_for_pane(pane)


def read(pane: str, lines: int = 60) -> dict:
    cp = dispatch._tmux("capture-pane", "-p", "-J", "-t", pane, "-S", f"-{max(lines, 1)}")
    text = cp.stdout.rstrip("\n").splitlines()
    while text and not text[-1].strip():
        text.pop()
    return {"ok": True, "pane": pane, "who": _who(pane), "state": _state(pane),
            "screen": "\n".join(text[-lines:])}


def send(pane: str, text: str) -> dict:
    ok = dispatch.send(pane, text)
    if ok:
        dispatch._stash_payload(text)
    return {"ok": ok, "pane": pane, "who": _who(pane)} if ok else \
        {"ok": False, "error": f"pane {pane} is gone"}


def wait(pane: str, timeout: float = 600, start_grace: float = 8) -> dict:
    """Block until the pane's turn ends (idle) or it asks a question. Waits up to
    `start_grace` for the turn to START first, since a just-sent message takes a
    moment before the spinner shows."""
    t0 = time.time()
    while time.time() - t0 < start_grace and _state(pane) == "idle":
        time.sleep(0.5)
    while time.time() - t0 < timeout:
        st = _state(pane)
        if st != "busy":
            break
        time.sleep(1.0)
    res = read(pane)
    res["waited_s"] = round(time.time() - t0, 1)
    res["timed_out"] = res["state"] == "busy"
    return res


def question(pane: str) -> dict:
    scr = askanswer.read_screen(pane)
    if scr is None:
        return {"ok": False, "error": "can't read pane"}
    if scr.get("kind") == "gone":
        return {"ok": True, "pane": pane, "asking": False}
    return {"ok": True, "pane": pane, "who": _who(pane), "asking": True,
            "questions_total": askanswer._count_questions(pane), "current": scr}


def answer(pane: str, picks: list[list[int]], do_submit: bool = True) -> dict:
    """Answer the live AskUserQuestion selector: picks[i] = 1-based option numbers
    for question i (several for multi-select). Same closed-loop walk as the voice
    answerer — every step re-reads the screen."""
    scr = askanswer.read_screen(pane)
    if scr is None or scr.get("kind") == "gone":
        return {"ok": False, "error": f"{_who(pane)} isn't asking anything right now"}
    chosen = []
    idx = 0
    while True:
        scr = askanswer.read_screen(pane)
        if scr is None:
            return {"ok": False, "error": "lost the prompt", "chosen": chosen}
        if scr.get("kind") in ("submit", "gone"):
            break
        if idx >= len(picks):
            return {"ok": False, "error": f"no pick given for question {idx + 1}",
                    "current": scr, "chosen": chosen}
        bad = [p for p in picks[idx] if not 1 <= p <= len(scr["options"])]
        if bad or not picks[idx]:
            return {"ok": False, "error": f"question {idx + 1} has options 1-{len(scr['options'])}",
                    "current": scr, "chosen": chosen}
        prev_q = scr.get("question")
        chosen.append({"question": prev_q,
                       "picked": [scr["options"][p - 1] for p in picks[idx]]})
        askanswer.apply_picks(pane, scr, picks[idx])
        idx += 1
        askanswer._wait_screen(pane, lambda s, pq=prev_q: s.get("kind") != "question"
                               or s.get("question") != pq)
    submitted = False
    if scr.get("kind") == "submit" and do_submit:
        submitted = askanswer.submit(pane)
    elif scr.get("kind") == "gone":
        submitted = True           # lone single-select auto-submits on the pick
    askanswer.clear_pending(pane)
    return {"ok": submitted or not do_submit, "pane": pane, "who": _who(pane),
            "chosen": chosen, "submitted": submitted}


def interrupt(pane: str) -> dict:
    dispatch._tmux("send-keys", "-t", pane, "Escape")
    return {"ok": True, "pane": pane}


def new(name: str, cwd: str, tool: str = "claude") -> dict:
    import os
    cwd = os.path.expanduser(cwd)
    if not os.path.isdir(cwd):
        return {"ok": False, "error": f"no such directory: {cwd}"}
    if tool not in config.SESSION_LAUNCH:
        return {"ok": False, "error": f"tool must be one of {sorted(config.SESSION_LAUNCH)}"}
    if dispatch._match_key(name, dispatch.load()) == name:
        return {"ok": False, "error": f"a pane named {name!r} already exists"}
    if dispatch._tmux("has-session", "-t", REMOTE_SESSION).returncode != 0:
        cp = dispatch._tmux("new-session", "-d", "-s", REMOTE_SESSION, "-c", cwd,
                            "-P", "-F", "#{pane_id}")
    else:
        cp = dispatch._tmux("new-window", "-d", "-t", REMOTE_SESSION, "-c", cwd,
                            "-P", "-F", "#{pane_id}")
    pane = cp.stdout.strip()
    if cp.returncode != 0 or not pane:
        return {"ok": False, "error": cp.stderr.strip() or "tmux failed"}
    dispatch.register(name, pane)
    time.sleep(0.5)                # let the shell prompt come up before typing
    dispatch.run_in_pane(pane, config.SESSION_LAUNCH[tool])
    return {"ok": True, "pane": pane, "name": name, "cwd": cwd, "tool": tool,
            "attach": f"tmux attach -t {REMOTE_SESSION}"}


def end(pane: str) -> dict:
    return {"ok": dispatch.end_session_in_pane(pane), "pane": pane}


def name_pane(pane: str, name: str) -> dict:
    dispatch.register(name, pane)
    return {"ok": True, "pane": pane, "name": name}


def _parse_picks(args: list[str]) -> list[list[int]]:
    return [[int(n) for n in a.split(",") if n.strip()] for a in args]


def main(argv: list[str]) -> int:
    verb = argv[0] if argv else "list"
    rest = argv[1:]
    try:
        if verb == "list":
            res = list_sessions("--all" in rest)
        elif verb == "new":
            if len(rest) < 2:
                raise ValueError("usage: new <name> <dir> [claude|codex]")
            res = new(rest[0], rest[1], rest[2] if len(rest) > 2 else "claude")
        elif verb in ("read", "send", "wait", "ask", "question", "answer",
                      "interrupt", "end", "name"):
            if not rest:
                raise ValueError(f"usage: {verb} <target> …")
            pane = resolve(rest[0])
            if pane is None:
                res = {"ok": False, "error": f"no session matches {rest[0]!r}",
                       "sessions": [s["label"] or s["pane"] for s in list_sessions()["sessions"]]}
            elif verb == "read":
                res = read(pane, int(rest[1]) if len(rest) > 1 else 60)
            elif verb == "send":
                res = send(pane, " ".join(rest[1:]))
            elif verb == "wait":
                res = wait(pane, float(rest[1]) if len(rest) > 1 else 600)
            elif verb == "ask":
                words, timeout = rest[1:], 600.0
                if "--timeout" in words:
                    i = words.index("--timeout")
                    timeout = float(words[i + 1])
                    words = words[:i] + words[i + 2:]
                res = send(pane, " ".join(words))
                if res["ok"]:
                    res = wait(pane, timeout)
            elif verb == "question":
                res = question(pane)
            elif verb == "answer":
                res = answer(pane, _parse_picks(rest[1:]))
            elif verb == "interrupt":
                res = interrupt(pane)
            elif verb == "end":
                res = end(pane)
            else:
                res = name_pane(pane, " ".join(rest[1:]))
        else:
            raise ValueError(__doc__)
    except ValueError as e:
        res = {"ok": False, "error": str(e)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
