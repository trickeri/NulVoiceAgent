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
    nulvoiceagent claude windows [<window>]       # Ghostty/tmux windows, panes by row/col + state
    nulvoiceagent claude start <pane|empty|empty:<window>> <dir> <name…> [--codex]
    nulvoiceagent claude new <name> <dir> [claude|codex]   # in a NEW pane (tmux session "remote")
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


_SHELLS = {"zsh", "bash", "fish", "sh", "dash"}
_WIN_FIELDS = ("#{session_name}\t#{window_index}\t#{pane_id}\t#{pane_current_command}\t"
               "#{pane_pid}\t#{pane_current_path}\t#{@va_name}\t#{pane_left}\t#{pane_top}\t"
               "#{pane_width}\t#{pane_height}")


def _focused_window() -> Optional[str]:
    pane = dispatch.active_pane()
    if not pane:
        return None
    cp = dispatch._tmux("display-message", "-p", "-t", pane, "#{session_name}:#{window_index}")
    return cp.stdout.strip() or None


def _shell_state(pane: str, pid: str) -> str:
    """'empty' = a bare shell prompt (no children, nothing on screen but the prompt);
    'shell' = an idle shell with history on screen."""
    import subprocess
    if subprocess.run(["pgrep", "-P", pid], capture_output=True).returncode == 0:
        return "shell-busy"
    cp = dispatch._tmux("capture-pane", "-p", "-t", pane)
    lines = [ln for ln in cp.stdout.splitlines() if ln.strip()]
    return "empty" if len(lines) <= 2 else "shell"


def windows() -> dict:
    """Every tmux window (= one Ghostty window, since each Ghostty window owns a
    `w<pid>` session) with its panes laid out in rows/columns, each pane
    classified: empty | shell | shell-busy | claude/codex (idle|busy|question) |
    <other program>."""
    cp = dispatch._tmux("list-panes", "-a", "-F", _WIN_FIELDS)
    wins: dict[str, dict] = {}
    for ln in cp.stdout.splitlines():
        p = ln.split("\t")
        if len(p) != 11:
            continue
        sess, widx, pane, cmd, pid, cwd, name, left, top, w, h = p
        key = f"{sess}:{widx}"
        wins.setdefault(key, {"window": key, "ghostty": sess.startswith("w") and sess[1:].isdigit(),
                              "panes": []})
        agent = _AGENT_CMDS.get(cmd.lower())
        if agent:
            state = f"{agent}:{_state(pane)}"
        elif cmd.lower() in _SHELLS:
            state = _shell_state(pane, pid)
        else:
            state = cmd
        wins[key]["panes"].append({"pane": pane, "name": name.strip() or None, "command": cmd,
                                   "cwd": cwd, "state": state, "_xy": (int(top), int(left))})
    focused = _focused_window()
    out = []
    for key, win in wins.items():
        panes = win["panes"]
        tops = sorted({p["_xy"][0] for p in panes})
        for p in panes:
            row = tops.index(p["_xy"][0])
            cols = sorted(q["_xy"][1] for q in panes if q["_xy"][0] == p["_xy"][0])
            p["position"] = f"row {row + 1} col {cols.index(p['_xy'][1]) + 1}"
        for p in panes:
            p.pop("_xy")
        panes.sort(key=lambda p: p["position"])
        win["pane_count"] = len(panes)
        win["empty_panes"] = [p["pane"] for p in panes if p["state"] == "empty"]
        win["focused"] = key == focused
        out.append(win)
    out.sort(key=lambda w: (not w["focused"], -w["pane_count"]))
    return {"ok": True, "windows": out}


def _pick_window(sel: str, wins: list) -> Optional[dict]:
    """'w33627' / 'w33627:1' / '8' or '8-pane' (pane count) / 'focused' / '' (any)."""
    s = sel.strip().lower().removesuffix("-pane").removesuffix(" pane").removesuffix("pane").strip()
    if not s:
        return None
    if s in ("focused", "this", "current"):
        return next((w for w in wins if w["focused"]), None)
    if s.isdigit():
        hits = [w for w in wins if w["pane_count"] == int(s)]
        return hits[0] if len(hits) == 1 else (hits[0] if hits else None)
    return next((w for w in wins if w["window"].lower() == s or w["window"].split(":")[0].lower() == s), None)


def start_in_pane(where: str, cwd: str, name: str, tool: str = "claude") -> dict:
    """Start an agent in an EXISTING pane and name it. `where` = a pane id, or
    'empty' / 'empty:<window selector>' to take the first empty pane (in that
    window). The pane must be an idle shell; we never type into a running program."""
    import os
    import shlex
    cwd = os.path.expanduser(cwd)
    if not os.path.isdir(cwd):
        return {"ok": False, "error": f"no such directory: {cwd}"}
    if tool not in config.SESSION_LAUNCH:
        return {"ok": False, "error": f"tool must be one of {sorted(config.SESSION_LAUNCH)}"}
    wins = windows()["windows"]
    allp = {p["pane"]: (w, p) for w in wins for p in w["panes"]}
    if where.lower().startswith("empty"):
        sel = where.partition(":")[2]
        win = _pick_window(sel, wins) if sel else None
        if sel and win is None:
            return {"ok": False, "error": f"no window matches {sel!r}",
                    "windows": [{"window": w["window"], "panes": w["pane_count"]} for w in wins]}
        pool = [win] if win else wins
        pane = next((pid for w in pool for pid in w["empty_panes"]), None)
        if pane is None:
            return {"ok": False, "error": "no empty pane" + (f" in {win['window']}" if win else "")}
    else:
        pane = where if where in allp else resolve(where)
        if pane is None or pane not in allp:
            return {"ok": False, "error": f"no pane matches {where!r}"}
    state = allp[pane][1]["state"]
    if state not in ("empty", "shell"):
        return {"ok": False, "error": f"pane {pane} isn't an idle shell (state: {state})"}
    if dispatch._match_key(name, dispatch.load()) == name:
        return {"ok": False, "error": f"a pane named {name!r} already exists"}
    dispatch.register(name, pane)
    dispatch.run_in_pane(pane, f"cd {shlex.quote(cwd)} && {config.SESSION_LAUNCH[tool]}")
    w, p = allp[pane]
    return {"ok": True, "pane": pane, "name": name, "cwd": cwd, "tool": tool,
            "window": w["window"], "position": p["position"]}


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
        elif verb in ("windows", "panes"):
            res = windows()
            if rest:
                win = _pick_window(" ".join(rest), res["windows"])
                res = {"ok": win is not None, "windows": [win] if win else [],
                       **({} if win else {"error": f"no window matches {' '.join(rest)!r}"})}
        elif verb == "start":
            if len(rest) < 3:
                raise ValueError("usage: start <pane|empty|empty:<window>> <dir> <name…> [--codex]")
            tool = "codex" if "--codex" in rest else "claude"
            rest = [a for a in rest if a != "--codex"]
            res = start_in_pane(rest[0], rest[1], " ".join(rest[2:]), tool)
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
