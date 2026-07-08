"""The command-mode allowlist — the safety gate.

Actions are DATA, not code: a user-editable file `~/.config/nulvoiceagent/actions.conf`
with `name = shell command` lines. The brain only ever returns a *name* from this
set (constrained by an enum / schema); nulvoiceagent looks the name up here and runs
the mapped command. The model can never inject or invent a command — anything not
in the file is refused.

Replace the seeded `echo` placeholders with your real launchers, e.g.
    setup = setupstream
"""
from __future__ import annotations
import subprocess

from . import config

_SEED = """\
# nulvoiceagent command allowlist (GLOBAL / OS-level) — `name = shell command`.
# The voice brain picks a name; nulvoiceagent runs the matching command below.
# Lines starting with # are ignored. Edit freely; no restart needed.
#
# Per-app (fork) commands live in apps/<app>.conf and are exposed namespaced as
# <app>.<name> (e.g. stream.start, krita.launch).

# --- examples you can enable ---
# lockscreen  = loginctl lock-session
# sleep       = systemctl suspend
"""


def _ensure_seed() -> None:
    if not config.ACTIONS_FILE.exists():
        config.ACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.ACTIONS_FILE.write_text(_SEED, encoding="utf-8")


def _parse(text: str, prefix: str = "") -> dict[str, str]:
    """Parse `name = command` lines; optionally namespace names as `<prefix>.<name>`."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, cmd = line.split("=", 1)
        name, cmd = name.strip(), cmd.strip()
        if name and cmd:
            out[f"{prefix}.{name}" if prefix else name] = cmd
    return out


def load() -> dict[str, str]:
    """Return {name: command} merging the global allowlist (`actions.conf`) with
    each per-fork allowlist (`apps/<app>.conf`, namespaced `<app>.<name>`). The
    classifier picks one of these names; `run()` maps it back to its command."""
    _ensure_seed()
    actions = _parse(config.ACTIONS_FILE.read_text(encoding="utf-8"))
    if config.APPS_DIR.is_dir():
        for conf in sorted(config.APPS_DIR.glob("*.conf")):
            actions.update(_parse(conf.read_text(encoding="utf-8"), prefix=conf.stem))
    return actions


def names() -> list[str]:
    return list(load().keys())


def run(name: str) -> bool:
    """Run the allowlisted command for `name`. Returns True if it was found+run."""
    cmd = load().get(name)
    if not cmd:
        return False
    subprocess.Popen(cmd, shell=True)  # detached; name is gated by the allowlist
    return True
