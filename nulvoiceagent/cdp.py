"""Minimal Chrome DevTools Protocol client — stdlib only.

NulVoiceAgent has no venv and no third-party packages, so instead of pulling in a
websocket library we speak just enough of RFC 6455 to drive CDP: a client
handshake, masked text frames out, (possibly fragmented) frames in.

Nulium already runs with `--remote-debugging-port=9222`, so this needs no
browser-side change and no extension. We deliberately send NO `Origin` header —
Chromium's `--remote-allow-origins` check only rejects *non-null* origins, so an
origin-less client is allowed by default.

Used by narrate.py to pull the focused tab's live selection and body text.
"""
from __future__ import annotations

import base64
import json
import os
import socket
import struct
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from . import config


class CdpError(RuntimeError):
    pass


# --- websocket ------------------------------------------------------------- #

class _WS:
    """A single CDP websocket connection. Buffered because the server may pack
    the tail of the handshake and the first frame into one TCP read."""

    def __init__(self, url: str, timeout: float):
        u = urllib.parse.urlparse(url)
        host, port = u.hostname or "127.0.0.1", u.port or 80
        path = u.path + (("?" + u.query) if u.query else "")
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.settimeout(timeout)
        self._buf = b""
        key = base64.b64encode(os.urandom(16)).decode()
        self._sock.sendall(
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n".encode()
        )
        while b"\r\n\r\n" not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise CdpError("no handshake response")
            self._buf += chunk
        head, self._buf = self._buf.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise CdpError("websocket upgrade refused")
        self._id = 0

    # -- framing --
    def _read(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self._sock.recv(max(4096, n - len(self._buf)))
            if not chunk:
                raise CdpError("websocket closed")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        hdr = bytearray([0x80 | opcode])
        n = len(payload)
        if n < 126:
            hdr.append(0x80 | n)
        elif n < 65536:
            hdr.append(0x80 | 126)
            hdr += struct.pack(">H", n)
        else:
            hdr.append(0x80 | 127)
            hdr += struct.pack(">Q", n)
        key = os.urandom(4)                      # client frames MUST be masked
        hdr += key
        self._sock.sendall(bytes(hdr) + bytes(b ^ key[i % 4]
                                              for i, b in enumerate(payload)))

    def _recv_message(self) -> str:
        """Assemble one full text message, answering pings and skipping pongs."""
        data = b""
        while True:
            b1, b2 = self._read(2)
            fin, opcode, masked, ln = b1 & 0x80, b1 & 0x0F, b2 & 0x80, b2 & 0x7F
            if ln == 126:
                ln = struct.unpack(">H", self._read(2))[0]
            elif ln == 127:
                ln = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if masked else b""
            payload = self._read(ln) if ln else b""
            if masked:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            if opcode == 0x9:                    # ping -> pong
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:                    # pong
                continue
            if opcode == 0x8:
                raise CdpError("websocket closed by peer")
            data = payload if opcode in (0x1, 0x2) else data + payload
            if fin:
                return data.decode("utf-8", "replace")

    # -- protocol --
    def call(self, method: str, params: Optional[dict] = None) -> dict:
        self._id += 1
        want = self._id
        self._send_frame(0x1, json.dumps(
            {"id": want, "method": method, "params": params or {}}).encode())
        while True:                              # skip unsolicited events
            msg = json.loads(self._recv_message())
            if msg.get("id") == want:
                if "error" in msg:
                    raise CdpError(str(msg["error"].get("message", msg["error"])))
                return msg.get("result", {})

    def close(self) -> None:
        try:
            self._send_frame(0x8, b"")
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "_WS":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# --- targets --------------------------------------------------------------- #

def targets() -> list[dict]:
    """Page targets from the browser's HTTP endpoint, minus devtools/extension
    pages. Empty list if the browser isn't running or the port isn't open."""
    try:
        with urllib.request.urlopen(
                f"{config.NARRATE_CDP_URL}/json/list",
                timeout=config.NARRATE_CDP_TIMEOUT) as r:
            listing = json.load(r)
    except (OSError, ValueError):
        return []
    out = []
    for t in listing:
        if t.get("type") != "page" or not t.get("webSocketDebuggerUrl"):
            continue
        url = t.get("url", "")
        if url.startswith(("devtools://", "chrome-extension://", "chrome://")):
            continue
        out.append(t)
    return out


def evaluate(target: dict, expression: str,
             timeout: float | None = None) -> Any:
    """Run `expression` in a target's page and return its value (by value)."""
    timeout = timeout or config.NARRATE_CDP_TIMEOUT
    with _WS(target["webSocketDebuggerUrl"], timeout) as ws:
        res = ws.call("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": True,
            # Pages that override globals (or run in a hostile frame) shouldn't be
            # able to trip us up; this also lets us read cross-origin-ish frames.
            "userGesture": False,
        })
        if res.get("exceptionDetails"):
            raise CdpError(str(res["exceptionDetails"].get("text", "eval failed")))
        return res.get("result", {}).get("value")


def focused_page() -> Optional[dict]:
    """The page target the user is actually looking at, or None if the browser
    isn't the focused application.

    `document.hasFocus()` answers BOTH questions at once — it is only true for the
    focused tab of the focused window — so we need no KWin/window-manager query.
    Probed in parallel with a short timeout so one wedged tab can't stall the turn.
    """
    tabs = targets()
    if not tabs:
        return None

    def probe(t: dict) -> tuple[dict, bool, bool]:
        try:
            v = evaluate(t, "[document.hasFocus(),document.visibilityState==='visible']",
                         timeout=config.NARRATE_CDP_PROBE_TIMEOUT)
            return t, bool(v and v[0]), bool(v and v[1])
        except (CdpError, OSError, ValueError):
            return t, False, False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(probe, tabs))
    for t, has_focus, _visible in results:
        if has_focus:
            return t
    return None
