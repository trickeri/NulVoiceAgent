---
name: streamtoons-camera
description: "Show or hide the StreamToons in-engine matted facecam by voice. Use\
  \ when Troy cuts OBS to (or away from) the raw full-camera scene \u2014 \"hide the\
  \ stream camera\", \"kill the toon cam\", \"bring the camera back\", \"show the\
  \ facecam again\". Runs one local command against the nul-chat-hub control endpoint."
---

## Voice-agent execution boundary

When running as the read-only NulVoiceAgent chat brain, do not run the mutating commands below directly. Use a matching action from the current supplied allowlist, or route the exact task to a relevant supplied agent pane with `[[ask: pane name | request]]`. If neither route exists, report that this operation is not connected to the voice dispatcher. Skill installation does not grant execution permission. In an authorized interactive coding session, follow the commands below and verify the result.


# StreamToons camera — remote show/hide

## What this controls

StreamToons draws Troy's camera **matted** (background removed) as an in-engine
overlay on the avatar stage. Separately, Troy sometimes cuts OBS to a scene that
shows the **raw** camera (no background removal). When he does that, the matted
in-engine feed should be **hidden** so it doesn't double up / conflict — and
**restored** when he cuts back to the avatar stage.

This skill flips that master switch. It does **not** touch OBS scenes; Troy
switches those himself and asks you to hide/show the toon cam to match.

## How to run it

There is one command. It talks to the nul-chat-hub HTTP endpoint, which fans a
`!camera` command out to the running StreamToons engine over its bridge socket.

```bash
streamtoons-cam off      # hide the in-engine matted camera
streamtoons-cam on       # show it again
streamtoons-cam toggle   # flip current state (use only if state is unknown)
```

The script lives at `~/programming/stream/nul-chat-hub/scripts/streamtoons-cam`.
If it is not on `$PATH`, call it by that full path. A successful call prints
`{"ok":true,"state":"off"}` (or `on`/`toggle`).

## Intent → command mapping

- "hide / kill / turn off / drop the stream camera (or toon cam / facecam)" → `streamtoons-cam off`
- "show / bring back / turn on / restore the camera" → `streamtoons-cam on`
- Ambiguous / "flip the camera" → `streamtoons-cam toggle`

Prefer explicit `off`/`on` over `toggle` — you usually know which direction Troy
means from context (he just switched OBS scenes). `off` and `on` are idempotent,
so repeating one is harmless.

## Notes / troubleshooting

- Requires the `nul-chat-hub` user service to be running
  (`systemctl --user is-active nul-chat-hub`). If the call fails with a
  connection error, the hub is down — say so rather than retrying blindly.
- The override persists until you flip it back; it survives the engine's
  per-frame authored camera state. So if you hid the camera, remember to show it
  again when Troy returns to the avatar scene.
- Override the hub URL with `HUB_URL=http://host:port` if ever needed (default
  `http://localhost:3737`).
