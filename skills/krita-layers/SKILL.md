---
name: krita-layers
description: "Use when the user asks (by voice) to rearrange/reorganize/manage layers\
  \ or groups in the OPEN Krita document \u2014 move or reparent a layer, move it\
  \ into/out of a group, reorder, create/delete a group, rename, show/hide, set active.\
  \ Drives the live document over the nulpaint bridge using the `nulpaint` CLI; a\
  \ true node move preserving blend mode/opacity/masks."
license: MIT
---

## Voice-agent execution boundary

When running as the read-only NulVoiceAgent chat brain, do not run the mutating commands below directly. Use a matching action from the current supplied allowlist, or route the exact task to a relevant supplied agent pane with `[[ask: pane name | request]]`. If neither route exists, report that this operation is not connected to the voice dispatcher. Skill installation does not grant execution permission. In an authorized interactive coding session, follow the commands below and verify the result.


# Krita live layer & group management

## Overview

Troy runs a forked Krita with the **nulpaint** plugin, which exposes a loopback bridge so an
agent can manipulate the **currently open document** in real time. This skill lets you fulfil
spoken requests like "move the cutout layers into their groups", "put Magi1 cut above Magi1",
"make a new group called Cutouts and move all the cut layers into it", "delete that empty
group", "rename the group to Heroes", or "hide the background layer" — and Troy watches the
Layers docker update as you work.

Every change goes through bridge commands on Krita's GUI thread. A move is a **true node move**
(re-parent), so blend mode, opacity, masks, and layer styles are preserved — it is NOT a pixel
copy. No files are touched and no Krita restart is needed.

## When to Use

- The user asks to reorganize / rearrange / move / group / ungroup / reorder / rename / delete /
  show / hide layers or groups in Krita.
- The user references layer or group names from the open document ("the Cyren cuts", "the Magi
  group", "the background").

## Don't use for

- Painting, filters, AI generation, or selections → those are other nulpaint verbs
  (`nulpaint inpaint|outpaint|style|select-subject|segment-layers`), not this skill.
- Editing a `.kra` file that isn't open in Krita — this only drives the live, open document.

## Prerequisites (check first)

1. Krita is running with the target document **open and active**:
   ```bash
   nulpaint ping            # -> pong
   nulpaint call node.tree  # -> the layer tree as JSON
   ```
   If `ping` fails, ask Troy to open Krita and the document. If a `node.*` call returns
   `unknown command`, the plugin predates these commands and needs a **Krita restart** — but
   **never suggest restarting without saving first** (the doc may be untitled/unsaved).

## Commands (via `nulpaint call <cmd> --args '<json>'`)

- **`node.tree {path?|uuid?, depth?, text?}`** → nested `{name, uuid, type, visible, children}`.
  ALWAYS read this first to learn the structure and grab uuids.
- **`node.move {node|uuid|path, parent?, above?, below?}`** → reparent/reorder. `parent` is the
  target group (name or uuid; omit to keep current parent); `above`/`below` is a sibling to sit
  directly above/below; with neither, the node goes to the TOP of the parent.
- **`node.create_group {name, parent?, above?}`** → new group; returns its uuid.
- **`node.delete {node|uuid|path}`** → delete a node and its children.
- **`node.rename {node|uuid|path, new_name}`**.
- **`node.set_active {node|uuid|path}`**, **`node.set_visible {uuid|path, visible}`**.
- **`layer.list`** → flat name/type/depth list.

Layers are addressed by **uuid** (preferred — stable and unambiguous), **name** (first match;
fine when unique), or **path** (list of names from the doc root, e.g.
`["Cyrens","Cyren1"]`) for duplicate names.

## How to work

1. **Read** `nulpaint call node.tree` and parse the JSON — build name→uuid and layer→group maps.
2. **Confirm** anything ambiguous out loud ("above or below the original?", "which group?").
3. **Act** with the commands above. For many moves, a tiny Python driver is cleaner than many
   CLI calls:
   ```bash
   PYTHONPATH=/home/trikeri/programming/Krita/nulpaint/src \
     /home/trikeri/programming/Krita/nulpaint/.venv/bin/python - <<'PY'
   from nulpaint.bridge import BridgeClient
   with BridgeClient() as c:
       tree = c.call("node.tree")["tree"]
       # ... compute moves from the tree ...
       c.call("node.move", node="Magi1 (cut)", parent="Magi", above="Magi1")
   PY
   ```
4. **Verify** by re-reading `node.tree` and confirm the structure matches the request before
   telling the user it's done. Optionally `nulpaint call app.grab_canvas --args '{"path":"/tmp/k.png"}'`.

## Examples

- "Move every cut layer next to its original" → read the tree, for each `"<X> (cut)"` find `X`
  and its group, then `node.move(node="<X> (cut)", parent=<X's group>, above="X")`.
- "Group the Cyren cutouts" → `node.create_group(name="Cyren Cutouts", parent="Cyrens")`, then
  `node.move` each `"Cyren* (cut)"` into it.
- "Delete the empty AutoSegmented group" → `node.delete(node="AutoSegmented")`.

Single quick action over the CLI:
```bash
nulpaint call node.move --args '{"node":"Background","parent":"Scene","above":"Foreground"}'
```
