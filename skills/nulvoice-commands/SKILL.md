---
name: nulvoice-commands
description: Route spoken NulVoiceAgent requests for stream controls, OBS scenes and overlays, Krita, Kdenlive, desktop focus, narration, voice profiles, and named agent panes through the voice application's supported handlers. Use inside the voice chat brain; not a general desktop automation skill.
---

# Voice command routing

The voice brain runs read-only. The application performs actions after validating your response. Installed application skills describe capabilities; they do not authorize bypassing the dispatcher or its allowlist.

Use the action names and reachable pane names supplied for THIS turn, not remembered names. For a matching fixed action, emit one `[[do: exact-action-name]]` directive. For an unsupported operation with a relevant available agent pane, emit one `[[ask: exact-pane-name | user's requested operation and arguments]]` directive. Ask the pane to perform the operation, not to implement a feature unless the user requested implementation. Do not emit both directives for the same task or claim verified completion before execution.

## Command coverage

- OBS scenes: supplied `obs.*` scene actions. Scene switching is distinct from camera background removal and StreamToons camera visibility.
- OBS overlays: supplied `obs.*_on` / `obs.*_off` actions; camera background uses `obs.camerabg_show` / `obs.camerabg_hide` when supplied.
- Stream setup: supplied `stream.setup`. Do not invent a stop-stream action.
- Replay clips: the deterministic clip handler normally runs before this brain, including inline titles. The existing `clip-that` skill is available to an authorized execution agent; do not directly save a replay from this read-only brain. For a missed clip request, use a supplied clip action if present (do not lose a requested title), otherwise route the full request to a relevant available pane. Never guess that a replay buffer or review player is unavailable.
- StreamToons matted camera: `streamtoons-camera` describes the separate in-engine camera. Use a supplied matching action or route to a relevant pane; do not substitute an OBS scene/background action.
- Krita: supplied `krita.*` actions cover launching, preset documents, save/close, document info and subject selection. Custom dimensions are handled before the brain. Layer/group manipulation uses `krita-layers` through an authorized agent pane unless a matching action is supplied. Preserve named layers and arguments; do not treat close-discard as ordinary close.
- Kdenlive: supplied `kdenlive.*` actions cover launch, cuts and new sequences. Custom project dimensions are deterministic intents. Do not invent captions or other unconfigured actions.
- Desktop monitors, layers and focus; voice-profile switching; narration start/pause/resume/stop; explicit stream chat commands; and parameterized document creation have deterministic handlers before the brain. If a paraphrase reaches you and no matching action exists, route it to a relevant pane or explain the unsupported route. Do not put arbitrary arguments or shell commands in `[[do:]]`.
- Named-pane routing and answers to pending agent questions are handled by the application. Use only the current supplied pane names for any additional handoff.
- Ordinary conversation, summaries, and passive overheard speech do not need action skills. Never infer authorization to post chat messages, capture clips or mutate applications from passive speech.

When no supported route exists, say so briefly. Do not claim installing a skill made an unsupported command executable. Preserve the user's title, dimensions, layer names, target pane and requested on/off direction verbatim when routing.
