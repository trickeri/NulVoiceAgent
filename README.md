# NulVoiceAgent

A small, local **spoken-agent framework**: talk to your computer, it reasons and
talks back. It wires a mic to a speech-to-text daemon, a pluggable "brain", and a
text-to-speech daemon — nothing else. You bring the personality and the voice.

```
mic ─parec─▶ NulSpeech2Text (STT :48450) ─▶ BRAIN ─▶ kokoromodel (TTS :48490) ─▶ speakers
                                             │
                              chat    : Hermes / codex / claude (conversational)
                              command : schema classifier ─▶ allowlist gate
                              route   : hand a task to a named tmux pane
```

- **STT** — [NulSpeech2Text](https://github.com/trickeri/NulSpeech2Text) on `:48450`
  (`POST /inference`, multipart wav → text).
- **TTS** — a **Kokoro** server (`kokoromodel`) on `:48490` (`POST /speak` with
  `{"text","voice","speed","pitch"}` → `audio/wav`). Ships pointed at a **stock
  Kokoro voice** (`af_heart`); pick any voice in a profile.
- **Brain** — pluggable, **no local LLM**. Default chat brain is the
  [Hermes agent](https://github.com/NousResearch/hermes-agent) (persistent memory +
  skills, and it can hand tasks off to named terminal panes). `codex` and `claude`
  CLIs also work as brains and as fallbacks.

It ships **neutral**: a stock voice and a plain, read-only persona. You make it
yours with a **persona file** and **voice profiles** (below) — none of which live
in this repo.

---

## Install with an AI agent (recommended)

> **If you are an AI coding agent (e.g. Claude Code) installing this for a user,
> follow the steps in order. Stop and ask the user each question marked 🟢 using
> your interactive question tool — do not assume the answers. Pick the marked
> default if the user has no preference. This is a stdlib-only Python project — no
> `pip install`, no venv.**

### 1. Prerequisites (check, install what's missing)

- **Python 3.10+** (stdlib only).
- **Audio**: `parec` and `pw-play` (PipeWire — Arch: `pipewire-pulse`; Debian:
  `pipewire-pulse` / `pulseaudio-utils`). `paplay` works as a fallback player.
- **tmux** — only needed for `route` mode (talking to named panes). Optional otherwise.
- **The two model daemons must be running** (each is its own project):
  - **NulSpeech2Text** on `:48450` — install from
    <https://github.com/trickeri/NulSpeech2Text> (it has its own agent-install README).
  - **kokoromodel** (a Kokoro TTS server) on `:48490`, answering `POST /speak`.
    Any server matching that contract works; set `NULVOICEAGENT_KOKORO_URL` if it
    lives elsewhere.
- **At least one brain CLI** on PATH — see the next question.

Verify the daemons before continuing:
```bash
curl -fsS http://127.0.0.1:48450/ >/dev/null && echo "STT up"
curl -fsS -X POST http://127.0.0.1:48490/speak -H 'content-type: application/json' \
  -d '{"text":"hello"}' -o /tmp/nva-test.wav && echo "TTS up"
```

### 2. 🟢 Ask: which chat brain?

*"Which brain should the conversational side use — Hermes (memory + can route to
panes), Codex, or Claude?"*

- **Hermes** *(default, recommended)* — the [Hermes agent](https://github.com/NousResearch/hermes-agent).
  Persistent memory + skills, and it's the brain that can hand a task to a named
  tmux pane. Needs one-time setup (step 4).
- **Codex** — thin `codex exec`, read-only. Reads the persona from `AGENTS.md`.
- **Claude** — headless `claude -p`, read-only. Gets the persona as a system prompt.

Whatever they pick, also make sure **`claude`** is installed if possible — it's the
default *fallback* brain (used when the primary is out of tokens) and powers the
optional completion-summary feature. Set the choice with `NULVOICEAGENT_CHAT_BRAIN`.

### 3. Clone + put the launcher on PATH

```bash
mkdir -p ~/programming && cd ~/programming
git clone https://github.com/trickeri/NulVoiceAgent.git
ln -sf "$PWD/NulVoiceAgent/bin/nulvoiceagent" ~/.local/bin/nulvoiceagent
nulvoiceagent say "installation working"   # should speak, if the TTS daemon is up
```

### 4. Set up the chat brain

- **Hermes**: install it (<https://github.com/NousResearch/hermes-agent>), then log
  it into a provider and pick a model **once**:
  ```bash
  hermes auth add <provider>   # e.g. openai-codex, anthropic — links your account
  hermes model                 # choose the provider + default model
  hermes -z "say hi"           # smoke test: should print a reply
  ```
  Give the agent its personality in Hermes (its `SOUL.md` / memory), since the
  Hermes brain uses that instead of this repo's `AGENTS.md`.
- **Codex / Claude**: just make sure the CLI is installed and authenticated
  (`codex` / `claude`). The persona comes from this repo's `AGENTS.md` (or a user
  override — see *Make it yours*).

### 5. Seed your config (voice, allowlist, persona)

Config lives in `~/.config/nulvoiceagent/` — never in the repo. Copy the examples:
```bash
mkdir -p ~/.config/nulvoiceagent/profiles ~/.config/nulvoiceagent/apps
cp examples/profiles/default.conf ~/.config/nulvoiceagent/profiles/default.conf
echo default > ~/.config/nulvoiceagent/active
cp examples/actions.conf ~/.config/nulvoiceagent/actions.conf
```
- 🟢 **Ask: which voice?** *"Which Kokoro voice should the agent use?"* Put it in
  `profiles/default.conf` as `VOICE=<id>` (default `af_heart`).
- 🟢 **Ask: give it a personality?** Offer to edit `AGENTS.md` (or create
  `~/.config/nulvoiceagent/persona.md`) to name the agent and shape how it talks.

### 6. 🟢 Ask: enable the Claude Code / Codex extras? (optional)

*"Want spoken 'agent is done' announcements and voice-answering of Claude's
AskUserQuestion prompts?"* If yes, merge `examples/claude-code-settings.json` into
`~/.claude/settings.json` and/or drop `examples/codex-hooks.json` at a project's
`.codex/hooks.json`. For **route mode**, also merge `examples/tmux.conf` into your
tmux config so panes show titles and prune themselves.

### 7. 🟢 Ask: bind hotkeys?

The agent runs **per-press**, not as a daemon — bind a key to each mode in your
desktop environment:
- `nulvoiceagent chat` — push-to-talk conversation (toggle: press to start, press
  again to stop). Suggested `Meta+``.
- `nulvoiceagent command` — one auto-endpointed command. Suggested `Meta+Escape`.
- `nulvoiceagent route` — record + hand off to a named pane (toggle).

On KDE, `examples/kde/nulvoiceagent-chat.desktop` registers `Meta+`` when dropped in
`~/.local/share/applications/`. On other DEs, bind the commands however you bind
shortcuts.

### 8. Verify

```bash
nulvoiceagent ask "what time is it"     # text in → brain → spoken reply (mic-free)
nulvoiceagent do  "lock the screen"     # text in → classify → allowlisted action
```

---

## Modes

| Command | Flow |
|---|---|
| `nulvoiceagent chat` | mic → STT → route? → brain (chats, may run 1 allowlisted action or hand off to a pane) → speak |
| `nulvoiceagent command` | mic → STT → classify → run allowlisted action → speak |
| `nulvoiceagent route` | mic (toggle) → STT → dispatch to a named tmux pane → speak |
| `nulvoiceagent ask "<text>"` | text → chat brain → speak (mic-free) |
| `nulvoiceagent do "<text>"` | text → classify → action → speak (mic-free) |
| `nulvoiceagent dispatch "<text>"` | text → route to a named pane (mic-free) |
| `nulvoiceagent targets [forget <name>]` | list / drop named panes |
| `nulvoiceagent say "<text>"` | just speak (TTS check) |
| `nulvoiceagent newtopic` | forget chat context (codex/claude sessions) |
| `nulvoiceagent claude-done` / `codex-done` | Stop-hook: announce done + offer summary |
| `nulvoiceagent ask-pending` / `ask-clear` | Claude Code AskUserQuestion hooks |
| `… --brain hermes\|codex\|claude` | override the brain for one run |

**Barge-in:** pressing a voice hotkey while the agent is thinking or speaking stops
it and starts listening again in one press.

## Routing commands to named tmux panes

Address an individual terminal pane (e.g. a Claude Code session) by voice:

- **name** — focus a pane, hit the chat/route key, say *"name this pane Build Claude"* →
  the registry stores `Build Claude → %id` and titles the pane.
- **dispatch** — *"tell Build Claude to add a double jump"* → bracketed-pastes the
  request into that pane and submits. Names are fuzzy-matched so STT slips are OK.
- **Hermes hand-off** — in `chat` mode the Hermes brain is told which panes are
  open; if you ask it to do something it has no built-in action for and a relevant
  builder pane exists, it forwards the task there itself (`[[ask: <pane> | …]]`,
  validated against the live registry before anything is sent).

Registry is data at `~/.config/nulvoiceagent/targets.json`. Route mode is pure tmux —
a pane's `%id` is reachable with no window focus. Needs the `examples/tmux.conf` bits.

## Make it yours

- **Personality** — edit `AGENTS.md` (read by the codex/claude brains), or drop
  `~/.config/nulvoiceagent/persona.md` to override it without touching the repo.
  With the Hermes brain, shape the agent in Hermes' own memory instead.
- **Voice** — profiles at `~/.config/nulvoiceagent/profiles/<name>.conf` set
  `VOICE`/`SPEED`/`PITCH` and an optional output `TTS_SINK`. Switch the active
  profile by writing its name into `~/.config/nulvoiceagent/active`. Want effects?
  Route `TTS_SINK` into your own PipeWire chain — the framework stays dry by default.
- **Commands** — the allowlist (`actions.conf`, and per-app `apps/<app>.conf`) is
  the only thing the command brain can run. Add a `name = command` line; no restart.
- **Parameterized commands** — for a command that needs a value from speech (a
  size, a filename), add a regex parser in `nulvoiceagent/intents.py` (ships empty,
  with a commented example).

## Config (env)

Everything is env-overridable (`NULVOICEAGENT_*`). Common ones:

| Var | Default | Meaning |
|---|---|---|
| `NULVOICEAGENT_STT_URL` | `http://127.0.0.1:48450/inference` | NulSpeech2Text endpoint |
| `NULVOICEAGENT_KOKORO_URL` | `http://127.0.0.1:48490/speak` | Kokoro TTS endpoint |
| `NULVOICEAGENT_CHAT_BRAIN` | `hermes` | `hermes` \| `codex` \| `claude` |
| `NULVOICEAGENT_CMD_BRAIN` | `codex` | classifier brain (`codex` \| `claude`) |
| `NULVOICEAGENT_CHAT_FALLBACK` / `_CMD_FALLBACK` | `claude` | brain to retry on when the primary is out of tokens |
| `NULVOICEAGENT_VOICE` / `_SPEED` / `_PITCH` | profile | override the active voice profile |
| `NULVOICEAGENT_TTS_SINK` | *(default sink)* | play into a named PipeWire sink |
| `NULVOICEAGENT_SOURCE` | *(default input)* | pin a specific mic (`pactl list sources short`) |

See `nulvoiceagent/config.py` for the full set (mic gate, done-announcer, ask-answer,
route timings).

## Safety

Both brains run **read-only** (codex `-s read-only`; claude with Write/Edit/Bash
disallowed). Command mode never lets the model *run* anything — it only picks a
*name* from your allowlist, which `nulvoiceagent` maps to the command. The model
can't invent or inject a command.
