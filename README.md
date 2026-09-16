<div align="center">

# 📐 pi-pareto-inference-provider

**GLM-5.3 Flash on [Pareto Inference](https://paretoinference.com) — Pareto's own GPUs, pay per token**

_OpenAI-compatible provider extension for [pi](https://github.com/earendil-works/pi-coding-agent)._

[![pi extension](https://img.shields.io/badge/pi-extension-blueviolet)](https://github.com/earendil-works/pi-coding-agent)
[![npm](https://img.shields.io/npm/v/pi-pareto-inference-provider)](https://www.npmjs.com/package/pi-pareto-inference-provider)
[![license](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

</div>

---

## Features

- **3+ AI Models** — `z-ai/glm-5.3-flash` (the documented offer), `z-ai/glm-5.3`, and `deepseek/deepseek-v4-flash`, synced live from Pareto's public `GET /v1/models`
- **Endpoint-verified thinking ladder** — `/thinking` levels map one-to-one to `reasoning_effort` (`off → none`, `minimal → minimal`, `low → low`, `medium → medium`, `high → high`, `max → max`); `xhigh` is hidden because the API returns HTTP 500 for it
- **Image input** — verified against the live endpoint (1×1 PNG round-trip), so image attachments work on `z-ai/glm-5.3-flash`
- **Published token pricing** — $0.03/M input, $0.10/M output, $0.006/M cached input for GLM-5.3 Flash ([pricing](https://docs.paretoinference.com/pricing))
- **Stale-While-Revalidate catalog** — instant startup from disk cache → embedded `models.json`, background refresh on session start, hot-swap without restart, 14-day grace period for delisted models
- **Key-free catalog sync** — `GET /v1/models` is public, so model lists refresh even before auth is configured
- **Tool-call safe** — verified multi-turn replay (assistant tool calls without `reasoning_content`) and streaming tool calls via [synbad](https://github.com/synthetic-lab/synbad)

## Available Models

| Model | Type | Context | Max Tokens | Input Cost | Output Cost |
|-------|------|---------|------------|------------|-------------|
| DeepSeek V4 Flash | Text | 131K | 131K | — | — |
| GLM 5.3 | Text | 131K | 131K | — | — |
| GLM 5.3 Flash | Text + Image | 131K | 131K | $0.03 | $0.10 |
*Costs are per million tokens. Pareto publishes pricing for GLM-5.3 Flash only. Pareto has not published deployment context/output limits — ask Pareto for them; until then this extension ships a conservative 131,072-token context and output floor derived from the documented `max_tokens` range (1–131,072). Raise the values in `patch.json` once real limits are published.*

Pareto also accepts the alias `glm-5.3-flash` for `z-ai/glm-5.3-flash`.

## Installation

### Option 1: Using `pi install` (Recommended)

Install from npm:

```bash
pi install npm:pi-pareto-inference-provider
```

Or directly from GitHub:

```bash
pi install https://github.com/monotykamary/pi-pareto-inference-provider
```

### Option 2: Manual Clone

```bash
git clone https://github.com/monotykamary/pi-pareto-inference-provider.git
cd pi-pareto-inference-provider
pi -e /path/to/pi-pareto-inference-provider
```

## Authentication

Get your key from the [Pareto dashboard](https://paretoinference.com/dashboard). Pick one:

```bash
# Option 1 (recommended): store in ~/.pi/agent/auth.json
#   "pareto": { "type": "api_key", "key": "your-api-key" }

# Option 2: environment variable
export PARETO_API_KEY=your-api-key
```

Secret managers work too — pi resolves `!command` values, so an `auth.json` entry like this keeps the key out of your shell history:

```json
"pareto": { "type": "api_key", "key": "!localterm secret get pareto_inference_api_key" }
```

## Usage

Select a model with `/model`, or start directly:

```bash
pi --provider pareto --model z-ai/glm-5.3-flash
```

### Thinking levels

```bash
pi --provider pareto --model z-ai/glm-5.3-flash --thinking high
```

| pi level | sent as `reasoning_effort` |
|----------|----------------------------|
| `off` | `none` |
| `minimal` | `minimal` |
| `low` | `low` |
| `medium` | `medium` |
| `high` | `high` |
| `max` | `max` |
| `xhigh` | hidden — Pareto returns HTTP 500 |

`off → none` is sent explicitly instead of dropping the field, so `/thinking off` really disables reasoning.

### Compat notes

- `store: true` is rejected by the API (`unsupported_storage`), so requests send `store: false`
- `max_tokens` is used (documented cap 131,072); `max_completion_tokens` also works
- `developer` role is avoided (`supportsDeveloperRole: false`); system messages are used
- Replays of assistant tool calls without `reasoning_content` are accepted as-is

## Model Sync

`models.json` is generated from Pareto's public catalog:

```bash
node scripts/update-models.js
```

It fetches `GET https://api.paretoinference.com/v1/models` (no key required), preserves curated fields for known ids, moves delisted models into `deprecated-models.json` with a 14-day grace period, and regenerates the table above. See `AGENTS.md` for the patch pipeline (`patch.json` for per-model overrides, `custom-models.json` for models absent from the catalog).

## Verification

[Synbad](https://github.com/synthetic-lab/synbad) — Synthetic's inference-provider eval suite — is the reference check for tool calling and reasoning parsing:

```bash
cd synbad
PARETO_API_KEY=<your-key> ./synbad.sh eval --env-var PARETO_API_KEY \
  --base-url "https://api.paretoinference.com/v1" \
  --model "z-ai/glm-5.3-flash" --count 1
PARETO_API_KEY=<your-key> ./synbad.sh eval --env-var PARETO_API_KEY \
  --base-url "https://api.paretoinference.com/v1" \
  --model "z-ai/glm-5.3-flash" --count 1 --stream
```

Result (2026-09-15, `count=1`):

| Mode | Result | Failing eval |
|------|--------|--------------|
| non-streaming | **14/15** | `reasoning/multiturn-reasoning-parsing` — HTTP 504 |
| streaming | **14/15** | `reasoning/reasoning-claude-tool-call` — textual `<invoke>` instead of structured `tool_calls` |

Both failures are provider behavior, not extension behavior:

- **Non-streaming 504 (reproducible).** The eval replays a short reasoning history and sends no `max_tokens`; the model reasons past Pareto's non-streaming gateway timeout (~65–70 s) and the API returns `{"error":{"message":"The model request failed.","code":"504"}}`. The identical request succeeds with `--stream` (HTTP 200, 7,282 completion tokens, 75 s), and even a non-stream `max_tokens: 8192` still 504s at ~67 s. pi always streams, so the extension is unaffected. Request IDs for follow-up: `d66f9a22-b3c4-49b3-a825-92655ca2a2e7`, `dca81a2f-cba5-4f7b-97dd-2899397b3b73`.
- **Flaky Claude-style tool call.** `reasoning/reasoning-claude-tool-call` runs at `temperature: 1` and asks the model to "put the tool call inside your thinking"; some samples emit `<invoke name="Bash">` as text in `content` instead of a structured `tool_calls` entry. It passed the non-stream `count=1` run, failed the stream `count=1` run, failed at least one of three non-stream samples, and passed 3/3 stream samples — sampling variance, not a wiring bug.

All other evals passed in both modes (14 tool/reasoning evals including `simple-tool`, `parallel-tool`, `multi-turn-tools`, `stream-multi-tool`, `tool-path-corruption`, and the reasoning-parsing set). The extension itself was validated end to end in pi: `--list-models` shows the three catalog entries, a headless `--thinking off` prompt returns `PARETO_OK`, and a `read` tool round-trip returns the file's contents.

## License

MIT — see [LICENSE](./LICENSE).
