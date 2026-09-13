# Unsloth AI Assistant for Home Assistant

A Home Assistant conversation agent (Assist) that sends chat to a model served by
[Unsloth](https://unsloth.ai) — or any OpenAI-compatible server (vLLM, llama.cpp, LM Studio, Ollama).

## Install via HACS

1. HACS → Integrations → ⋮ → **Custom repositories**
2. Repository: `https://github.com/icush/Unsloth_HomeAssistant`, Type: **Integration**
3. Install **Unsloth AI Assistant**, then restart Home Assistant.

## Configure

Settings → Devices & Services → **Add Integration** → *Unsloth AI Assistant*.

| Field | Example |
|---|---|
| API base URL | `http://192.168.1.50:8000/v1` (llama.cpp / vLLM) or `http://192.168.1.50:11434/v1` (Ollama) |
| Model name | `unsloth/gemma-4-26B-A4B-it-GGUF` (must match what the server reports at `/v1/models`) |
| API key | optional |

Then Settings → Voice assistants → pick **Unsloth AI** as the conversation agent.

## Device control

Open the integration's **Configure** button and set **Control Home Assistant** to *Assist*.
Only entities that are **exposed** to Assist (Settings → Voice assistants → Expose) are visible
to the model. The model must be served with tool calling enabled:

```bash
llama-server -hf unsloth/gemma-4-26B-A4B-it-GGUF:Q4_K_M --host 0.0.0.0 --port 8000 --jinja
```

`--jinja` is required so llama.cpp parses Gemma's tool calls into OpenAI `tool_calls`
instead of leaving them as raw `<|tool_call>` text. (The integration also recovers raw
text tool calls as a fallback, but parsed calls are far more reliable.)

Instructions (system prompt template), temperature, max tokens and timeout are in the same
Configure dialog.

## Serving gemma-4-26B-A4B-it-GGUF

Unsloth's GGUF exports run on llama.cpp's server:

```bash
llama-server -hf unsloth/gemma-4-26B-A4B-it-GGUF:Q4_K_M --host 0.0.0.0 --port 8000
```

That exposes `http://<host>:8000/v1/chat/completions`, which is what this integration calls.
