"""Conversation entity that talks to an Unsloth-served model via an OpenAI-compatible API.

Supports Home Assistant's LLM tool API (Assist device control) using OpenAI-style
function calling against /v1/chat/completions.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Literal

import aiohttp

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.json import json_dumps

from .const import (
    CONF_API_KEY,
    CONF_API_URL,
    CONF_MAX_HISTORY,
    CONF_MAX_TOKENS,
    CONF_MODEL_NAME,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_THINK,
    CONF_TIMEOUT,
    DEFAULT_MAX_HISTORY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINK,
    DEFAULT_TIMEOUT,
    DOMAIN,
    MAX_TOOL_ITERATIONS,
)

_LOGGER = logging.getLogger(__name__)

# Gemma sometimes emits its native tool-call syntax as plain text when the
# inference server did not parse it, e.g.
#   <|tool_call>call:HassTurnOn(name="bedroom light")<tool_call|>
_RAW_TOOL_CALL = re.compile(
    r"<\|tool_call\>\s*call:(?P<name>[\w.]+)\((?P<args>.*?)\)\s*<tool_call\|>",
    re.DOTALL,
)


def _schema_to_openapi(schema: Any, custom_serializer: Any) -> dict[str, Any]:
    """Convert a tool parameter schema to JSON schema, whichever helper core ships."""
    try:
        import probatio  # noqa: PLC0415

        return probatio.to_openapi(schema, custom_serializer=custom_serializer)
    except ImportError:
        from voluptuous_openapi import convert  # noqa: PLC0415

        return convert(schema, custom_serializer=custom_serializer)


def _format_tool(tool: llm.Tool, custom_serializer: Any) -> dict[str, Any]:
    """Format an HA tool as an OpenAI function tool."""
    spec: dict[str, Any] = {
        "name": tool.name,
        "parameters": _schema_to_openapi(tool.parameters, custom_serializer),
    }
    if tool.description:
        spec["description"] = tool.description
    return {"type": "function", "function": spec}


def _parse_args(raw: Any) -> dict[str, Any]:
    """Parse tool arguments (JSON string or dict) and drop empty values."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            _LOGGER.warning("Could not parse tool arguments: %s", raw)
            raw = {}
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if v is not None and v != ""}


def _parse_raw_tool_calls(text: str) -> tuple[str, list[llm.ToolInput]]:
    """Recover tool calls the model wrote as text. Returns (clean_text, calls)."""
    calls: list[llm.ToolInput] = []

    def _repl(match: re.Match[str]) -> str:
        name = match.group("name").split(".")[-1]  # strip "homeassistant." prefix
        args: dict[str, Any] = {}
        try:
            call = ast.parse(f"f({match.group('args')})", mode="eval").body
            for kw in call.keywords:  # type: ignore[attr-defined]
                args[kw.arg] = ast.literal_eval(kw.value)
        except (SyntaxError, ValueError):
            _LOGGER.warning("Could not parse raw tool call: %s", match.group(0))
        calls.append(llm.ToolInput(tool_name=name, tool_args=_parse_args(args)))
        return ""

    return _RAW_TOOL_CALL.sub(_repl, text).strip(), calls


def _trim_history(messages: list[dict[str, Any]], max_turns: int) -> list[dict[str, Any]]:
    """Keep the system prompt, the last `max_turns` previous user turns, and the current turn.

    A "turn" starts at a user message and includes the assistant/tool messages
    that follow it. max_turns <= 0 means unlimited.
    """
    if max_turns <= 0:
        return messages
    system = [m for m in messages[:1] if m["role"] == "system"]
    body = messages[len(system):]
    user_idx = [i for i, m in enumerate(body) if m["role"] == "user"]
    # user_idx[-1] is the in-progress message; keep max_turns before it.
    if len(user_idx) - 1 <= max_turns:
        return messages
    start = user_idx[-1 - max_turns]
    return system + body[start:]


def _content_to_message(content: Any) -> dict[str, Any] | None:
    """Convert a chat log entry into an OpenAI chat message."""
    if isinstance(content, conversation.SystemContent):
        return {"role": "system", "content": content.content}
    if isinstance(content, conversation.UserContent):
        return {"role": "user", "content": content.content}
    if isinstance(content, conversation.AssistantContent):
        msg: dict[str, Any] = {"role": "assistant", "content": content.content or ""}
        if content.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json_dumps(tc.tool_args),
                    },
                }
                for tc in content.tool_calls
            ]
        return msg
    if isinstance(content, conversation.ToolResultContent):
        return {
            "role": "tool",
            "tool_call_id": content.tool_call_id,
            "name": content.tool_name,
            "content": json_dumps(content.tool_result),
        }
    return None


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the conversation entity."""
    async_add_entities([UnslothConversationEntity(entry)])


class UnslothConversationEntity(conversation.ConversationEntity):
    """Conversation agent backed by /v1/chat/completions with tool calling."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Unsloth",
            model=entry.data[CONF_MODEL_NAME],
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """The model handles any language."""
        return MATCH_ALL

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Run the tool-calling loop against the model and return its reply."""
        opts = self.entry.options

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                opts.get(CONF_LLM_HASS_API),
                opts.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        try:
            await self._async_handle_chat_log(chat_log)
        except HomeAssistantError as err:
            response = intent.IntentResponse(language=user_input.language)
            response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, str(err))
            return conversation.ConversationResult(
                response=response, conversation_id=chat_log.conversation_id
            )

        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def _async_handle_chat_log(self, chat_log: conversation.ChatLog) -> None:
        """Send the chat log to the model, executing tools until it stops calling them."""
        data = self.entry.data
        opts = self.entry.options

        tools: list[dict[str, Any]] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        messages = [
            m for c in chat_log.content if (m := _content_to_message(c)) is not None
        ]
        messages = _trim_history(
            messages, int(opts.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY))
        )

        url = data[CONF_API_URL].rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if data.get(CONF_API_KEY):
            headers["Authorization"] = f"Bearer {data[CONF_API_KEY]}"
        timeout = aiohttp.ClientTimeout(total=float(opts.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)))
        session = async_get_clientsession(self.hass)

        for _ in range(MAX_TOOL_ITERATIONS):
            payload: dict[str, Any] = {
                "model": data[CONF_MODEL_NAME],
                "messages": messages,
                "temperature": float(opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)),
                "max_tokens": int(opts.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)),
                "stream": False,
            }
            if tools:
                payload["tools"] = tools
            # llama.cpp (--jinja) honours chat_template_kwargs; Ollama honours "think".
            # Other servers ignore unknown keys.
            think = bool(opts.get(CONF_THINK, DEFAULT_THINK))
            payload["chat_template_kwargs"] = {"enable_thinking": think}
            payload["think"] = think

            try:
                async with session.post(
                    url, json=payload, headers=headers, timeout=timeout
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        _LOGGER.error("Model server returned %s: %s", resp.status, body[:500])
                        raise HomeAssistantError(
                            f"The model server returned HTTP {resp.status}."
                        )
                    result = await resp.json()
            except TimeoutError as err:
                raise HomeAssistantError("The model server timed out.") from err
            except aiohttp.ClientError as err:
                _LOGGER.error("Error talking to %s: %s", url, err)
                raise HomeAssistantError("I couldn't reach the model server.") from err

            _LOGGER.debug("Model server response: %s", result)
            try:
                choice = result["choices"][0]
                message = choice["message"]
            except (KeyError, IndexError, TypeError):
                _LOGGER.error("Unexpected response shape: %s", result)
                raise HomeAssistantError("The model returned an unexpected response.")

            text = (message.get("content") or "").strip()
            thinking = (
                message.get("reasoning_content") or message.get("reasoning") or None
            )
            finish_reason = choice.get("finish_reason")
            tool_calls: list[llm.ToolInput] = []
            for tc in message.get("tool_calls") or []:
                fn = tc.get("function", {})
                tool_calls.append(
                    llm.ToolInput(
                        id=tc.get("id") or llm.ToolInput(tool_name="", tool_args={}).id,
                        tool_name=fn.get("name", ""),
                        tool_args=_parse_args(fn.get("arguments")),
                    )
                )

            # Fallback: model wrote the tool call as text.
            raw_text = text
            if not tool_calls and "<|tool_call>" in text:
                text, tool_calls = _parse_raw_tool_calls(text)

            # Only honor tool calls when an LLM API is active and the tool exists.
            if tool_calls and chat_log.llm_api:
                known = {t.name for t in chat_log.llm_api.tools}
                unknown = [tc.tool_name for tc in tool_calls if tc.tool_name not in known]
                if unknown:
                    _LOGGER.warning("Model called unknown tool(s): %s", unknown)
                tool_calls = [tc for tc in tool_calls if tc.tool_name in known]
            elif tool_calls:
                _LOGGER.warning(
                    "Model tried to call tools but no LLM API is enabled. "
                    "Enable 'Control Home Assistant' in the integration options."
                )
                tool_calls = []

            if not text and not tool_calls:
                # Never leave the user with a blank reply.
                if finish_reason == "length":
                    text = (
                        "My answer was cut off before I could reply. "
                        "Raise Max tokens or turn off Thinking in the integration options."
                    )
                elif "<|tool_call>" in raw_text:
                    text = (
                        "I tried to control a device, but device control is not enabled "
                        "or the tool doesn't exist."
                    )
                elif thinking:
                    text = "I thought about it but produced no answer. Try turning off Thinking."
                else:
                    text = "The model returned an empty response."
                _LOGGER.warning(
                    "Empty model reply (finish_reason=%s, thinking=%s): %s",
                    finish_reason, bool(thinking), result,
                )

            assistant = conversation.AssistantContent(
                agent_id=self.entity_id,
                content=text or None,
                thinking_content=thinking,
                tool_calls=tool_calls or None,
            )
            messages.append(_content_to_message(assistant))  # type: ignore[arg-type]

            async for tool_result in chat_log.async_add_assistant_content(assistant):
                messages.append(_content_to_message(tool_result))  # type: ignore[arg-type]

            if not chat_log.unresponded_tool_results:
                break
