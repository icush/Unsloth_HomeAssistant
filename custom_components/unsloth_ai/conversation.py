"""Conversation entity that talks to an Unsloth-served model via an OpenAI-compatible API."""

from __future__ import annotations

import logging
from typing import Literal

import aiohttp

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_API_KEY,
    CONF_API_URL,
    CONF_MAX_TOKENS,
    CONF_MODEL_NAME,
    CONF_SYSTEM_PROMPT,
    CONF_TEMPERATURE,
    CONF_TIMEOUT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the conversation entity."""
    async_add_entities([UnslothConversationEntity(entry)])


class UnslothConversationEntity(conversation.ConversationEntity):
    """Conversation agent backed by /v1/chat/completions."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.title,
            "manufacturer": "Unsloth",
            "model": entry.data[CONF_MODEL_NAME],
            "entry_type": "service",
        }

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """The model handles any language."""
        return MATCH_ALL

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Send the chat history to the model and return its reply."""
        data = self.entry.data
        opts = self.entry.options

        system_prompt = opts.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT)
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        # Replay prior turns from the chat log so multi-turn conversations work.
        for content in chat_log.content:
            if isinstance(content, conversation.UserContent):
                messages.append({"role": "user", "content": content.content})
            elif isinstance(content, conversation.AssistantContent) and content.content:
                messages.append({"role": "assistant", "content": content.content})

        # The current user message is already the last UserContent in the chat log;
        # guard against older cores where it is not.
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": user_input.text})

        payload = {
            "model": data[CONF_MODEL_NAME],
            "messages": messages,
            "temperature": float(opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)),
            "max_tokens": int(opts.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)),
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if data.get(CONF_API_KEY):
            headers["Authorization"] = f"Bearer {data[CONF_API_KEY]}"

        url = data[CONF_API_URL].rstrip("/") + "/chat/completions"
        timeout = aiohttp.ClientTimeout(total=float(opts.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)))
        session = async_get_clientsession(self.hass)

        try:
            async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    _LOGGER.error("Model server returned %s: %s", resp.status, body[:500])
                    return self._error_result(
                        user_input, f"The model server returned HTTP {resp.status}."
                    )
                result = await resp.json()
        except TimeoutError:
            return self._error_result(user_input, "The model server timed out.")
        except aiohttp.ClientError as err:
            _LOGGER.error("Error talking to %s: %s", url, err)
            return self._error_result(user_input, "I couldn't reach the model server.")

        try:
            reply = result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError):
            _LOGGER.error("Unexpected response shape: %s", result)
            return self._error_result(user_input, "The model returned an unexpected response.")

        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(agent_id=user_input.agent_id, content=reply)
        )

        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(reply)
        return conversation.ConversationResult(
            response=response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

    def _error_result(
        self, user_input: conversation.ConversationInput, message: str
    ) -> conversation.ConversationResult:
        """Build an error result."""
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, message)
        return conversation.ConversationResult(
            response=response, conversation_id=user_input.conversation_id
        )
