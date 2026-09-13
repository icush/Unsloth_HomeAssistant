"""Config flow for the Unsloth AI Assistant integration."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    CONF_API_KEY,
    CONF_API_URL,
    CONF_MAX_TOKENS,
    CONF_MODEL_NAME,
    CONF_SYSTEM_PROMPT,
    CONF_TEMPERATURE,
    CONF_TIMEOUT,
    DEFAULT_API_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_URL, default=DEFAULT_API_URL): str,
        vol.Required(CONF_MODEL_NAME, default=DEFAULT_MODEL_NAME): str,
        vol.Optional(CONF_API_KEY, default=""): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
    }
)


async def _validate_endpoint(hass, api_url: str, api_key: str) -> None:
    """Hit /models on the OpenAI-compatible server to confirm it answers."""
    session = async_get_clientsession(hass)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = api_url.rstrip("/") + "/models"
    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
        if resp.status in (401, 403):
            raise PermissionError
        if resp.status >= 400:
            raise ConnectionError(f"HTTP {resp.status}")


class UnslothConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the initial setup."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the user step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            api_url = user_input[CONF_API_URL].strip()
            if not api_url.startswith(("http://", "https://")):
                errors[CONF_API_URL] = "invalid_url"
            else:
                try:
                    await _validate_endpoint(
                        self.hass, api_url, user_input.get(CONF_API_KEY, "")
                    )
                except PermissionError:
                    errors["base"] = "invalid_auth"
                except Exception:  # noqa: BLE001
                    _LOGGER.exception("Could not reach %s", api_url)
                    errors["base"] = "cannot_connect"

            if not errors:
                user_input[CONF_API_URL] = api_url
                await self.async_set_unique_id(api_url.lower())
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title=f"Unsloth AI ({user_input[CONF_MODEL_NAME]})",
                    data=user_input,
                )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_SCHEMA, errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Return the options flow."""
        return UnslothOptionsFlow()


class UnslothOptionsFlow(OptionsFlow):
    """Options: prompt, temperature, max tokens, timeout."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        opts = self.config_entry.options
        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_SYSTEM_PROMPT,
                    default=opts.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT),
                ): TextSelector(TextSelectorConfig(multiline=True)),
                vol.Optional(
                    CONF_TEMPERATURE,
                    default=opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                ): NumberSelector(
                    NumberSelectorConfig(min=0, max=2, step=0.05, mode=NumberSelectorMode.SLIDER)
                ),
                vol.Optional(
                    CONF_MAX_TOKENS,
                    default=opts.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                ): NumberSelector(
                    NumberSelectorConfig(min=16, max=8192, step=16, mode=NumberSelectorMode.BOX)
                ),
                vol.Optional(
                    CONF_TIMEOUT,
                    default=opts.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                ): NumberSelector(
                    NumberSelectorConfig(min=5, max=600, step=5, mode=NumberSelectorMode.BOX)
                ),
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema)
