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
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

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
    DEFAULT_API_URL,
    DEFAULT_MAX_HISTORY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINK,
    DEFAULT_TIMEOUT,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_URL, default=DEFAULT_API_URL): str,
        vol.Optional(CONF_API_KEY, default=""): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
    }
)


async def async_fetch_models(hass: HomeAssistant, api_url: str, api_key: str) -> list[str]:
    """Return model ids from the OpenAI-compatible /models endpoint.

    Raises PermissionError on 401/403 and ConnectionError on other failures.
    """
    session = async_get_clientsession(hass)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = api_url.rstrip("/") + "/models"
    async with session.get(
        url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
    ) as resp:
        if resp.status in (401, 403):
            raise PermissionError
        if resp.status >= 400:
            raise ConnectionError(f"HTTP {resp.status}")
        data = await resp.json(content_type=None)
    models: list[str] = []
    for item in data.get("data", []) if isinstance(data, dict) else []:
        model_id = item.get("id") if isinstance(item, dict) else None
        if model_id:
            models.append(str(model_id))
    return sorted(set(models))


def _model_selector(models: list[str], current: str | None) -> SelectSelector:
    """Dropdown of server models, free text allowed, current value always present."""
    options = list(models)
    if current and current not in options:
        options.insert(0, current)
    if not options:
        options = [DEFAULT_MODEL_NAME]
    return SelectSelector(
        SelectSelectorConfig(
            options=[SelectOptionDict(label=m, value=m) for m in options],
            custom_value=True,
            sort=False,
        )
    )


class UnslothConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the initial setup: server, then model."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize."""
        self._server: dict[str, Any] = {}
        self._models: list[str] = []

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Step 1: server URL and key. Validates and fetches the model list."""
        errors: dict[str, str] = {}

        if user_input is not None:
            api_url = user_input[CONF_API_URL].strip()
            api_key = user_input.get(CONF_API_KEY, "")
            if not api_url.startswith(("http://", "https://")):
                errors[CONF_API_URL] = "invalid_url"
            else:
                try:
                    self._models = await async_fetch_models(self.hass, api_url, api_key)
                except PermissionError:
                    errors["base"] = "invalid_auth"
                except Exception:  # noqa: BLE001
                    _LOGGER.exception("Could not reach %s", api_url)
                    errors["base"] = "cannot_connect"

            if not errors:
                await self.async_set_unique_id(api_url.lower())
                self._abort_if_unique_id_configured()
                self._server = {CONF_API_URL: api_url, CONF_API_KEY: api_key}
                return await self.async_step_model()

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_SCHEMA, errors=errors
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Step 2: pick a model from what the server reports."""
        if user_input is not None:
            model = user_input[CONF_MODEL_NAME].strip()
            return self.async_create_entry(
                title=f"Unsloth AI ({model})",
                data=self._server,
                options={CONF_MODEL_NAME: model},
            )

        default = self._models[0] if self._models else DEFAULT_MODEL_NAME
        for m in self._models:
            if "gemma" in m.lower():
                default = m
                break
        schema = vol.Schema(
            {
                vol.Required(CONF_MODEL_NAME, default=default): _model_selector(
                    self._models, None
                ),
            }
        )
        return self.async_show_form(
            step_id="model",
            data_schema=schema,
            description_placeholders={"count": str(len(self._models))},
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Return the options flow."""
        return UnslothOptionsFlow()


class UnslothOptionsFlow(OptionsFlow):
    """Options: model, prompt, LLM API, sampling, history."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            # Empty list = no device control; drop the key so it reads as "off".
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
            user_input[CONF_MODEL_NAME] = user_input[CONF_MODEL_NAME].strip()
            return self.async_create_entry(title="", data=user_input)

        entry = self.config_entry
        opts = entry.options
        current_model = opts.get(CONF_MODEL_NAME) or entry.data.get(CONF_MODEL_NAME)

        try:
            models = await async_fetch_models(
                self.hass, entry.data[CONF_API_URL], entry.data.get(CONF_API_KEY, "")
            )
        except Exception:  # noqa: BLE001
            _LOGGER.warning("Could not list models from %s", entry.data[CONF_API_URL])
            models = []

        apis = llm.async_get_apis(self.hass)
        valid_ids = [api.id for api in apis]
        selected = [a for a in opts.get(CONF_LLM_HASS_API, []) if a in valid_ids]
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_MODEL_NAME, default=current_model or DEFAULT_MODEL_NAME
                ): _model_selector(models, current_model),
                vol.Optional(
                    CONF_PROMPT,
                    description={
                        "suggested_value": opts.get(
                            CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                        )
                    },
                ): TemplateSelector(),
                vol.Optional(
                    CONF_LLM_HASS_API,
                    description={"suggested_value": selected},
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(label=api.name, value=api.id)
                            for api in apis
                        ],
                        multiple=True,
                    )
                ),
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
                vol.Optional(
                    CONF_THINK, default=opts.get(CONF_THINK, DEFAULT_THINK)
                ): bool,
                vol.Optional(
                    CONF_MAX_HISTORY,
                    default=opts.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY),
                ): NumberSelector(
                    NumberSelectorConfig(min=0, max=100, step=1, mode=NumberSelectorMode.BOX)
                ),
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema)
