"""Unsloth AI Assistant - conversation agent backed by an OpenAI-compatible API."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import CONF_MODEL_NAME

PLATFORMS: list[Platform] = [Platform.CONVERSATION]


def _title_for(entry: ConfigEntry) -> str:
    """Entry title that reflects the currently selected model."""
    model = entry.options.get(CONF_MODEL_NAME) or entry.data.get(CONF_MODEL_NAME) or "?"
    return f"Unsloth AI ({model})"


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Unsloth AI from a config entry."""
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Rename the entry to match the model, then reload so the change takes effect."""
    title = _title_for(entry)
    if entry.title != title:
        # This re-triggers the listener; the second pass falls through to the reload.
        hass.config_entries.async_update_entry(entry, title=title)
        return
    await hass.config_entries.async_reload(entry.entry_id)
