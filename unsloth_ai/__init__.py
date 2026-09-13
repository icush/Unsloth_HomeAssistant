from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from .const import DOMAIN

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Unsloth AI from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    # We store the entry in hass.data so the conversation platform can access it
    hass.data[DOMAIN] = entry
    return True