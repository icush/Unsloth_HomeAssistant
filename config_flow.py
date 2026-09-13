import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import url
from .const import DOMAIN, CONF_API_URL, CONF_API_KEY, CONF_MODEL_NAME

class UnslothConfigFlow(config_entries.ConfigFlow):
    """Handle a config flow for Unsloth AI."""
    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Handle the user input."""
        errors = {}
        if user_input is not None:
            # Basic validation: Ensure URL starts with http
            if not user_input[CONF_API_URL].startswith("http"):
                errors["api_url"] = "invalid_url"
            else:
                return self.async_create_entry(title="Unsloth AI", data=user_input)

        return self.async_show_form(
            step=1, 
            data_schema=vol.Schema({
                vol.Required(CONF_API_URL): str,
                vol.Required(CONF_MODEL_NAME): str,
                vol.Optional(CONF_API_KEY): str,
            }),
            errors=errors,
        )