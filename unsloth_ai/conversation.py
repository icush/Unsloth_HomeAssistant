import aiohttp
import logging
from homeassistant.components.conversation import ConversationEntity
from .const import DOMAIN, CONF_API_URL, CONF_MODEL_NAME

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass, entry, async_add_entities):
    async_add_entities([UnslothConversation(hass, entry)])

class UnslothConversation(ConversationEntity):
    def __init__(self, hass, entry):
        self.hass = hass
        self._attr_name = "Unsloth AI"
        self._attr_unique_id = entry.entry_id
        self._api_url = entry.data.get(CONF_API_URL)
        self._model_name = entry.data.get(CONF_MODEL_NAME)
        self._api_key = entry.data.get("api_key")
        self._system_prompt = "You are a helpful, smart Home Assistant. Respond concisely."

    @property
    def supported_conversation_agent_types(self):
        return ["text"]

    async def async_get_conversation_response(self, incoming_message):
        payload = {"model": self._model_name, "messages": [{"role": "system", "content": self._system_prompt}, {"role": "user", "content": incoming_message.text}], "temperature": 0.7}
        headers = {"Content-Type": "application/json"}
        if self._api_key: headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self._api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self.async_wrap_conversation_response(data["choices"][0]["message"]["content"])
                    return self.async_wrap_conversation_response("Error connecting to AI.")
        except Exception as e:
            return self.async_wrap_conversation_response("Connection error.")

    def async_wrap_conversation_response(self, text):
        from homeassistant.components.conversation import ConversationResponse
        return ConversationResponse(text=text)