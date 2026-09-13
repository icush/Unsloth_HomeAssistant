"""Constants for the Unsloth AI Assistant integration."""

DOMAIN = "unsloth_ai"

CONF_API_URL = "api_url"
CONF_API_KEY = "api_key"
CONF_MODEL_NAME = "model_name"
CONF_SYSTEM_PROMPT = "system_prompt"
CONF_TEMPERATURE = "temperature"
CONF_MAX_TOKENS = "max_tokens"
CONF_TIMEOUT = "timeout"

DEFAULT_API_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL_NAME = "unsloth/gemma-4-26B-A4B-it-GGUF"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant running inside Home Assistant. "
    "Answer briefly and clearly."
)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512
DEFAULT_TIMEOUT = 120
