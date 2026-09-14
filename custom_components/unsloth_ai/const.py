"""Constants for the Unsloth AI Assistant integration."""

DOMAIN = "unsloth_ai"

CONF_API_URL = "api_url"
CONF_API_KEY = "api_key"
CONF_MODEL_NAME = "model_name"
CONF_PROMPT = "prompt"
CONF_TEMPERATURE = "temperature"
CONF_MAX_TOKENS = "max_tokens"
CONF_TIMEOUT = "timeout"
CONF_THINK = "think"
CONF_MAX_HISTORY = "max_history"

DEFAULT_API_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL_NAME = "unsloth/gemma-4-26B-A4B-it-GGUF"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 120
DEFAULT_THINK = False
DEFAULT_MAX_HISTORY = 5  # previous user turns kept; 0 = unlimited

MAX_TOOL_ITERATIONS = 10
