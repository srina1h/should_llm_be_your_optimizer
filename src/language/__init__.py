"""Model loader. API-only build: any OpenRouter / `openai/` model string is
served through LiteLLM. Local/transformers backends are omitted here."""
from src.language.llms import API_LLM, LLM

api_model_path = {
    "gemini": "gemini-1.5-pro",
    "gpt": "gpt-4o",
    "gpt4-turbo": "gpt-4-turbo",
    "claude-opus": "claude-3-opus-20240229",
    "claude-sonnet": "claude-3-sonnet-20240229",
}


def load_model(args, name=None) -> LLM:
    key = name or args.llm
    if key in api_model_path:
        return API_LLM(api_model_path[key], args.temperature, args.max_tokens, args.top_p)
    if key and ("openrouter" in key or "openai/" in key):
        return API_LLM(key, args.temperature, args.max_tokens, args.top_p)
    raise Exception(f"Model not found: {key}. Use an OpenRouter / openai/ model string.")
