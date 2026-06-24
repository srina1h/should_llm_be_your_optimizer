"""LLM wrappers. API-only build (LiteLLM); local/transformers backend removed
for the reproduction package — all reported runs use an OpenRouter API model."""
import os
from litellm import completion


class LiteLLM_Wrapper:
    """Minimal LangChain-style `.invoke(messages) -> obj.content` over LiteLLM."""
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def invoke(self, input_messages):
        messages = []
        if isinstance(input_messages, str):
            messages = [{"role": "user", "content": input_messages}]
        elif isinstance(input_messages, list):
            for msg in input_messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    messages.append({"role": role, "content": content})
                elif isinstance(msg, tuple):
                    role = msg[0].replace("human", "user")
                    messages.append({"role": role, "content": msg[1]})
                else:
                    messages.append({"role": "user", "content": str(msg)})

        response = completion(model=self.model, messages=messages, **self.kwargs)
        content = response.choices[0].message.content
        return type('Response', (object,), {'content': content})()


class LLM:
    def __init__(self, model_name: str, temperature: float, max_tokens: int, top_p: float, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def __repr__(self):
        return (f'LLM(model={self.model_name}, temperature={self.temperature}, '
                f'max_tokens={self.max_tokens}, top_p={self.top_p})')


class API_LLM(LLM):
    def get_pipeline(self):
        if "gemini" in self.model_name and "GOOGLE_API_KEY" not in os.environ:
            print("Warning: GOOGLE_API_KEY not found.")
        if "gpt" in self.model_name and "OPENAI_API_KEY" not in os.environ:
            print("Warning: OPENAI_API_KEY not found.")
        return LiteLLM_Wrapper(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
