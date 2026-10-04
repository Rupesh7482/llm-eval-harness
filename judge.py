import ollama
from deepeval.models import DeepEvalBaseLLM


class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt, schema=None):
        kwargs = {}
        if schema is not None:
            kwargs["format"] = schema.model_json_schema()  # force JSON output
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        content = response["message"]["content"]
        if schema is not None:
            return schema.model_validate_json(content)
        return content

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return f"{self.model_name} (Ollama)"