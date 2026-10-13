"""The first speed test: 5 short questions with no document, sent to both models.

Run from the project folder:   python -m scripts.speed_test
It overwrites data/latency_raw.csv and uses about 12 Gemini requests,
so only run it if you really want to repeat the test.
"""
import time

import pandas as pd
from google.genai import errors

from models import ask_gemini, ask_ollama

PROMPTS = [
    "What is a RAG system? Answer in two sentences.",
    "What does a vector database do? Answer in two sentences.",
    "Explain what an embedding is in two sentences.",
    "What is the difference between precision and recall? Answer in two sentences.",
    "What is chunking in RAG? Answer in two sentences.",
]


def call_with_retry(fn, prompt, attempts=4, wait_seconds=15):
    for attempt in range(1, attempts + 1):
        try:
            return fn(prompt)
        except errors.ServerError as error:  # temporary Google-side problems (5xx)
            print(f"  attempt {attempt} failed: {error.code}")
            if attempt == attempts:
                raise
            time.sleep(wait_seconds)


def run_benchmark():
    rows = []
    models = [("llama3.2 (Ollama)", ask_ollama), ("gemini (cloud)", ask_gemini)]
    for name, fn in models:
        call_with_retry(fn, "Say hello.")  # warm-up call, not recorded
        for i, prompt in enumerate(PROMPTS, start=1):
            result = call_with_retry(fn, prompt)
            rows.append({"model": name, "question_id": i, "prompt": prompt, **result})
            print(name, i, result["total_ms"], "ms")
            pd.DataFrame(rows).to_csv("data/latency_raw.csv", index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_benchmark()