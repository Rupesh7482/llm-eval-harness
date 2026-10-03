import os
import time

import ollama
from dotenv import load_dotenv
from google import genai
from google.genai import errors
import pandas as pd

load_dotenv()  # reads GOOGLE_API_KEY from your .env file
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def ask_ollama(prompt, model="llama3.2"):
    start = time.perf_counter()
    first_token_seconds = None
    answer = ""
    last_chunk = None

    for chunk in ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        if first_token_seconds is None:
            first_token_seconds = time.perf_counter() - start
        answer += chunk["message"]["content"]
        last_chunk = chunk

    total_seconds = time.perf_counter() - start
    tokens = last_chunk.get("eval_count") or 0
    return build_result(answer, first_token_seconds, total_seconds, tokens)


def ask_gemini(prompt, model="gemini-3.8-flash"):
    start = time.perf_counter()
    first_token_seconds = None
    answer = ""
    tokens = 0

    for chunk in gemini_client.models.generate_content_stream(
        model=model, contents=prompt
    ):
        if first_token_seconds is None:
            first_token_seconds = time.perf_counter() - start
        answer += chunk.text or ""
        if chunk.usage_metadata and chunk.usage_metadata.candidates_token_count:
            tokens = chunk.usage_metadata.candidates_token_count

    total_seconds = time.perf_counter() - start
    return build_result(answer, first_token_seconds, total_seconds, tokens)


def build_result(answer, first_token_seconds, total_seconds, tokens):
    return {
        "answer": answer,
        "first_token_ms": round(first_token_seconds * 1000),
        "total_ms": round(total_seconds * 1000),
        "tokens": tokens,
        "tokens_per_sec": round(tokens / total_seconds, 1),
    }


if __name__ == "__main__":
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