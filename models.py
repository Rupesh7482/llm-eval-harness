"""Two functions that ask a model a question and time the answer.

ask_ollama  -> Llama 3.2 running on this computer (via Ollama)
ask_gemini  -> Gemini running in Google's cloud
Both return the same dictionary, so the rest of the project treats them the same way.
"""
import os
import time

import ollama
from dotenv import load_dotenv
from google import genai

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