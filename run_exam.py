import json
import os
import time

from google.genai import errors

from evaluate import ask_gemini, ask_ollama

golden = json.load(open("data/golden.json"))
CACHE = "data/answers.json"
answers = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, say you don't know.
Keep the answer to 1-3 sentences.

CONTEXT:
{context}

QUESTION: {question}"""

MODELS = [("llama3.2 (Ollama)", ask_ollama), ("gemini (cloud)", ask_gemini)]

for name, fn in MODELS:
    for item in golden:
        key = f"{name}|{item['id']}"
        if key in answers:
            continue  # already answered, never spend a call twice

        prompt = PROMPT.format(context=item["context"], question=item["question"])

        result = None
        for attempt in range(4):
            try:
                result = fn(prompt)
                break
            except errors.ServerError:  # temporary Google problem (503), wait and retry
                print(f"  {name} question {item['id']}: server busy, waiting 20s")
                time.sleep(20)
            except errors.APIError as error:  # quota or another real problem
                print(f"{name}: stopped at question {item['id']} (error {error.code}).")
                break

        if result is None:
            print("Finished answers are saved. Run this script again later.")
            break

        answers[key] = {"model": name, "question_id": item["id"], **result}
        with open(CACHE, "w") as f:
            json.dump(answers, f, indent=2)  # saved after every answer
        print(name, "answered question", item["id"], "in", result["total_ms"], "ms")

print(len(answers), "answers saved in total")